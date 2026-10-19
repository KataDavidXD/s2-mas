"""Paired, observed-label QA evaluation; input states are never copied to reports.

Example:
  python paired_metrics.py --test test.jsonl.gz \
    --baseline-predictions baseline_predictions.jsonl --model-predictions predictions.jsonl \
    --output paired_report.json --draws 1000

For an explicitly incomplete model run, add --allow-partial. Every model index must
still have a matching baseline prediction. Full-test metrics are never substituted
for the baseline metrics recomputed on the selected model rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record is not an object at line {line_number}")
            yield line_number, row


def read_test(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Retain only IDs, candidates, gold and cluster membership in memory."""
    records: list[dict[str, Any]] = []
    candidates: dict[str, list[str]] = {}
    for line_number, row in iter_jsonl(path):
        questions = row.get("questions")
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError(f"Test line {line_number}: expected one question")
        qid, question = next(iter(questions.items()))
        criteria = question.get("criteria") if isinstance(question, dict) else None
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"Test line {line_number}: empty or missing candidates")
        labels = list(criteria)
        if any(not isinstance(label, str) for label in labels):
            raise ValueError(f"Test line {line_number}: candidate IDs must be strings")
        expected = row.get("expected")
        if not isinstance(expected, dict) or qid not in expected:
            raise ValueError(f"Test line {line_number}: missing observed label")
        truth = str(expected[qid])
        if truth not in labels:
            raise ValueError(f"Test line {line_number}: observed label is not a candidate")
        if qid in candidates and candidates[qid] != labels:
            raise ValueError(f"Test line {line_number}: candidate order changed within question")
        candidates[qid] = labels
        meta = row.get("meta")
        group = meta.get("group") if isinstance(meta, dict) else None
        if group is None or isinstance(group, (dict, list)) or str(group).strip() == "":
            raise ValueError(f"Test line {line_number}: missing household/respondent group")
        records.append({"qid": qid, "labels": labels, "truth": truth, "group": str(group)})
    if not records:
        raise ValueError("The test file is empty")
    return records, candidates


def read_predictions(path: Path, test: list[dict[str, Any]], name: str) -> dict[int, Any]:
    predictions: dict[int, Any] = {}
    for line_number, row in iter_jsonl(path):
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(test):
            raise ValueError(f"{name} line {line_number}: index is absent or outside the test")
        if index in predictions:
            raise ValueError(f"{name}: duplicate prediction index {index}")
        source = test[index]
        for field in ("qid", "labels", "truth"):
            if row.get(field) != source[field]:
                raise ValueError(f"{name}: {field} mismatch at index {index}")
        if "prediction" not in row:
            raise ValueError(f"{name}: missing prediction field at index {index}")
        # None and any value outside the original candidate IDs count as invalid.
        predictions[index] = row["prediction"]
    if not predictions:
        raise ValueError(f"{name}: predictions are empty")
    return predictions


def summarize(
    truth: np.ndarray,
    prediction: np.ndarray,
    qids: list[str],
    labels: dict[str, list[str]],
    offsets: np.ndarray,
) -> dict[str, Any]:
    classes = int(offsets[-1])
    correct = prediction == truth
    true_count = np.bincount(truth, minlength=classes)
    pred_count = np.bincount(prediction, minlength=classes + 1)[:classes]
    true_positive = np.bincount(truth[correct], minlength=classes)
    denominator = true_count + pred_count
    class_f1 = np.divide(
        2.0 * true_positive, denominator, out=np.zeros(classes), where=denominator != 0
    )
    by_question: dict[str, dict[str, Any]] = {}
    for j, qid in enumerate(qids):
        start, stop = int(offsets[j]), int(offsets[j + 1])
        relevant = (truth >= start) & (truth < stop)
        k = stop - start
        local_true = truth[relevant] - start
        local_pred = np.where(prediction[relevant] < classes, prediction[relevant] - start, k)
        # Predictions are encoded using the matching question's candidates.
        if np.any((local_pred < 0) | (local_pred > k)):
            raise ValueError("Prediction encoding crossed question boundaries")
        confusion = np.bincount(local_true * (k + 1) + local_pred, minlength=k * (k + 1)).reshape(k, k + 1)
        n = int(relevant.sum())
        by_question[qid] = {
            "n": n,
            "valid": int(confusion[:, :k].sum()),
            "accuracy": float(correct[relevant].mean()),
            "macro_f1": float(class_f1[start:stop].mean()),
            "candidate_ids": labels[qid],
            "class_f1": dict(zip(labels[qid], class_f1[start:stop].tolist())),
            "confusion_true_rows_predicted_columns_plus_invalid": confusion.tolist(),
        }
    valid = int((prediction < classes).sum())
    return {
        "n": int(len(truth)), "correct": int(correct.sum()), "valid": valid,
        "invalid": int(len(truth) - valid), "accuracy": float(correct.mean()),
        "mean_question_macro_f1": float(np.mean([x["macro_f1"] for x in by_question.values()])),
        "mean_question_accuracy": float(np.mean([x["accuracy"] for x in by_question.values()])),
        "question_versions": len(qids), "by_question": by_question,
    }


def bootstrap(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    group_ids: np.ndarray,
    offsets: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    classes = int(offsets[-1])
    lengths = np.diff(offsets)
    group_count = int(group_ids.max()) + 1
    rng = np.random.default_rng(seed)
    samples = {name: np.zeros((draws, 2)) for name in predictions}
    correct = {name: prediction == truth for name, prediction in predictions.items()}
    for draw in range(draws):
        counts = np.bincount(rng.integers(0, group_count, size=group_count), minlength=group_count)
        weights = counts[group_ids].astype(float)
        true_count = np.bincount(truth, weights=weights, minlength=classes)
        for name, prediction in predictions.items():
            pred_count = np.bincount(prediction, weights=weights, minlength=classes + 1)[:classes]
            true_positive = np.bincount(truth[correct[name]], weights=weights[correct[name]], minlength=classes)
            denominator = true_count + pred_count
            f1 = np.divide(2 * true_positive, denominator, out=np.zeros(classes), where=denominator != 0)
            samples[name][draw] = (
                float(weights[correct[name]].sum() / weights.sum()),
                float(np.mean(np.add.reduceat(f1, offsets[:-1]) / lengths)),
            )
    return samples


def evaluate(
    test_path: Path,
    baseline_path: Path,
    model_path: Path,
    *,
    allow_partial: bool = False,
    draws: int = 1000,
    seed: int = 20261018,
    baseline_name: str = "baseline",
    model_name: str = "model",
) -> dict[str, Any]:
    started = time.monotonic()
    if draws < 1:
        raise ValueError("--draws must be at least 1")
    if baseline_name == model_name:
        raise ValueError("Model names must differ")
    test, all_labels = read_test(test_path)
    baseline = read_predictions(baseline_path, test, baseline_name)
    model = read_predictions(model_path, test, model_name)
    missing_baseline = set(model).difference(baseline)
    if missing_baseline:
        raise ValueError(f"Baseline predictions do not cover {len(missing_baseline)} model indices")
    full_test = len(model) == len(test) and len(baseline) == len(test)
    if not allow_partial and not full_test:
        raise ValueError("Incomplete coverage; --allow-partial is required for a sample comparison")
    indices = sorted(model)
    qids = sorted({test[index]["qid"] for index in indices})
    labels = {qid: all_labels[qid] for qid in qids}
    offsets = np.r_[0, np.cumsum([len(labels[qid]) for qid in qids])]
    class_start = {qid: int(offsets[j]) for j, qid in enumerate(qids)}
    classes = int(offsets[-1])
    truth = np.array([class_start[test[index]["qid"]] + labels[test[index]["qid"]].index(test[index]["truth"]) for index in indices], dtype=np.int64)
    encoded: dict[str, np.ndarray] = {}
    for name, model_predictions in ((baseline_name, baseline), (model_name, model)):
        values = []
        for index in indices:
            source = test[index]
            prediction = model_predictions[index]
            candidate_ids = labels[source["qid"]]
            # Only an exact string candidate ID is a valid prediction.
            valid = isinstance(prediction, str) and prediction in candidate_ids
            values.append(class_start[source["qid"]] + candidate_ids.index(prediction) if valid else classes)
        encoded[name] = np.array(values, dtype=np.int64)
    names = sorted({test[index]["group"] for index in indices})
    group_map = {name: index for index, name in enumerate(names)}
    group_ids = np.array([group_map[test[index]["group"]] for index in indices], dtype=np.int64)
    selected_by_group = np.bincount(group_ids, minlength=len(names))
    source_by_group = {name: 0 for name in names}
    for record in test:
        if record["group"] in source_by_group:
            source_by_group[record["group"]] += 1
    partially_selected_groups = sum(int(selected_by_group[j]) != source_by_group[name] for j, name in enumerate(names))
    overall = {name: summarize(truth, prediction, qids, labels, offsets) for name, prediction in encoded.items()}
    samples = bootstrap(truth, encoded, group_ids, offsets, draws, seed)
    comparison = {}
    for column, metric in enumerate(("accuracy", "mean_question_macro_f1")):
        difference = samples[model_name][:, column] - samples[baseline_name][:, column]
        lower, upper = np.quantile(difference, [0.025, 0.975])
        point = overall[model_name][metric] - overall[baseline_name][metric]
        comparison[metric] = {
            "model_minus_baseline": point,
            "model_minus_baseline_percentage_points": 100 * point,
            "paired_group_bootstrap_ci95": [float(lower), float(upper)],
            "paired_group_bootstrap_ci95_percentage_points": [float(100 * lower), float(100 * upper)],
        }
    return {
        "status": "complete" if full_test else "partial_sample",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "full_test": full_test, "sample_only": not full_test,
            "partial_mode_requested": allow_partial,
            "test_records": len(test), "paired_records": len(indices),
            "baseline_prediction_records": len(baseline), "model_prediction_records": len(model),
            "coverage_fraction": len(indices) / len(test),
            "test_question_versions": len(all_labels), "paired_question_versions": len(qids),
            "test_groups": len({record["group"] for record in test}),
            "paired_groups": len(names), "partially_selected_groups": partially_selected_groups,
            "sample_selection": "Every available model index, with the baseline evaluated on exactly those indices",
            "interpretation": "Full held-out test comparison" if full_test else "Incomplete selected sample; do not describe as full-test performance",
        },
        "method": {
            "accuracy": "All paired rows equally weighted; invalid outputs count as incorrect",
            "f1": "Macro F1 over every original candidate within each supported question version, zero_division=0; equal mean over those versions",
            "invalid_outputs": "Prediction other than an exact original candidate string; included in the denominator",
            "bootstrap": "Joint resampling with replacement of original meta.group household/respondent IDs, retaining every selected QA row for each sampled group; metrics recomputed in every draw",
            "bootstrap_draws": draws, "seed": seed,
            "bootstrap_question_universe": "Question versions present in this paired sample, kept fixed across draws",
            "comparison_direction": f"{model_name} minus {baseline_name}",
        },
        "models": overall,
        "comparison": comparison,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--model-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path, or a directory for paired_report.json")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--model-name", default="model")
    args = parser.parse_args()
    report = evaluate(args.test, args.baseline_predictions, args.model_predictions,
                      allow_partial=args.allow_partial, draws=args.draws,
                      seed=args.seed, baseline_name=args.baseline_name, model_name=args.model_name)
    destination = args.output if args.output.suffix.lower() == ".json" else args.output / "paired_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"status": report["status"], "scope": report["scope"],
                      "models": {name: {k: v for k, v in result.items() if k != "by_question"} for name, result in report["models"].items()},
                      "comparison": report["comparison"], "output": str(destination),
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

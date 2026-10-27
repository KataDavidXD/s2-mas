"""CPU-only, full-input preparation for the unmodified native Laya renderer.

The returned items have Laya's native ``make_item`` shape plus ordinary row
metadata.  No model weights are loaded.  A measured window preserves the complete input; an insufficient explicit
window is an error.  Native
Laya currently caps an individual option at 48 tokens.  This module measures
uncapped option text and refuses that input if the native cap would drop it.

``prepare_encoded(input_path, model_path, encoded_path, max_len=0,
head_max_len=0)`` writes a torch CPU package and returns it.  Zero budgets mean
"choose a sufficient window from this split".  Test files are only opened when
the caller explicitly supplies their path.  ``--measure-only`` writes only
aggregate token statistics and never retains respondent text in its report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time


INPUT_POLICY = "native-full-state-question-options-v1"
NATIVE_OPTION_CAP = 48


def ensure_full_text_renderer():
    """Describe the native guard; do not monkeypatch Laya or its model.

    The measured-option guard in ``prepare_encoded`` is mandatory.  Train and
    evaluation must use the package's max_len/head_max_len with the same native
    renderer; calling this function does not make a 48-token cap lossless.
    """
    return {"renderer": "laya.common.build_sequence", "option_token_cap": NATIVE_OPTION_CAP,
            "requires_measured_option_guard": True, "input_policy": INPUT_POLICY}


def _imports():
    import torch
    from transformers import AutoTokenizer
    from laya.common import (QTYPES, build_head, build_sequence, render_options,
                             serialize_state)
    from laya.train import make_item, target_from_expected, target_from_gold, to_internal
    return (torch, AutoTokenizer, QTYPES, build_head, build_sequence, render_options,
            serialize_state, make_item, target_from_expected, target_from_gold, to_internal)


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _ordinary_file_info(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns}


def _same_ordered(left, right):
    """Compare retained native values including all rendered mapping order."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (list(left) == list(right)
                and all(_same_ordered(left[key], right[key]) for key in left))
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_ordered(a, b) for a, b in zip(left, right))
    return left == right


def _tokenizer_config_files(model_path):
    """Ordinary input-cache compatibility metadata, without derived file IDs."""
    tok_dir = model_path / "tokenizer" if (model_path / "tokenizer").is_dir() else model_path
    files = [value for value in tok_dir.iterdir() if value.is_file()
             and (value.suffix in {".json", ".txt", ".model", ".vocab"}
                  or value.name in {"vocab", "merges"})]
    files += [value for value in (model_path / "encoder" / "config.json", model_path / "config.json")
              if value.is_file() and value not in files]
    return [_ordinary_file_info(value) for value in sorted(files)]


def _load_tokenizer(model_path, auto_tokenizer):
    tok_path = model_path / "tokenizer" if (model_path / "tokenizer").is_dir() else model_path
    tok = auto_tokenizer.from_pretrained(str(tok_path), local_files_only=True)
    required = ("cls_token_id", "sep_token_id", "mask_token_id", "mask_token")
    if any(getattr(tok, key, None) is None for key in required):
        raise ValueError("Native tokenizer lacks a required marker token")
    return tok


def _normalise_tokenizer_metadata(model_path):
    # Use the same compatibility normalization as native load_checkpoint before
    # recording tokenizer metadata in a reusable input cache.
    from laya.agent import _fix_tokenizer_config
    _fix_tokenizer_config(str(model_path))


def _position_limit(model_path):
    for candidate in (model_path / "encoder" / "config.json", model_path / "config.json"):
        if candidate.is_file():
            cfg = json.loads(candidate.read_text(encoding="utf-8"))
            value = cfg.get("max_position_embeddings")
            if isinstance(value, int) and value > 0:
                return value
    raise ValueError("Model config must declare max_position_embeddings")


def _native_row(row, prepared_qwen=False):
    if prepared_qwen:
        required = ("row_id", "group_id", "task_family", "qid", "state", "question", "candidate_ids", "target")
        if any(key not in row for key in required):
            raise ValueError("Prepared row lacks a required native field")
        candidates, target = row["candidate_ids"], row["target"]
        if (not isinstance(candidates, list) or len(candidates) < 2
                or len(set(candidates)) != len(candidates)
                or isinstance(target, bool) or not isinstance(target, int)
                or not 0 <= target < len(candidates)):
            raise ValueError("Prepared row has invalid candidates or target")
        if list(row["question"].get("criteria", {})) != candidates:
            raise ValueError("Prepared candidate order differs from native criteria")
        return {"state": row["state"], "questions": {row["qid"]: row["question"]},
                "expected": {row["qid"]: candidates[target]},
                "meta": {"row_id": row["row_id"], "group": row["group_id"],
                         "task_family": row["task_family"]}}
    return row


def _summary(histogram):
    total = sum(histogram.values())
    if not total:
        return {"count": 0, "min": None, "max": None, "mean": None}
    ordered = sorted(histogram.items())
    result = {"count": total, "min": ordered[0][0], "max": ordered[-1][0],
              "mean": sum(key * value for key, value in ordered) / total}
    for label, percentile in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
        threshold, cumulative = max(1, int(total * percentile + .999999)), 0
        for value, count in ordered:
            cumulative += count
            if cumulative >= threshold:
                result[label] = value
                break
    return result


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _prepare(input_path, model_path, encoded_path, max_len, head_max_len,
             *, measure_only=False, prepared_qwen=False, batch_size=512):
    started = time.monotonic()
    input_path, model_path, encoded_path = map(Path, (input_path, model_path, encoded_path))
    _normalise_tokenizer_metadata(model_path)
    if max_len < 0 or head_max_len < 0 or batch_size < 1:
        raise ValueError("Budgets must be nonnegative and batch_size must be positive")
    (torch, auto_tokenizer, qtypes, build_head, build_sequence, render_options,
     serialize_state, make_item, target_expected, target_gold, to_internal) = _imports()
    tok = _load_tokenizer(model_path, auto_tokenizer)
    encoder_limit = _position_limit(model_path)
    source = _ordinary_file_info(input_path)
    hist = {key: Counter() for key in ("state_tokens", "instruction_tokens", "option_tokens",
                                      "head_tokens", "sequence_tokens", "options_per_row")}
    items, pending, question_cache = [], [], {}
    rows = 0
    long_options = 0
    changed_mask_occurrences = 0
    max_required_head = 0
    max_required_sequence = 0
    checks = {"input_rows": 0, "items": 0, "option_collapse": 0,
              "missing_markers": 0, "state_tokens_dropped": 0,
              "instruction_tokens_dropped": 0, "option_tokens_dropped": 0}

    def flush():
        nonlocal max_required_sequence
        if not pending:
            return
        texts = [entry[0] for entry in pending]
        token_batches = tok(texts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        for (_, item, qinfo), state_ids in zip(pending, token_batches):
            length = len(state_ids)
            sequence_len = qinfo["head_tokens"] + length + 1
            hist["state_tokens"][length] += 1
            hist["sequence_tokens"][sequence_len] += 1
            max_required_sequence = max(max_required_sequence, sequence_len)
            if not measure_only:
                item["state_ids"] = state_ids
                items.append(item)
        pending.clear()

    with _open(input_path) as source_file:
        for position, line in enumerate(source_file):
            if not line.strip():
                raise ValueError(f"Blank row at input position {position}")
            try:
                row = _native_row(json.loads(line), prepared_qwen)
                state, questions = row["state"], row["questions"]
                if state is None or not isinstance(questions, dict) or len(questions) != 1:
                    raise ValueError("Exactly one native question and a state are required per row")
                qid, external_q = next(iter(questions.items()))
                meta = row.get("meta", {})
                row_id, group = meta.get("row_id"), meta.get("group", meta.get("group_id"))
                if not isinstance(row_id, str) or not row_id or not isinstance(group, str) or not group:
                    raise ValueError("Ordinary row_id and group metadata are required")
                cached = question_cache.get(qid)
                if cached is None or not _same_ordered(cached[0], external_q):
                    q = to_internal(qid, external_q)
                    options = render_options(q)
                    instruction = f"{q['t']} question: {q['ins']}"
                    changed_mask_occurrences += instruction.count(tok.mask_token)
                    changed_mask_occurrences += sum(option.count(tok.mask_token) for option in options)
                    instruction_ids = tok(instruction.replace(tok.mask_token, " "), add_special_tokens=False,
                                          truncation=False)["input_ids"]
                    option_ids = tok([" " + option.replace(tok.mask_token, " ") for option in options],
                                     add_special_tokens=False, truncation=False, padding=False)["input_ids"]
                    if len({tuple(value) for value in option_ids}) != len(option_ids):
                        checks["option_collapse"] += 1
                        raise ValueError("Distinct native options do not have distinct full token spans")
                    option_spans = sum(len(value) + 1 for value in option_ids)
                    qinfo = {"q": q, "instruction_tokens": len(instruction_ids),
                             "option_lengths": [len(value) for value in option_ids],
                             "head_tokens": 3 + len(instruction_ids) + option_spans,
                             "required_head_budget": option_spans + max(16, len(instruction_ids)),
                             "head_ids_full": [tok.cls_token_id] + instruction_ids + [tok.sep_token_id]}
                    markers = []
                    for value in option_ids:
                        markers.append(len(qinfo["head_ids_full"]))
                        qinfo["head_ids_full"] += [tok.mask_token_id] + value
                    qinfo["head_ids_full"].append(tok.sep_token_id)
                    qinfo["markers_full"] = markers
                    question_cache[qid] = external_q, qinfo
                else:
                    qinfo = cached[1]
                q = qinfo["q"]
                k = len(qinfo["option_lengths"])
                expected = row.get("expected", {})
                gold = row.get("gold", {})
                if isinstance(gold, dict) and qid in gold:
                    target = target_gold(q, gold[qid])
                elif isinstance(expected, dict) and qid in expected:
                    target = target_expected(q, expected[qid])
                else:
                    raise ValueError("Native row lacks its observed target")
                if len(target) != k:
                    raise ValueError("Target and candidate counts differ")
                state_text = serialize_state(state)
                changed_mask_occurrences += state_text.count(tok.mask_token)
                state_text = state_text.replace(tok.mask_token, " ")
                if not state_text.strip():
                    raise ValueError("State is empty")
                item = {"q": q, "target": [float(value) for value in target],
                        "qtype": qtypes[q["t"]], "k": k,
                        "meta": {"row_id": row_id, "group": group, "group_id": group,
                                 "task_family": meta.get("task_family"),
                                 "qid": str(qid), "index": position}}
                hist["instruction_tokens"][qinfo["instruction_tokens"]] += 1
                hist["head_tokens"][qinfo["head_tokens"]] += 1
                hist["options_per_row"][k] += 1
                for length in qinfo["option_lengths"]:
                    hist["option_tokens"][length] += 1
                    long_options += int(length > NATIVE_OPTION_CAP)
                max_required_head = max(max_required_head, qinfo["required_head_budget"])
                pending.append((state_text, item, qinfo))
                rows += 1
            except (ValueError, TypeError, KeyError, IndexError) as error:
                # Never include respondent fields, observed labels, or upstream error text.
                raise ValueError(f"Invalid native input at position {position} ({type(error).__name__})") from None
            if len(pending) >= batch_size:
                flush()
    flush()
    if not rows:
        raise ValueError("Input split is empty")
    chosen_head = head_max_len or max_required_head
    chosen_length = max_len or max_required_sequence
    settings = {"max_len": chosen_length, "head_max_len": chosen_head,
                "input_policy": INPUT_POLICY, "native_option_token_cap": NATIVE_OPTION_CAP,
                "encoder_position_limit": encoder_limit}
    violations = []
    if long_options:
        violations.append("native_option_cap_would_drop_tokens")
    if changed_mask_occurrences:
        violations.append("native_mask_sanitization_would_change_input")
    if chosen_head < max_required_head:
        violations.append("head_window_is_too_small")
    if chosen_length < max_required_sequence:
        violations.append("sequence_window_is_too_small")
    if chosen_length > encoder_limit:
        violations.append("sequence_exceeds_encoder_position_limit")
    report = {"input_policy": INPUT_POLICY, "measurement_only": measure_only, "source": source,
              "model_path": str(model_path.resolve()), "model_name": model_path.name,
              "tokenizer_config_files": _tokenizer_config_files(model_path),
              "rows": rows, "settings": settings,
              "requested_windows": {"max_len": max_len, "head_max_len": head_max_len},
              "required_windows": {"max_len": max_required_sequence, "head_max_len": max_required_head},
              "length_stats": {key: _summary(value) for key, value in hist.items()},
              "option_spans_above_native_cap": long_options,
              "native_mask_sanitization_occurrences": changed_mask_occurrences,
              "violations": violations, "checks": checks,
              "elapsed_seconds": round(time.monotonic() - started, 3)}
    checks["input_rows"] = rows
    checks["items"] = len(items)
    _atomic_json(encoded_path.with_suffix(encoded_path.suffix + ".lengths.json"), report)
    if measure_only:
        return report
    if violations:
        raise ValueError("Full-input native preparation refused: " + ", ".join(violations)
                         + "; see aggregate .lengths.json report")
    # Verify the native construction is exactly the uncapped rendering measured
    # above.  The head is shared by ordinary qid, so one full check per version
    # suffices; state preservation is checked for every retained item.
    verified_q = {}
    verified_versions = 0
    for item in items:
        qid = item["meta"]["qid"]
        q = item["q"]
        cached = verified_q.get(qid)
        if cached is None or not _same_ordered(cached[0], q):
            head, markers, stats = build_head(tok, q, chosen_head)
            instruction = f"{q['t']} question: {q['ins']}"
            full_instruction = tok(instruction, add_special_tokens=False, truncation=False)["input_ids"]
            full_options = tok([" " + value for value in render_options(q)], add_special_tokens=False,
                               truncation=False, padding=False)["input_ids"]
            full_head = [tok.cls_token_id] + full_instruction + [tok.sep_token_id]
            full_markers = []
            for value in full_options:
                full_markers.append(len(full_head))
                full_head += [tok.mask_token_id] + value
            full_head.append(tok.sep_token_id)
            if head != full_head or markers != full_markers:
                raise ValueError("Native head construction differs from the measured full rendering")
            prototype, reason = make_item(tok, q, item["target"], [], chosen_head, chosen_length)
            if reason or prototype is None or stats["options_distinct"] != item["k"]:
                raise ValueError("Native item construction does not preserve all options")
            verified_q[qid] = q, len(head)
            verified_versions += 1
        head_length = verified_q[qid][1]
        if head_length + len(item["state_ids"]) + 1 > chosen_length:
            raise ValueError("State tokens would be lost in the chosen sequence window")
    # Native build_sequence checks on first/last examples also cover assembly,
    # rather than relying only on budget arithmetic.
    for item in (items[0], items[-1]):
        ids, markers, stats, truncation = build_sequence(
            tok, "", item["q"], max_len=chosen_length, head_max_len=chosen_head,
            state_ids=item["state_ids"], return_stats=True, return_truncation_stats=True)
        head_length = len(build_head(tok, item["q"], chosen_head)[0])
        if (truncation["state_tokens_dropped"] or len(markers) != item["k"]
                or len(ids) != head_length + len(item["state_ids"]) + 1):
            raise ValueError("Native sequence assembly failed its full-input check")
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["checks"]["verified_question_versions"] = verified_versions
    package = {"items": items, **report}
    encoded_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = encoded_path.with_name(encoded_path.name + ".tmp")
    torch.save(package, temporary)
    os.replace(temporary, encoded_path)
    _atomic_json(encoded_path.with_suffix(encoded_path.suffix + ".lengths.json"), report)
    return package


def prepare_encoded(input_path: Path, model_path: Path, encoded_path: Path,
                    max_len: int = 0, head_max_len: int = 0) -> dict:
    """Prepare/reuse a CPU native cache, refusing any loss of input tokens."""
    input_path, model_path, encoded_path = map(Path, (input_path, model_path, encoded_path))
    _normalise_tokenizer_metadata(model_path)
    if encoded_path.is_file():
        import torch
        value = torch.load(encoded_path, map_location="cpu", weights_only=False)
        settings = value.get("settings", {})
        if (value.get("source") != _ordinary_file_info(input_path)
                or value.get("model_path") != str(model_path.resolve())
                or value.get("tokenizer_config_files") != _tokenizer_config_files(model_path)
                or value.get("input_policy") != INPUT_POLICY
                or value.get("requested_windows") != {"max_len": max_len, "head_max_len": head_max_len}
                or value.get("rows") != len(value.get("items", []))
                or settings.get("input_policy") != INPUT_POLICY
                or settings.get("encoder_position_limit") != _position_limit(model_path)
                or value.get("violations")):
            raise ValueError("Existing encoded cache does not match this ordinary input/model/window configuration")
        return value
    return _prepare(input_path, model_path, encoded_path, max_len, head_max_len)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=0)
    parser.add_argument("--head-max-len", type=int, default=0)
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--prepared-qwen", action="store_true",
                        help="Read retained Qwen rows using their unchanged native fields")
    args = parser.parse_args()
    if args.measure_only or args.prepared_qwen:
        result = _prepare(args.input, args.model, args.output, args.max_len, args.head_max_len,
                          measure_only=args.measure_only, prepared_qwen=args.prepared_qwen)
    else:
        result = prepare_encoded(args.input, args.model, args.output, args.max_len, args.head_max_len)
    print(json.dumps({key: value for key, value in result.items() if key != "items"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

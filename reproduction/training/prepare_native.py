"""Map the existing Qwen samples to Laya native choice rows, without resampling.

The Qwen preparation retains the complete state, question, ordered candidate IDs,
observed target, task_family and group. The mapping reconstructs those fields and
copies the caller-provided ordinary row IDs without interpreting them.
Shared-test input is opened only
when ``--prepare-test`` is supplied; it is never used for sample or method choice.

Example::

    python prepare_native.py --source-dir /path/to/qwen/data --output /path/to/laya/data

To append the final test after train/dev preparation, use ``--prepare-test
--test-only`` with the same output directory.  Existing train/dev files are read
for ordinary ID/group checks and are never rewritten.

Only the output JSONL files and an aggregate prepare_report.json are retained.
Temporary SQLite tables hold IDs and nested-sample rows on disk rather than
retaining all respondent text in memory.  This script uses the Python standard
library and does not import any model or GPU package.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time


@dataclass(frozen=True)
class Split:
    filename: str
    partition: str
    expected_rows: int
    child: str | None = None
    retain_rows: bool = False


SPLITS = (
    Split("train.50000.jsonl", "train", 50000, retain_rows=True),
    Split("train.200000.jsonl", "train", 200000,
          child="train.50000.jsonl"),
    Split("dev.20000.jsonl", "dev", 20000, retain_rows=True),
    Split("dev.full.jsonl", "dev", 412873,
          child="dev.20000.jsonl"),
)
TEST = Split("test.shared119.jsonl", "test", 240458)


def dump(value):
    # Do not sort keys: criteria insertion order is part of the prepared input.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def nonempty_text(value, field):
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {"none", "null"}:
        raise ValueError(f"Missing or invalid {field}")
    return value


def to_native(row):
    required = {"row_id", "group_id", "task_family", "qid", "state", "question", "candidate_ids", "target"}
    if not isinstance(row, dict) or not required.issubset(row):
        raise ValueError("Prepared row is missing required fields")
    row_id = nonempty_text(row["row_id"], "row_id")
    group = nonempty_text(row["group_id"], "group_id")
    task_family = nonempty_text(row["task_family"], "task_family")
    qid = nonempty_text(row["qid"], "qid")
    question = row["question"]
    if not isinstance(question, dict) or question.get("type") != "choice":
        raise ValueError("This mapping requires a native choice question")
    criteria = question.get("criteria")
    candidates = row["candidate_ids"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("At least two candidate IDs are required")
    if any(not isinstance(item, str) or not item for item in candidates):
        raise ValueError("Candidate IDs must be nonempty strings")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate candidate ID")
    if not isinstance(criteria, dict) or list(criteria) != candidates:
        raise ValueError("Ordered candidate IDs differ from question criteria")
    target = row["target"]
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(candidates):
        raise ValueError("Observed target is outside the candidate IDs")
    if row["state"] is None:
        raise ValueError("Missing native respondent state")
    mapped = {
        "state": row["state"],
        "questions": {qid: question},
        "expected": {qid: candidates[target]},
        "meta": {"task_family": task_family, "group": group,
                 "row_id": row_id, "language": "en"},
    }
    return mapped


def iter_prepared(path):
    with path.open("r", encoding="utf-8") as source:
        for position, line in enumerate(source):
            if not line.strip():
                raise ValueError(f"Blank prepared row in {path.name} at position {position}")
            try:
                row = json.loads(line)
                mapped = to_native(row)
            except (ValueError, TypeError, KeyError) as error:
                # Error context contains positions, never respondent text or labels.
                raise ValueError(f"Invalid prepared row in {path.name} at position {position}: {error}") from None
            yield position, mapped


def same_retained_fields(left, right):
    """Compare actual retained values and candidate order, with no derived ID."""
    if left != right:
        return False
    qid = next(iter(left["questions"]))
    return list(left["questions"][qid]["criteria"]) == list(right["questions"][qid]["criteria"])


def add_index(connection, split, position, mapped, payload=None):
    try:
        connection.execute(
            "INSERT INTO rows VALUES(?,?,?,?,?)",
            (split.filename, mapped["meta"]["row_id"], position, payload, split.partition),
        )
    except sqlite3.IntegrityError:
        raise ValueError(f"Duplicate ordinary row ID in {split.filename}") from None
    connection.execute("INSERT OR IGNORE INTO groups VALUES(?,?)",
                       (split.partition, mapped["meta"]["group"]))


def check_child_row(connection, split, mapped):
    if split.child is None:
        return
    previous = connection.execute(
        "SELECT payload FROM rows WHERE dataset=? AND row_id=?",
        (split.child, mapped["meta"]["row_id"]),
    ).fetchone()
    if previous and not same_retained_fields(json.loads(previous[0]), mapped):
        raise ValueError(f"A nested sample row changed its native input or label in {split.filename}")


def check_containment(connection, split):
    if split.child is None:
        return None
    missing = connection.execute(
        "SELECT COUNT(*) FROM rows AS child LEFT JOIN rows AS parent "
        "ON parent.dataset=? AND parent.row_id=child.row_id "
        "WHERE child.dataset=? AND parent.row_id IS NULL",
        (split.filename, split.child),
    ).fetchone()[0]
    if missing:
        raise ValueError(f"Nested sample is not fully contained in {split.filename}")
    return True


def map_split(connection, source_dir, staging, split, expected_rows):
    source = source_dir / split.filename
    output = staging / split.filename
    started = time.monotonic()
    task_families = Counter()
    question_strata = set()
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for position, mapped in iter_prepared(source):
            check_child_row(connection, split, mapped)
            encoded = dump(mapped)
            add_index(connection, split, position, mapped,
                      encoded if split.retain_rows else None)
            destination.write(encoded + "\n")
            count += 1
            task_family = mapped["meta"]["task_family"]
            task_families[task_family] += 1
            question_strata.add((task_family, next(iter(mapped["questions"]))))
            if count % 4000 == 0:
                connection.commit()
    connection.commit()
    if count != expected_rows:
        raise ValueError(f"Unexpected row count in {split.filename}: {count}; expected {expected_rows}")
    contained = check_containment(connection, split)
    return {"rows": count, "task_family_counts": dict(task_families),
            "question_strata": len(question_strata), "bytes": output.stat().st_size,
            "nested_child_contained": contained,
            "elapsed_seconds": round(time.monotonic() - started, 2)}


def index_existing_split(connection, output, split, expected_rows, previous):
    """Read completed native files for ID/group checks without writing to them."""
    path = output / split.filename
    if not path.is_file() or previous.get("rows") != expected_rows or previous.get("bytes") != path.stat().st_size:
        raise ValueError(f"Existing output/report is incomplete for {split.filename}")
    count = 0
    with path.open("r", encoding="utf-8") as source:
        for position, line in enumerate(source):
            try:
                raw = json.loads(line)
                questions = raw["questions"]
                if not isinstance(questions, dict) or len(questions) != 1:
                    raise ValueError("Expected one native question")
                qid, question = next(iter(questions.items()))
                labels = list(question["criteria"])
                truth = raw["expected"][qid]
                meta = raw["meta"]
                row = {"row_id": meta["row_id"], "group_id": meta["group"],
                       "task_family": meta["task_family"], "qid": qid, "state": raw["state"],
                       "question": question, "candidate_ids": labels, "target": labels.index(truth)}
                mapped = to_native(row)
            except (ValueError, TypeError, KeyError, AttributeError):
                raise ValueError(f"Invalid existing native row in {split.filename} at position {position}") from None
            check_child_row(connection, split, mapped)
            add_index(connection, split, position, mapped,
                      dump(mapped) if split.retain_rows else None)
            count += 1
            if count % 4000 == 0:
                connection.commit()
    connection.commit()
    if count != expected_rows:
        raise ValueError(f"Existing output row count differs for {split.filename}")
    check_containment(connection, split)


def prepare(source_dir, output, prepare_test=False, *,
            test_only=False, expected_counts=None, expected_test_strata=119):
    """Perform the mapping. Count overrides are for small synthetic verification."""
    source_dir = Path(source_dir)
    output = Path(output)
    if test_only and not prepare_test:
        raise ValueError("--test-only requires --prepare-test")
    all_specs = list(SPLITS) + ([TEST] if prepare_test else [])
    specs = [TEST] if test_only else all_specs
    counts = {split.filename: split.expected_rows for split in all_specs}
    if expected_counts is not None:
        if set(expected_counts) != set(counts):
            raise ValueError("Synthetic expected-count overrides must cover exactly the requested files")
        counts = dict(expected_counts)
    output.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() == output.resolve():
        raise ValueError("Source and output directories must differ")
    for name in [split.filename for split in specs] + ([] if test_only else ["prepare_report.json"]):
        if (output / name).exists():
            raise ValueError("An output already exists; use a new output directory")
    for split in specs:
        if not (source_dir / split.filename).is_file():
            raise ValueError(f"Missing prepared input file: {split.filename}")
    started = time.monotonic()
    report = {
        "status": "preparing", "format": "Laya native single-question choice JSONL",
        "mapping": "reconstructed_from_complete_prepared_fields",
        "ordinary_row_ids": "Caller-provided opaque IDs retained without decoding or external lookup",
        "selection": "Unchanged Qwen 50k/200k train rows, fixed dev20k, and full dev",
        "test_prepared": bool(prepare_test), "test_rows_used_for_training_or_selection": 0,
        "splits": {},
    }
    if test_only:
        report_path = output / "prepare_report.json"
        if not report_path.is_file():
            raise ValueError("--test-only requires a completed existing prepare_report.json")
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if (previous.get("status") != "complete" or previous.get("test_prepared")
                or previous.get("train_dev_group_overlap") != 0
                or previous.get("nested_train_samples") is not True
                or previous.get("fixed_dev_sample_contained_in_full_dev") is not True
                or set(previous.get("splits", {})) != {split.filename for split in SPLITS}):
            raise ValueError("--test-only requires the complete four-file train/dev preparation")
        previous["test_mapping"] = report["mapping"]
        report = previous
    with tempfile.TemporaryDirectory(prefix="laya-map-", dir=output) as temporary:
        staging = Path(temporary)
        connection = sqlite3.connect(staging / "rows.sqlite")
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("CREATE TABLE rows(dataset TEXT,row_id TEXT,position INTEGER,payload TEXT,partition TEXT,PRIMARY KEY(dataset,row_id),UNIQUE(dataset,position))")
            connection.execute("CREATE TABLE groups(partition TEXT,group_id TEXT,PRIMARY KEY(partition,group_id))")
            if test_only:
                for split in SPLITS:
                    index_existing_split(connection, output, split, counts[split.filename],
                                         report["splits"][split.filename])
            for split in specs:
                item = map_split(connection, source_dir, staging, split, counts[split.filename])
                if split.partition == "test" and item["question_strata"] != expected_test_strata:
                    raise ValueError("Shared final-test question count differs from the existing 119-question benchmark")
                report["splits"][split.filename] = item
                print(dump({"phase": "mapped", "file": split.filename, "rows": item["rows"],
                            "elapsed_seconds": item["elapsed_seconds"]}), flush=True)
            overlap = connection.execute(
                "SELECT COUNT(*) FROM groups AS train JOIN groups AS dev "
                "ON train.group_id=dev.group_id WHERE train.partition='train' AND dev.partition='dev'"
            ).fetchone()[0]
            if overlap:
                raise ValueError(f"Train/dev group overlap: {overlap}")
            report["train_dev_group_overlap"] = overlap
            if prepare_test:
                for partition in ("train", "dev"):
                    overlap = connection.execute(
                        "SELECT COUNT(*) FROM groups AS chosen JOIN groups AS final_test "
                        "ON chosen.group_id=final_test.group_id "
                        "WHERE chosen.partition=? AND final_test.partition='test'",
                        (partition,),
                    ).fetchone()[0]
                    if overlap:
                        raise ValueError(f"{partition.capitalize()}/test group overlap: {overlap}")
                    report[f"{partition}_test_group_overlap"] = overlap
            report["nested_train_samples"] = True
            report["fixed_dev_sample_contained_in_full_dev"] = True
        finally:
            connection.close()
        report["status"] = "complete"
        if test_only:
            report["test_prepared"] = True
            report["test_added_after_train_dev"] = True
            report["test_preparation_elapsed_seconds"] = round(time.monotonic() - started, 2)
        else:
            report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (staging / "prepare_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Publish only after every requested split and group check succeeded.
        for split in specs:
            os.replace(staging / split.filename, output / split.filename)
        os.replace(staging / "prepare_report.json", output / "prepare_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="Existing Qwen prepared-data directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for Laya native files")
    parser.add_argument("--prepare-test", action="store_true", help="Explicitly map the shared final-only 119-question test")
    parser.add_argument("--test-only", action="store_true", help="With --prepare-test, append test to an existing completed preparation without rewriting train/dev")
    args = parser.parse_args()
    if args.test_only and not args.prepare_test:
        parser.error("--test-only requires --prepare-test")
    report = prepare(args.source_dir, args.output, args.prepare_test,
                     test_only=args.test_only)
    print(dump({"phase": "complete", "test_prepared": report["test_prepared"],
                "elapsed_seconds": report["test_preparation_elapsed_seconds"] if args.test_only else report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

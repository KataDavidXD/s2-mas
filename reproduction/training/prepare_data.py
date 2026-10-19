"""Prepare existing English frozen splits and nested decision-head learning curves.

Callers supply ordinary row, question, task and group IDs. Test conversion is
an optional final-only operation and never participates in train/dev selection.
"""
import argparse
import gzip
import json
import math
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def normal_parts(path, split):
    original = json.loads(path.read_text(encoding="utf-8"))
    allowed = ("language", "split", "path", "records")
    parts = [{key: item.get(key) for key in allowed} for item in original
             if item.get("language") == "en" and item.get("split") == split]
    for part in parts:
        if not Path(part["path"]).is_absolute():
            part["path"] = str(path.parent / part["path"])
    return parts


def convert(raw):
    questions = raw["questions"]
    if len(questions) != 1:
        raise ValueError("Expected one observed question per native row")
    qid, question = next(iter(questions.items()))
    labels = list(question["criteria"])
    truth = raw["expected"][qid]
    if isinstance(truth, dict):
        truth = truth.get("choice")
    truth = str(truth)
    if not labels or truth not in labels:
        raise ValueError("Observed answer is outside candidate IDs")
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate candidate ID")
    meta = raw["meta"]
    values = {key: meta[key] for key in ("row_id", "group", "task_family")}
    for key, value in values.items():
        if (not isinstance(value, str) or not value.strip()
                or value.strip().lower() in {"none", "null"}):
            raise ValueError(f"Missing or invalid {key}")
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError("Missing question ID")
    return {"row_id": values["row_id"], "group_id": values["group"],
            "task_family": values["task_family"], "qid": qid, "state": raw["state"],
            "question": question, "candidate_ids": labels,
            "target": labels.index(truth)}


def quotas(total, counts):
    """Proportional allocation with at least one row in each nonempty stratum."""
    if total > sum(counts.values()) or total < len(counts):
        raise ValueError("Requested sample cannot cover the available strata")
    denominator = sum(counts.values())
    ideals = {key: total * count / denominator for key, count in counts.items()}
    result = {key: min(counts[key], max(1, math.floor(value)))
              for key, value in ideals.items()}
    while sum(result.values()) > total:
        key = max((key for key in result if result[key] > 1),
                  key=lambda key: (result[key] - ideals[key], result[key], key))
        result[key] -= 1
    while sum(result.values()) < total:
        key = max((key for key in result if result[key] < counts[key]),
                  key=lambda key: (ideals[key] - result[key], counts[key], key))
        result[key] += 1
    return result


def select_nested(connection, split, counts, sizes):
    sizes = sorted(sizes)
    previous = {key: 0 for key in counts}
    allocations = {}
    for size in sizes:
        if not allocations:
            allocation = quotas(size, counts)
        else:
            remaining = {key: counts[key] - previous[key] for key in counts
                         if counts[key] > previous[key]}
            extra = quotas(size - sum(previous.values()), remaining)
            allocation = {key: previous[key] + extra.get(key, 0) for key in counts}
        allocations[size] = allocation
        previous = allocation
    chosen = {size: set() for size in sizes}
    for (task_family, qid), largest in allocations[sizes[-1]].items():
        rows = connection.execute(
            "SELECT ordinal FROM rows WHERE split=? AND task_family=? AND qid=? ORDER BY score,ordinal LIMIT ?",
            (split, task_family, qid, largest)).fetchall()
        for size in sizes:
            chosen[size].update(row[0] for row in rows[:allocations[size][(task_family, qid)]])
    for size in sizes:
        if len(chosen[size]) != size:
            raise ValueError("Sample size mismatch")
    return chosen, {str(size): {f"{task_family}:{qid}": count for (task_family, qid), count in allocation.items()}
                    for size, allocation in allocations.items()}


def insert_rows(connection, rows):
    try:
        connection.executemany("INSERT INTO rows VALUES(?,?,?,?,?,?)", rows)
    except sqlite3.IntegrityError:
        raise ValueError("Duplicate ordinary row ID within a partition") from None


def scan_split(manifest, split, output, connection, seed):
    rng = random.Random(seed)
    parts = normal_parts(manifest, split)
    if not parts:
        raise ValueError("No native files for requested split")
    paths = [Path(part["path"]).resolve() for part in parts]
    if len(paths) != len(set(paths)):
        raise ValueError("An input file is listed more than once")
    counts = Counter()
    task_family_counts = Counter()
    groups = set()
    part_reports = []
    index = 0
    started = time.monotonic()
    full_path = output / f"{split}.full.jsonl"
    with full_path.open("w", encoding="utf-8") as destination:
        for part_index, part in enumerate(parts):
            local_count = 0
            batch = []
            with gzip.open(part["path"], "rt", encoding="utf-8") as source:
                for line in source:
                    row = convert(json.loads(line))
                    destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[(row["task_family"], row["qid"])] += 1
                    task_family_counts[row["task_family"]] += 1
                    groups.add(row["group_id"])
                    batch.append((split, index, row["row_id"], row["task_family"], row["qid"], rng.random()))
                    local_count += 1
                    index += 1
                    if len(batch) >= 4000:
                        insert_rows(connection, batch)
                        connection.commit()
                        batch = []
                    if index % 100000 == 0:
                        log(phase="convert", split=split, rows=index,
                            elapsed_seconds=round(time.monotonic() - started, 2))
            if batch:
                insert_rows(connection, batch)
                connection.commit()
            if local_count != part["records"]:
                raise ValueError("Native row count differs from declared records")
            part_reports.append({"part": part_index, "rows": local_count})
    return counts, groups, {"rows": index, "task_family_counts": dict(task_family_counts),
                           "question_strata": len(counts), "groups": len(groups),
                           "parts": part_reports, "path": str(full_path),
                           "bytes": full_path.stat().st_size,
                           "conversion_seconds": round(time.monotonic() - started, 2)}


def save_samples(split, output, chosen):
    paths = {size: output / f"{split}.{size}.jsonl" for size in chosen}
    handles = {size: path.open("w", encoding="utf-8") for size, path in paths.items()}
    written = Counter()
    try:
        with (output / f"{split}.full.jsonl").open(encoding="utf-8") as source:
            for ordinal, line in enumerate(source):
                for size, indices in chosen.items():
                    if ordinal in indices:
                        handles[size].write(line)
                        written[size] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if any(written[size] != size for size in chosen):
        raise ValueError("Written sample count mismatch")
    return {str(size): {"path": str(paths[size]), "rows": written[size], "bytes": paths[size].stat().st_size}
            for size in chosen}


def convert_shared_test(manifest, output):
    parts = normal_parts(manifest, "test")
    if len(parts) != 1:
        raise ValueError("Expected one fixed shared-test part")
    part = parts[0]
    count = 0
    counts = Counter()
    row_ids = set()
    path = output / "test.shared119.jsonl"
    with gzip.open(part["path"], "rt", encoding="utf-8") as source, path.open("w", encoding="utf-8") as destination:
        for line in source:
            row = convert(json.loads(line))
            if row["row_id"] in row_ids:
                raise ValueError("Duplicate ordinary row ID within the test partition")
            row_ids.add(row["row_id"])
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[(row["task_family"], row["qid"])] += 1
            count += 1
    if count != 240458 or len(counts) != 119:
        raise ValueError("Shared final test population differs from existing benchmark")
    return {"path": str(path), "rows": count, "question_strata": len(counts),
            "purpose": "Final evaluation only; excluded from training and selection"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--prepare-test", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "preparing", "dataset": "Fixed English decision-record partitions",
              "seed": args.seed, "selection": "Proportional task_family+qid strata; random within each stratum; 50k is contained in 200k",
              "test_rows_used_for_training_or_selection": 0, "splits": {}}
    write_json(args.output / "prep_report.json", report)
    index_path = args.output / "selection_index.sqlite"
    if index_path.exists():
        raise ValueError("Output index exists; use a new output directory rather than mixing preparations")
    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE rows(split TEXT,ordinal INTEGER,row_id TEXT,task_family TEXT,qid TEXT,score REAL,PRIMARY KEY(split,row_id))")
    train_counts, train_groups, train_report = scan_split(args.manifest, "train", args.output, connection, args.seed)
    dev_counts, dev_groups, dev_report = scan_split(args.manifest, "dev", args.output, connection, args.seed + 1)
    overlap = len(train_groups & dev_groups)
    if overlap:
        raise ValueError(f"Train/dev group overlap: {overlap}")
    connection.execute("CREATE INDEX selection ON rows(split,task_family,qid,score,ordinal)")
    connection.commit()
    train_chosen, train_allocation = select_nested(connection, "train", train_counts, [50000, 200000])
    dev_chosen, dev_allocation = select_nested(connection, "dev", dev_counts, [min(20000, sum(dev_counts.values()))])
    train_report["samples"] = save_samples("train", args.output, train_chosen)
    train_report["sample_allocation"] = train_allocation
    dev_report["samples"] = save_samples("dev", args.output, dev_chosen)
    dev_report["sample_allocation"] = dev_allocation
    report["splits"] = {"train": train_report, "dev": dev_report}
    report["train_dev_group_overlap"] = overlap
    report["nested_train_samples"] = train_chosen[50000] <= train_chosen[200000]
    report["status"] = "train_dev_complete"
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    write_json(args.output / "prep_report.json", report)
    log(phase="train_dev_complete", train_rows=train_report["rows"], dev_rows=dev_report["rows"],
        samples=[50000, 200000], dev_sample_rows=20000, train_dev_group_overlap=overlap,
        elapsed_seconds=report["elapsed_seconds"], output=str(args.output))
    if args.prepare_test:
        report["shared_test"] = convert_shared_test(args.manifest, args.output)
    connection.close()
    report["status"] = "complete"
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    write_json(args.output / "prep_report.json", report)
    log(phase="complete", elapsed_seconds=report["elapsed_seconds"], output=str(args.output))


if __name__ == "__main__":
    main()

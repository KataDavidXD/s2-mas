"""Merge distributed prediction files using their ordinary test indices."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path


def merge(paths, output, expected_rows=240458, expected_questions=119):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".pending")
    seen, questions = set(), set()
    opener = gzip.open if output.suffix == ".gz" else open
    try:
        with opener(temporary, "wt", encoding="utf-8") as destination:
            for path in sorted(map(Path, paths)):
                with path.open(encoding="utf-8") as source:
                    for line in source:
                        row = json.loads(line)
                        index = row.get("index")
                        if (isinstance(index, bool) or not isinstance(index, int)
                                or not 0 <= index < expected_rows or index in seen):
                            raise ValueError("Invalid or duplicate ordinary prediction index")
                        if not all(key in row for key in ("qid", "labels", "truth", "prediction")):
                            raise ValueError("Prediction lacks a standard question/answer field")
                        seen.add(index)
                        questions.add(row["qid"])
                        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        if seen != set(range(expected_rows)):
            raise ValueError("Predictions do not cover the complete test row indices")
        if expected_questions is not None and len(questions) != expected_questions:
            raise ValueError("Prediction question count differs from the test")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"rows": len(seen), "questions": len(questions), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=240458)
    parser.add_argument("--expected-questions", type=int, default=119)
    args = parser.parse_args()
    print(json.dumps(merge(args.input_dir.glob("evaluation.rank-*.jsonl"), args.output,
                           args.expected_rows, args.expected_questions)))


if __name__ == "__main__":
    main()

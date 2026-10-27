"""Select the 9B training method from the two fixed 4B full-dev evaluations."""
import argparse
import json
import math
from pathlib import Path


def select(head, lora):
    scores = {"head_only": head, "lora": lora}
    if head["rows"] != lora["rows"] or head["questions"] != lora["questions"]:
        raise ValueError("Method selection requires the same complete development set")
    for score in scores.values():
        for key in ("mean_question_macro_f1", "accuracy"):
            if not math.isfinite(score[key]) or not 0 <= score[key] <= 1:
                raise ValueError("Development metrics must be finite probabilities")
    method = max(scores, key=lambda name: (scores[name]["mean_question_macro_f1"], scores[name]["accuracy"]))
    return {"selected_method": method, "selection_split": "full dev",
            "test_used_for_selection": False,
            "scores": {name: {key: score[key] for key in ("rows", "questions", "accuracy", "mean_question_macro_f1")}
                       for name, score in scores.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-evaluation", type=Path, required=True)
    parser.add_argument("--lora-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = select(json.loads(args.head_evaluation.read_text()), json.loads(args.lora_evaluation.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value))


if __name__ == "__main__":
    main()

"""Run one explicitly selected scientific training recipe in the foreground."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys


def recipes():
    return json.loads(Path(__file__).with_name("recipes.json").read_text(encoding="utf-8"))


def command(name, data, model, output, world_size=8, parent=None):
    recipe = recipes()[name]
    directory = Path(__file__).parent
    rows = recipe["unique_train_rows"]
    common = [sys.executable, "-m", "torch.distributed.run", "--standalone",
              "--nproc_per_node=" + str(world_size)]
    if recipe["backend"] == "qwen":
        result = common + [str(directory / "train_qwen.py"), "--model-path", str(model),
            "--model-name", recipe["model_name"], "--output", str(output),
            "--method", recipe["method"], "--train-path", str(data / f"train.{rows}.jsonl"),
            "--dev-path", str(data / "dev.20000.jsonl"), "--max-length", "2048",
            "--feature-batch-size", str(recipe["feature_batch_size"]), "--cache-chunk-rows", "256",
            "--batch-size", str(recipe["batch_size"]),
            "--gradient-accumulation", str(recipe["gradient_accumulation"]),
            "--epochs", str(recipe["epochs"]), "--learning-rate", str(recipe["learning_rate"]),
            "--head-learning-rate", str(recipe["head_learning_rate"]),
            "--early-stopping-patience", "3", "--seed", str(recipe["seed"])]
        if recipe["method"] == "lora":
            result += ["--eval-every-samples", str(recipe["eval_every_samples"])]
        return result
    continued = bool(recipe.get("parent"))
    if continued and parent is None:
        raise ValueError("Continuation requires --parent pointing to the named SFT best checkpoint")
    result = common + [str(directory / "train_laya.py"),
        "--model-path", str(parent if continued else model),
        "--initialization-name", str(parent) if continued else "convaiinnovations/laya-upstream-original",
        "--output", str(output), "--objective", recipe["objective"], "--epochs", "1",
        "--train-path", str(data / f"train.{rows}.jsonl"), "--dev-path", str(data / "dev.20000.jsonl"),
        "--batch-size", "16", "--eval-batch-size", "64", "--gradient-accumulation", "6",
        "--encoder-learning-rate", str(recipe["encoder_learning_rate"]),
        "--head-learning-rate", str(recipe["head_learning_rate"]),
        "--eval-every-samples", "50000", "--seed", str(recipe["seed"])]
    if continued:
        result += ["--encoding-model-path", str(model), "--early-stopping-patience", "0"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--recipe", choices=list(recipes()))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(recipes()))
        return
    if not args.recipe or any(value is None for value in (args.data, args.model, args.output)):
        parser.error("A recipe requires --data, --model and --output")
    if args.world_size < 1:
        parser.error("--world-size must be positive")
    cmd = command(args.recipe, args.data, args.model, args.output, args.world_size, args.parent)
    if args.dry_run:
        print(json.dumps(cmd))
    else:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

"""Inference for a Qwen3.5 independent-candidate financial decision head.

The package supplies decision_config.json, head.pt, and (for ``lora``) an
adapter/ directory.  Original Qwen weights and tokenizer are an upstream
dependency, supplied separately to ``DecisionPredictor.from_pretrained``.

Input rows contain ``state``, ``qid``, ``question`` (``instructions`` and an
ordered ``criteria`` mapping), and ordered ``candidate_ids``.  Observed answers
and respondent identifiers are unnecessary for inference.  Each candidate gets
its own complete sequence containing the state, question, all options, and that
candidate's description.  No chat template is applied and no input is truncated.

Backbone inference uses BF16 and SDPA.  The last nonpadding token is converted
to FP32 and L2-normalized.  Head-only packages additionally round this feature
through BF16, matching their stored frozen features; LoRA packages retain the
normalized FP32 feature.  A bias-free FP32 linear head scores candidates, and a
softmax over the allowed candidates produces the response distribution.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROMPT_VERSION = "full-state-question-options-focus-v1"
ARCHITECTURE = "last-nonpadding-L2-linear-candidate-softmax-v1"
UPSTREAM_MODELS = {
    "Qwen3.5-4B-Base": "Qwen/Qwen3.5-4B-Base",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
}


def validate_row(row):
    """Validate the unlabeled inference fields without changing their order."""
    required = {"state", "qid", "question", "candidate_ids"}
    if not isinstance(row, dict) or not required.issubset(row):
        raise ValueError("Input requires state, qid, question, and candidate_ids")
    ids = row["candidate_ids"]
    if not isinstance(ids, list) or len(ids) < 2 or any(not isinstance(value, str) for value in ids):
        raise ValueError("candidate_ids must contain at least two distinct strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be distinct")
    question = row["question"]
    criteria = question.get("criteria") if isinstance(question, dict) else None
    if not isinstance(criteria, dict) or set(criteria) != set(ids):
        raise ValueError("Question criteria must match all allowed candidate IDs")
    return row


def prompt(row, focus):
    """The training-time full-state/question/options/focus rendering, verbatim."""
    question = row["question"]
    return ("Predict this respondent's observed answer using the supplied respondent state.\n"
            "Respondent state:\n" + json.dumps(row["state"], ensure_ascii=False) +
            "\nQuestion:\n" + str(question.get("instructions", row["qid"])) +
            "\nAll allowed options:\n" + json.dumps(question["criteria"], ensure_ascii=False) +
            "\nCandidate being scored:\n" + json.dumps({"id": focus, "description": question["criteria"][focus]}, ensure_ascii=False) +
            "\nCandidate compatibility with this respondent's answer:")


def encode_rows(rows, tokenizer, max_length):
    """Preserve complete candidate inputs and reject an insufficient window."""
    if not rows or isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("A nonempty row batch and a positive max_length are required")
    for row in rows:
        validate_row(row)
    texts = [prompt(row, candidate) for row in rows for candidate in row["candidate_ids"]]
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    lengths = [len(value) for value in encoded["input_ids"]]
    if max(lengths) > max_length:
        raise ValueError(f"Complete candidate input exceeds max_length ({max(lengths)} > {max_length}); input was not truncated")
    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
    return batch, [len(row["candidate_ids"]) for row in rows], max(lengths)


def text_model(backbone):
    base = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    if hasattr(base, "model") and hasattr(base.model, "language_model"):
        return base.model.language_model
    if hasattr(base, "language_model"):
        return base.language_model
    raise ValueError("Cannot locate Qwen3.5 text backbone")


def pooled_hidden(backbone, tokens):
    import torch
    hidden = text_model(backbone)(**tokens, use_cache=False, return_dict=True).last_hidden_state
    positions = tokens["attention_mask"].sum(-1) - 1
    values = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions].float()
    return torch.nn.functional.normalize(values, dim=-1)


def read_config(package_path):
    """Read scientific inference settings, with no training/optimizer files."""
    config = json.loads((Path(package_path) / "decision_config.json").read_text(encoding="utf-8"))
    required = {"base_model", "model_name", "method", "max_length", "architecture", "prompt_version"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("decision_config.json lacks required inference fields")
    if config["architecture"] != ARCHITECTURE or config["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Package architecture or prompt version is unsupported")
    if config["method"] not in {"head_only", "lora"}:
        raise ValueError("Package method must be head_only or lora")
    if config["model_name"] not in UPSTREAM_MODELS or config["base_model"] != UPSTREAM_MODELS[config["model_name"]]:
        raise ValueError("Package must identify its original supported Qwen3.5 dependency")
    length = config["max_length"]
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("Package max_length must be a positive integer")
    expected_precision = {"backbone_dtype": "bfloat16", "head_dtype": "float32",
                          "normalized_feature_dtype": "bfloat16" if config["method"] == "head_only" else "float32",
                          "attention_implementation": "sdpa"}
    for field, expected in expected_precision.items():
        if field in config and config[field] != expected:
            raise ValueError(f"Unsupported precision setting: {field}")
    return config


class DecisionPredictor:
    """A loaded upstream backbone plus the package's learned decision head."""
    def __init__(self, tokenizer, backbone, head, config, device):
        self.tokenizer, self.backbone, self.head = tokenizer, backbone, head
        self.config, self.device = config, device

    @classmethod
    def from_pretrained(cls, package_path, upstream_model, *, device="cuda", local_files_only=False):
        """Load a local decision package and its explicitly supplied dependency.

        ``upstream_model`` is the official repository named in the config or a
        local directory containing those upstream weights and tokenizer.
        ``local_files_only=True`` requires the dependency to be cached locally.
        """
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("USE_HUB_KERNELS", "0")
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        package_path = Path(package_path)
        config = read_config(package_path)
        upstream_model = str(upstream_model)
        if not Path(upstream_model).is_dir() and upstream_model != config["base_model"]:
            raise ValueError("Supplied upstream repository differs from decision_config.json")
        adapter = package_path / "adapter"
        adapter_present = (adapter / "adapter_config.json").is_file()
        if config["method"] == "lora" and not adapter_present:
            raise ValueError("LoRA package is missing its adapter configuration")
        if config["method"] == "head_only" and adapter.exists():
            raise ValueError("Head-only package unexpectedly contains an adapter directory")
        saved = torch.load(package_path / "head.pt", map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or any(field not in saved for field in
                ("head_state_dict", "hidden_size", "model_name", "architecture", "prompt_version")):
            raise ValueError("head.pt lacks required head fields")
        for field in ("model_name", "architecture", "prompt_version"):
            if saved[field] != config[field]:
                raise ValueError(f"Head metadata differs from package config: {field}")
        width = saved["hidden_size"]
        weights = saved["head_state_dict"]
        if (isinstance(width, bool) or not isinstance(width, int) or width < 1
                or not isinstance(weights, dict) or set(weights) != {"weight"}
                or not isinstance(weights["weight"], torch.Tensor)
                or tuple(weights["weight"].shape) != (1, width)
                or weights["weight"].dtype != torch.float32
                or not bool(torch.isfinite(weights["weight"]).all())):
            raise ValueError("Head must contain one finite FP32 bias-free linear weight")
        device = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(upstream_model, local_files_only=local_files_only)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            upstream_model, local_files_only=local_files_only, dtype=torch.bfloat16,
            device_map={"": str(device)}, attn_implementation="sdpa")
        backbone.requires_grad_(False)
        if backbone.config.text_config.hidden_size != width:
            raise ValueError("Upstream hidden size differs from the decision head")
        if config["method"] == "lora":
            from peft import PeftModel
            # PEFT's default adapter dtype handling matches the evaluation runner.
            backbone = PeftModel.from_pretrained(backbone, adapter, is_trainable=False)
        backbone.eval()
        head = torch.nn.Linear(width, 1, bias=False, dtype=torch.float32)
        head.load_state_dict(weights, strict=True)
        head.requires_grad_(False)
        head.to(device).eval()
        return cls(tokenizer, backbone, head, config, device)

    def predict(self, rows, *, row_batch_size=4, candidate_batch_size=16):
        """Return ordered logits, probabilities, and the highest-scoring choice.

        Candidate sequences occupy independent batch entries.  Batching changes
        only forward scheduling; it never combines candidates into one sequence.
        """
        import torch
        single = isinstance(rows, dict)
        rows = [rows] if single else list(rows)
        if row_batch_size < 1 or candidate_batch_size < 1:
            raise ValueError("Forward batch sizes must be positive")
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(rows), row_batch_size):
                selected_rows = rows[start:start + row_batch_size]
                tokens, counts, _ = encode_rows(selected_rows, self.tokenizer, self.config["max_length"])
                flat_scores = []
                for offset in range(0, tokens["input_ids"].shape[0], candidate_batch_size):
                    chosen = slice(offset, offset + candidate_batch_size)
                    batch_length = int(tokens["attention_mask"][chosen].sum(-1).max())
                    batch = {key: value[chosen, :batch_length].to(self.device) for key, value in tokens.items()}
                    features = pooled_hidden(self.backbone, batch)
                    if self.config["method"] == "head_only":
                        features = features.to(torch.bfloat16).float()
                    flat_scores.append(self.head(features).squeeze(-1))
                scores = torch.cat(flat_scores)
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("A valid candidate has a nonfinite score")
                for row, logits in zip(selected_rows, scores.split(counts)):
                    index = int(logits.argmax())
                    predictions.append({"qid": row["qid"], "candidate_ids": list(row["candidate_ids"]),
                                        "logits": logits.cpu().tolist(),
                                        "probabilities": logits.softmax(dim=-1).cpu().tolist(),
                                        "prediction_index": index,
                                        "prediction": row["candidate_ids"][index]})
        return predictions[0] if single else predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON row or JSON list of rows")
    parser.add_argument("--output", type=Path, help="Default: print JSON to stdout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    predictor = DecisionPredictor.from_pretrained(args.package, args.upstream_model,
                                                 device=args.device, local_files_only=args.local_files_only)
    result = predictor.predict(rows, row_batch_size=args.row_batch_size,
                               candidate_batch_size=args.candidate_batch_size)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()

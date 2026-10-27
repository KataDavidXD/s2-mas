"""Shared candidate scoring and observed-response metrics. """
from __future__ import annotations

import array
import bisect
import json
import os
from collections import OrderedDict
from pathlib import Path


PROMPT_VERSION = "full-state-question-options-focus-v1"
ARCHITECTURE = "last-nonpadding-L2-linear-candidate-softmax-v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_row(row):
    required = {"row_id", "group_id", "task_family", "qid", "state", "question", "candidate_ids", "target"}
    if not required.issubset(row):
        raise ValueError("Prepared row is missing required fields")
    ids = row["candidate_ids"]
    if not isinstance(ids, list) or len(ids) < 2 or len(set(map(str, ids))) != len(ids):
        raise ValueError("Candidate IDs must contain at least two distinct choices")
    if any(not isinstance(item, str) for item in ids):
        raise ValueError("Candidate IDs must be strings")
    target = row["target"]
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(ids):
        raise ValueError("Observed target must be a valid candidate index")
    criteria = row["question"].get("criteria")
    if not isinstance(criteria, dict) or set(criteria) != set(ids):
        raise ValueError("Question criteria must match the complete candidate IDs")
    return row


def prompt(row, focus):
    question = row["question"]
    return ("Predict this respondent's observed answer using the supplied respondent state.\n"
            "Respondent state:\n" + json.dumps(row["state"], ensure_ascii=False) +
            "\nQuestion:\n" + str(question.get("instructions", row["qid"])) +
            "\nAll allowed options:\n" + json.dumps(question["criteria"], ensure_ascii=False) +
            "\nCandidate being scored:\n" + json.dumps({"id": focus, "description": question["criteria"][focus]}, ensure_ascii=False) +
            "\nCandidate compatibility with this respondent's answer:")


class JsonlRows:
    """A byte-offset index avoids retaining millions of respondent records in RAM."""
    def __init__(self, path, limit=0):
        self.path = Path(path)
        if self.path.suffix == ".gz":
            raise ValueError("Training requires prepared uncompressed JSONL")
        self.offsets = array.array("Q")
        with self.path.open("rb") as handle:
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    raise ValueError("Blank rows are not permitted in prepared JSONL")
                self.offsets.append(position)
                if limit and len(self.offsets) >= limit:
                    break
        if not self.offsets:
            raise ValueError("Prepared split has no rows")
        self.handle = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = self.path.open("rb")
        self.handle.seek(self.offsets[index])
        row = validate_row(json.loads(self.handle.readline()))
        row.setdefault("index", index)
        return row

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handle"] = None
        return state


def row_metadata(row):
    return {key: row[key] for key in ("index", "row_id", "group_id", "task_family", "qid", "candidate_ids", "target") if key in row}


def encode_rows(rows, tokenizer, max_length):
    """Reject overlength full inputs; never silently truncate respondent information."""
    texts = [prompt(row, candidate) for row in rows for candidate in row["candidate_ids"]]
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    lengths = list(map(len, encoded["input_ids"]))
    if max(lengths) > max_length:
        raise ValueError(f"Complete candidate input exceeds --max-length ({max(lengths)} > {max_length}); input was not truncated")
    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
    return batch, [len(row["candidate_ids"]) for row in rows], [row_metadata(row) for row in rows], max(lengths)


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


def candidate_logits(flat_scores, counts):
    import torch
    pieces = list(flat_scores.split(counts))
    return torch.nn.utils.rnn.pad_sequence(pieces, batch_first=True, padding_value=-float("inf"))


class MetricCounts:
    """Per-question macro F1 includes every valid class, including unobserved classes."""
    def __init__(self):
        self.questions = {}
        self.rows = 0
        self.correct = 0
        self.loss_sum = 0.0

    def add(self, metadata, predictions, loss_sum=0.0):
        self.loss_sum += float(loss_sum)
        for row, predicted in zip(metadata, predictions):
            ids = row["candidate_ids"]
            target = row["target"]
            key = str(row["task_family"]) + "::" + str(row["qid"])
            entry = self.questions.setdefault(key, {"classes": {}, "rows": 0, "correct": 0, "labels": list(ids)})
            if entry["labels"] != ids:
                raise ValueError("A task_family/question ID has inconsistent ordered candidates; use versioned question IDs")
            for candidate in ids:
                entry["classes"].setdefault(candidate, [0, 0, 0])
            for index, candidate in enumerate(ids):
                entry["classes"][candidate][0] += int(index == target and predicted == target)
                entry["classes"][candidate][1] += int(index != target and predicted == index)
                entry["classes"][candidate][2] += int(index == target and predicted != target)
            entry["rows"] += 1
            entry["correct"] += int(predicted == target)
            self.rows += 1
            self.correct += int(predicted == target)

    def merge(self, other):
        self.rows += other["rows"]
        self.correct += other["correct"]
        self.loss_sum += other["loss_sum"]
        for key, source in other["questions"].items():
            target = self.questions.setdefault(key, {"classes": {}, "rows": 0, "correct": 0, "labels": source["labels"]})
            if target["labels"] != source["labels"]:
                raise ValueError("Ranks observed inconsistent ordered candidates for one question ID")
            target["rows"] += source["rows"]
            target["correct"] += source["correct"]
            for label, counts in source["classes"].items():
                present = target["classes"].setdefault(label, [0, 0, 0])
                for index in range(3):
                    present[index] += counts[index]

    def state(self):
        return {"rows": self.rows, "correct": self.correct, "loss_sum": self.loss_sum, "questions": self.questions}

    def result(self):
        per_question = {}
        for key, values in sorted(self.questions.items()):
            scores = [2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                      for tp, fp, fn in values["classes"].values()]
            per_question[key] = {"rows": values["rows"], "accuracy": values["correct"] / values["rows"],
                                 "macro_f1": sum(scores) / len(scores), "candidate_classes": len(scores)}
        return {"rows": self.rows, "accuracy": self.correct / self.rows if self.rows else 0.0,
                "mean_question_macro_f1": sum(v["macro_f1"] for v in per_question.values()) / len(per_question) if per_question else 0.0,
                "loss": self.loss_sum / self.rows if self.rows else 0.0, "questions": len(per_question),
                "per_question": per_question}


class CachedFeatures:
    def __init__(self, directory, split, sizes):
        self.directory, self.split, self.sizes = Path(directory), split, sizes
        self.ends = []
        for size in sizes:
            self.ends.append((self.ends[-1] if self.ends else 0) + size)
        self.loaded = OrderedDict()
        # The fixed dev20k split fits in 79 chunks. Retaining it avoids reading
        # the same features from shared storage at every head-only validation; the bound
        # remains small relative to node RAM and does not grow with full data.
        self.max_loaded_chunks = 128 if split == "dev" else 2

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        import torch
        chunk = bisect.bisect_right(self.ends, index)
        if chunk not in self.loaded:
            package = torch.load(self.directory / f"{self.split}-{chunk:07d}.pt", map_location="cpu", weights_only=False)
            offsets = package.get("row_offsets", [])
            counts = [len(row["candidate_ids"]) for row in package["metadata"]]
            if (package.get("cache_format") != "flat-candidates-v1" or package["features"].ndim != 2
                    or len(package["metadata"]) != self.sizes[chunk] or len(offsets) != self.sizes[chunk] + 1
                    or not offsets or offsets[0] != 0 or offsets[-1] != sum(counts)
                    or package["features"].shape[0] != sum(counts)
                    or any(offsets[row + 1] - offsets[row] != count for row, count in enumerate(counts))):
                raise ValueError("Feature cache has inconsistent flat candidate offsets/counts")
            self.loaded[chunk] = package
            if len(self.loaded) > self.max_loaded_chunks:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(chunk)
        offset = index - (self.ends[chunk - 1] if chunk else 0)
        package = self.loaded[chunk]
        start, end = package["row_offsets"][offset:offset + 2]
        return package["features"][start:end], package["metadata"][offset]


def feature_collate(items):
    import torch
    features, metadata = zip(*items)
    counts = [len(row["candidate_ids"]) for row in metadata]
    return torch.cat([feature[:count] for feature, count in zip(features, counts)], dim=0), counts, list(metadata)

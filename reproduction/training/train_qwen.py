"""Distributed Qwen3.5 candidate head training; head-only cache or LoRA + head.

Inputs are prepared JSONL with frozen native train/dev splits. No test
data is read during training. --evaluate-path evaluates a saved checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("USE_HUB_KERNELS", "0")



def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--dev-path", type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen3.5-4B-Base")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("head_only", "lora"), required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4, help="Respondent rows per GPU")
    parser.add_argument("--feature-batch-size", type=int, default=16, help="Candidate sequences per frozen extraction forward")
    parser.add_argument("--feature-token-budget", type=int, default=65536, help="Maximum padded tokens per frozen extraction forward")
    parser.add_argument("--cache-chunk-rows", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume output/last after its saved optimizer step")
    parser.add_argument("--eval-every-steps", type=int, default=0, help="Validate/checkpoint every N optimizer updates; 0 means epoch end only")
    parser.add_argument("--eval-every-samples", type=int, default=0, help="Also validate after each N globally processed training rows")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Stop after N consecutive dev evaluations without improvement; 0 disables")
    parser.add_argument("--checkpoint", type=Path, help="Load head and LoRA adapter for evaluation")
    parser.add_argument("--evaluate-path", type=Path)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if not args.evaluate_path and (not args.train_path or not args.dev_path):
        parser.error("Training requires --train-path and --dev-path")
    if args.evaluate_path and not args.checkpoint:
        parser.error("Evaluation requires --checkpoint")
    if args.resume and args.checkpoint:
        parser.error("--resume and --checkpoint cannot be combined")
    if min(args.batch_size, args.feature_batch_size, args.feature_token_budget, args.cache_chunk_rows, args.gradient_accumulation, args.epochs) < 1:
        parser.error("Batch sizes, token budget, chunks, accumulation and epochs must be positive")
    if min(args.eval_every_steps, args.eval_every_samples, args.early_stopping_patience) < 0:
        parser.error("Evaluation intervals and early stopping patience cannot be negative")
    return args


if __name__ == "__main__" and any(value in {"-h", "--help"} for value in __import__("sys").argv[1:]):
    arguments()


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from common import (ARCHITECTURE, PROMPT_VERSION, CachedFeatures, JsonlRows, MetricCounts,
                    atomic_json, candidate_logits, encode_rows, feature_collate,
                    pooled_hidden, row_metadata, text_model)


def setup():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("This training runner requires CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    return rank, world, torch.device("cuda", local_rank)


def barrier(world):
    if world > 1:
        dist.barrier()


def log(rank, **value):
    if rank == 0:
        print(json.dumps(value, ensure_ascii=False), flush=True)


def resolve_model(path):
    if (path / "config.json").exists():
        return path
    snapshots = list((path / "snapshots").glob("*/config.json"))
    if len(snapshots) != 1:
        raise ValueError("--model-path needs a model directory or one local cached release")
    return snapshots[0].parent


def resolve_checkpoint(directory):
    """Recover the last complete directory if a process stopped during its switch."""
    directory = Path(directory)
    if directory.exists():
        if not (directory / "complete.json").exists():
            raise ValueError("Requested checkpoint is incomplete")
        return directory
    previous = sorted(directory.parent.glob(f".{directory.name}.previous-*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in previous:
        if (candidate / "complete.json").exists():
            return candidate
    raise ValueError("No complete requested checkpoint is available")


def load_backbone(args, device):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    path = resolve_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": str(device)}, attn_implementation="sdpa")
    backbone.requires_grad_(False)
    width = backbone.config.text_config.hidden_size
    targets = []
    if args.method == "lora":
        from peft import LoraConfig, PeftModel, get_peft_model
        checkpoint = args.checkpoint or (args.output / "last" if args.resume else None)
        if checkpoint:
            checkpoint = resolve_checkpoint(checkpoint)
        if checkpoint and not (checkpoint / "adapter" / "adapter_config.json").exists():
            raise ValueError("Requested LoRA checkpoint has no saved adapter; refusing to evaluate or resume a fresh adapter")
        if checkpoint:
            backbone = PeftModel.from_pretrained(backbone, checkpoint / "adapter", is_trainable=not bool(args.evaluate_path))
        else:
            names = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
            targets = [name for name, module in backbone.named_modules()
                       if ".language_model." in "." + name and name.rsplit(".", 1)[-1] in names and isinstance(module, torch.nn.Linear)]
            if not targets:
                raise ValueError("No Qwen3.5 text attention/linear-attention LoRA targets found")
            backbone = get_peft_model(backbone, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout, target_modules=targets, bias="none"))
        if args.gradient_checkpointing and not args.evaluate_path:
            language_model = text_model(backbone)
            language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(language_model, "enable_input_require_grads"):
                language_model.enable_input_require_grads()
            if not getattr(language_model, "is_gradient_checkpointing", False):
                raise RuntimeError("Qwen3.5 language model did not enable gradient checkpointing")
    backbone.eval()
    return tokenizer, backbone, width, targets


class ChoiceModel(torch.nn.Module):
    def __init__(self, width, backbone=None):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(width, 1, bias=False, dtype=torch.float32)
        # A nonzero initial head lets the first LoRA step receive backbone gradients.
        torch.nn.init.normal_(self.head.weight, mean=0.0, std=0.01)

    def forward(self, inputs, counts):
        features = pooled_hidden(self.backbone, inputs) if self.backbone is not None else inputs.float()
        return candidate_logits(self.head(features).squeeze(-1), counts)


def save_tensor(path, package):
    temporary = path.with_name(path.name + ".pending")
    torch.save(package, temporary)
    os.replace(temporary, path)


def prepare_cache(args, split, rows, tokenizer, backbone, device, rank, world):
    directory = args.cache_dir or args.output / "features"
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / f"{split}.json"
    sizes = [min(args.cache_chunk_rows, len(rows) - offset) for offset in range(0, len(rows), args.cache_chunk_rows)]
    settings = {"version": ARCHITECTURE, "prompt_version": PROMPT_VERSION, "model_name": args.model_name,
                "prepared_path": str(rows.path), "rows": len(rows), "max_length": args.max_length,
                "chunk_rows": args.cache_chunk_rows, "sizes": sizes, "cache_format": "flat-candidates-v1"}
    if rank == 0:
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("Existing feature cache uses different named inputs/configuration; choose a new --cache-dir")
        atomic_json(config_path, settings)
    barrier(world)
    started = time.monotonic()
    local_rows = local_candidates = local_tokens = reused_rows = 0
    assigned_chunks = list(range(rank, len(sizes), world))
    assigned_rows = sum(sizes[chunk] for chunk in assigned_chunks)
    logged_first_1000 = False
    for local_chunk_index, chunk in enumerate(assigned_chunks):
        path = directory / f"{split}-{chunk:07d}.pt"
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=False)
            offsets = existing.get("row_offsets", [])
            counts = [len(row["candidate_ids"]) for row in existing["metadata"]]
            if (existing.get("cache_format") != "flat-candidates-v1"
                    or existing["features"].ndim != 2 or len(existing["metadata"]) != sizes[chunk]
                    or len(offsets) != sizes[chunk] + 1 or not offsets or offsets[0] != 0
                    or offsets[-1] != sum(counts) or existing["features"].shape[0] != sum(counts)
                    or any(offsets[index + 1] - offsets[index] != count for index, count in enumerate(counts))):
                raise ValueError("Existing feature chunk has inconsistent flat candidate offsets/counts")
            reused_rows += sizes[chunk]
            continue
        start = chunk * args.cache_chunk_rows
        chunk_rows = [rows[index] for index in range(start, start + sizes[chunk])]
        tokens, counts, metadata, max_tokens = encode_rows(chunk_rows, tokenizer, args.max_length)
        lengths = tokens["attention_mask"].sum(-1)
        order = lengths.argsort()
        features = torch.empty(tokens["input_ids"].shape[0], backbone.config.text_config.hidden_size, dtype=torch.bfloat16)
        with torch.inference_mode():
            offset = 0
            while offset < tokens["input_ids"].shape[0]:
                batch_size = min(args.feature_batch_size, tokens["input_ids"].shape[0] - offset)
                while batch_size > 1 and batch_size * int(lengths[order[offset + batch_size - 1]]) > args.feature_token_budget:
                    batch_size -= 1
                if int(lengths[order[offset]]) > args.feature_token_budget:
                    raise ValueError("One complete input exceeds --feature-token-budget; increase the resource budget")
                while True:
                    selected = order[offset:offset + batch_size]
                    batch_length = int(lengths[selected].max())
                    batch = None
                    oom = False
                    terminal_error = None
                    try:
                        batch = {key: value[selected, :batch_length].to(device) for key, value in tokens.items()}
                        features[selected] = pooled_hidden(backbone, batch).to(dtype=torch.bfloat16, device="cpu")
                    except torch.cuda.OutOfMemoryError as error:
                        error.__traceback__ = None
                        terminal_error = error if batch_size == 1 else None
                        oom = True
                    batch = None
                    if not oom:
                        offset += batch_size
                        break
                    # Exit the exception block before releasing cached CUDA pages:
                    # traceback frames must not retain the failed hidden tensors.
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    next_size = max(1, batch_size // 2)
                    print(json.dumps({"phase": "feature_batch_oom_backoff", "rank": rank, "split": split,
                        "chunk": chunk, "offset": offset, "previous_batch_size": batch_size,
                        "retry_batch_size": next_size if terminal_error is None else None,
                        "max_sequence_tokens": batch_length, "padded_tokens": batch_size * batch_length,
                        "feature_token_budget": args.feature_token_budget,
                        "allocated_bytes": torch.cuda.memory_allocated(device),
                        "reserved_bytes": torch.cuda.memory_reserved(device)}, ensure_ascii=False), flush=True)
                    if terminal_error is not None:
                        raise terminal_error from None
                    batch_size = next_size
        row_offsets = [0]
        for count in counts:
            row_offsets.append(row_offsets[-1] + count)
        save_tensor(path, {"features": features, "row_offsets": row_offsets, "metadata": metadata,
                          "max_tokens": max_tokens, "cache_format": "flat-candidates-v1"})
        local_rows += len(chunk_rows)
        local_candidates += sum(counts)
        local_tokens += int(lengths.sum())
        first_1000 = not logged_first_1000 and local_rows >= 1000
        if first_1000:
            logged_first_1000 = True
        if local_chunk_index % 10 == 0 or first_1000 or local_chunk_index + 1 == len(assigned_chunks):
            elapsed = time.monotonic() - started
            row_rate = local_rows / elapsed
            remaining = max(0, assigned_rows - reused_rows - local_rows)
            print(json.dumps({"phase": "feature_extraction", "rank": rank, "world_size": world,
                "split": split, "chunk": chunk, "local_chunks_done": local_chunk_index + 1,
                "local_chunks_total": len(assigned_chunks), "local_rows": local_rows,
                "local_rows_reused": reused_rows, "local_rows_assigned": assigned_rows,
                "local_candidates": local_candidates, "local_input_tokens": local_tokens,
                "rows_per_second": row_rate, "candidates_per_second": local_candidates / elapsed,
                "input_tokens_per_second": local_tokens / elapsed, "elapsed_seconds": elapsed,
                "local_remaining_eta_seconds": remaining / row_rate if row_rate else None,
                "first_1000_rows_reached": first_1000,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}, ensure_ascii=False), flush=True)
    barrier(world)
    return CachedFeatures(directory, split, sizes)


class TokenCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, rows):
        tokens, counts, metadata, _ = encode_rows(rows, self.tokenizer, self.max_length)
        return tokens, counts, metadata


class ChunkShuffleSampler(torch.utils.data.Sampler):
    """Shuffle cache chunks and their rows while retaining locality and equal ranks.

    Each rank consumes a contiguous slice of the shuffled global row order, so
    feature chunks are read mostly by one rank. Only the last few rows repeat
    when the split is not divisible by world size.
    """
    def __init__(self, dataset, rank, world, seed, chunk_rows=256):
        self.dataset, self.rank, self.world, self.seed = dataset, rank, world, seed
        self.epoch = 0
        self.per_rank = math.ceil(len(dataset) / world)
        self.sizes = [min(chunk_rows, len(dataset) - start) for start in range(0, len(dataset), chunk_rows)]
        self.ends = []
        for size in self.sizes:
            self.ends.append((self.ends[-1] if self.ends else 0) + size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.per_rank

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        chunks = list(range(len(self.sizes)))
        rng.shuffle(chunks)
        order = []
        for chunk in chunks:
            start = self.ends[chunk - 1] if chunk else 0
            indices = list(range(start, self.ends[chunk]))
            rng.shuffle(indices)
            order.extend(indices)
        required = self.per_rank * self.world - len(order)
        if required:
            order.extend((order * math.ceil(required / len(order)))[:required])
        start = self.rank * self.per_rank
        return iter(order[start:start + self.per_rank])


def to_device(inputs, device):
    if isinstance(inputs, dict) or hasattr(inputs, "items"):
        return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    return inputs.to(device, non_blocking=True)


def evaluate(model, dataset, collate, args, device, rank, world, name, save_predictions=True):
    model.eval()
    # No padded DistributedSampler: each observation enters validation once.
    indices = range(rank, len(dataset), world)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=indices, collate_fn=collate,
                        num_workers=args.num_workers, pin_memory=True)
    counts = MetricCounts()
    prediction_path = args.output / f"{name}.rank-{rank:02d}.jsonl"
    output = prediction_path.open("w", encoding="utf-8") if save_predictions else contextlib.nullcontext()
    with output as handle, torch.inference_mode():
        for inputs, candidates, metadata in loader:
            targets = torch.tensor([row["target"] for row in metadata], device=device)
            logits = model(to_device(inputs, device), candidates)
            valid = torch.arange(logits.shape[1], device=device).unsqueeze(0) < torch.tensor(candidates, device=device).unsqueeze(1)
            if not bool(torch.isfinite(logits[valid]).all()):
                raise FloatingPointError("A valid evaluation candidate has a nonfinite score")
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
            predicted = logits.argmax(-1).tolist()
            counts.add(metadata, predicted, float(loss))
            if handle is not None:
                for row, index in zip(metadata, predicted):
                    handle.write(json.dumps({**row, "labels": row["candidate_ids"], "prediction": row["candidate_ids"][index],
                                            "truth": row["candidate_ids"][row["target"]]}, ensure_ascii=False) + "\n")
    if world > 1:
        all_counts = [None] * world
        dist.all_gather_object(all_counts, counts.state())
        combined = MetricCounts()
        for package in all_counts:
            combined.merge(package)
        counts = combined
    return counts.result()


def collect_rng(world):
    local = {"torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate()}
    if world == 1:
        return [local]
    states = [None] * world
    dist.all_gather_object(states, local)
    return states


def save_checkpoint(directory, model, optimizer, scheduler, progress, args, metrics, rng_states):
    temporary = directory.with_name(f".{directory.name}.pending-{time.time_ns()}")
    temporary.mkdir(parents=True)
    save_tensor(temporary / "head.pt", {"head_state_dict": model.head.state_dict(),
                "hidden_size": model.head.in_features, "model_name": args.model_name,
                "architecture": ARCHITECTURE, "prompt_version": PROMPT_VERSION})
    if model.backbone is not None:
        model.backbone.save_pretrained(temporary / "adapter", safe_serialization=True)
    save_tensor(temporary / "training.pt", {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                **progress, "rng_by_rank": rng_states})
    atomic_json(temporary / "metrics.json", metrics)
    atomic_json(temporary / "settings.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    atomic_json(temporary / "complete.json", {"epoch": progress["epoch"] + 1, "next_batch": progress["next_batch"],
                "global_step": progress["global_step"], "method": args.method, "model_name": args.model_name})
    if directory.exists():
        directory.rename(directory.with_name(f".{directory.name}.previous-{time.time_ns()}"))
    temporary.rename(directory)
    # Keep one recovery generation per name. Frequent validation must not retain
    # an unbounded number of adapter and optimizer copies on shared storage.
    previous = sorted(directory.parent.glob(f".{directory.name}.previous-*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for obsolete in previous[1:]:
        resolved = obsolete.resolve()
        if not resolved.is_relative_to(directory.parent.resolve()):
            raise ValueError("Checkpoint cleanup target escaped its output directory")
        shutil.rmtree(obsolete)


def main():
    args = arguments()
    rank, world, device = setup()
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tokenizer, backbone, width, targets = load_backbone(args, device)
    eval_rows = JsonlRows(args.evaluate_path or args.dev_path, args.limit_dev)
    if args.method == "head_only":
        evaluation_data = prepare_cache(args, "evaluation" if args.evaluate_path else "dev", eval_rows,
                                        tokenizer, backbone, device, rank, world)
        training_data = None if args.evaluate_path else prepare_cache(args, "train", JsonlRows(args.train_path, args.limit_train),
                                        tokenizer, backbone, device, rank, world)
        frozen_count = sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)
        if frozen_count:
            raise RuntimeError("Head-only backbone must have zero trainable parameters")
        del backbone
        torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        model = ChoiceModel(width).to(device)
        collate = feature_collate
    else:
        evaluation_data = eval_rows
        training_data = None if args.evaluate_path else JsonlRows(args.train_path, args.limit_train)
        torch.manual_seed(args.seed)
        model = ChoiceModel(width, backbone).to(device)
        collate = TokenCollator(tokenizer, args.max_length)
    checkpoint = args.checkpoint or (args.output / "last" if args.resume else None)
    if checkpoint:
        checkpoint = resolve_checkpoint(checkpoint)
        package = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=False)
        if (package["architecture"] != ARCHITECTURE or package["hidden_size"] != width
                or package["model_name"] != args.model_name or package["prompt_version"] != PROMPT_VERSION):
            raise ValueError("Checkpoint architecture, model name, prompt version or hidden size differs")
        model.head.load_state_dict(package["head_state_dict"])
    if args.evaluate_path:
        metrics = evaluate(model, evaluation_data, collate, args, device, rank, world, "evaluation")
        if rank == 0:
            atomic_json(args.output / "evaluation.json", metrics)
        log(rank, phase="evaluation_complete", **{k: v for k, v in metrics.items() if k != "per_question"})
        barrier(world)
        if world > 1:
            dist.destroy_process_group()
        return
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    backbone_trainable = sum(parameter.numel() for name, parameter in trainable if name.startswith("backbone."))
    if args.method == "lora" and not backbone_trainable:
        raise RuntimeError("LoRA requires trainable backbone adapter parameters")
    learning_rate = args.learning_rate or (0.003 if args.method == "head_only" else 0.0001)
    head_learning_rate = args.head_learning_rate or learning_rate
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": head_learning_rate},
        {"params": [parameter for name, parameter in trainable if name.startswith("backbone.")], "lr": learning_rate}],
        weight_decay=args.weight_decay)
    # The first pass sees exactly the same row order in both methods. The block
    # order also prevents random single-row accesses from thrashing feature I/O.
    sampler = ChunkShuffleSampler(training_data, rank, world, args.seed, args.cache_chunk_rows)
    loader_generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(training_data, batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
                        num_workers=args.num_workers, pin_memory=True, generator=loader_generator)
    total_steps = args.epochs * math.ceil(len(loader) / args.gradient_accumulation)
    warmup = int(total_steps * args.warmup_ratio)
    def scale(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    start_epoch, resume_batch, global_step, samples_seen = 0, 0, 0, 0
    best_score, bad_evaluations, last_evaluated_step = (-1.0, -1.0), 0, -1
    resume_record = None
    resume_geometry = {"world_size": world, "batch_size": args.batch_size,
                       "gradient_accumulation": args.gradient_accumulation, "train_rows": len(training_data),
                       "seed": args.seed, "chunk_rows": args.cache_chunk_rows, "method": args.method,
                       "train_path": str(args.train_path), "dev_path": str(args.dev_path),
                       "model_name": args.model_name, "max_length": args.max_length}
    if args.resume:
        resume_checkpoint = resolve_checkpoint(args.output / "last")
        resume = torch.load(resume_checkpoint / "training.pt", map_location="cpu", weights_only=False)
        resume_record = json.loads((resume_checkpoint / "metrics.json").read_text(encoding="utf-8"))
        if resume.get("resume_geometry", resume_geometry) != resume_geometry:
            raise ValueError("Step resume requires the same world size, batch/accumulation, row count, sampler seed and method")
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        best_score = tuple(resume["best_score"])
        start_epoch = resume["epoch"]
        resume_batch = resume.get("next_batch", len(loader))
        if not 0 <= resume_batch <= len(loader) or (resume_batch < len(loader) and resume_batch % args.gradient_accumulation):
            raise ValueError("Resume batch must be a completed optimizer-update boundary in the same epoch")
        global_step = resume.get("global_step", (start_epoch + 1) * math.ceil(len(loader) / args.gradient_accumulation))
        samples_seen = resume.get("samples_seen", (start_epoch + 1) * len(sampler) * world)
        bad_evaluations = resume.get("bad_evaluations", 0)
        last_evaluated_step = resume.get("last_evaluated_step", global_step)
        if resume_batch >= len(loader):
            start_epoch += 1
            resume_batch = 0
        rng = resume["rng_by_rank"][rank] if "rng_by_rank" in resume else resume
        torch.set_rng_state(rng["torch_rng"].cpu())
        torch.cuda.set_rng_state(rng["cuda_rng"].cpu())
        random.setstate(rng["python_rng"])
    wrapped = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    log(rank, phase="training_start", method=args.method, world_size=world, train_rows=len(training_data),
        dev_rows=len(evaluation_data), trainable_parameters=sum(p.numel() for _, p in trainable),
        backbone_trainable_parameters=backbone_trainable, head_parameters=width,
        lora_target_modules=len(targets), prompt_version=PROMPT_VERSION, architecture=ARCHITECTURE,
        language_gradient_checkpointing=bool(model.backbone is not None and getattr(text_model(model.backbone), "is_gradient_checkpointing", False)),
        global_step=global_step, resume_epoch=start_epoch + 1, resume_batch=resume_batch,
        eval_every_steps=args.eval_every_steps, eval_every_samples=args.eval_every_samples,
        early_stopping_patience=args.early_stopping_patience)
    first_grad_check = False
    stopped = False

    def restore_terminal_result(status):
        # The complete last checkpoint may have been committed immediately before
        # a process stopped, leaving result.json at its previous running state.
        if rank == 0:
            if resume_record is None:
                raise RuntimeError("Terminal result recovery requires saved checkpoint metrics")
            atomic_json(args.output / "result.json", {**resume_record, "status": status,
                        "best_mean_question_macro_f1": best_score[0], "best_accuracy": best_score[1],
                        "global_step": global_step, "global_training_rows_processed": samples_seen,
                        "bad_evaluations": bad_evaluations})
        barrier(world)

    if args.early_stopping_patience and bad_evaluations >= args.early_stopping_patience:
        restore_terminal_result("early_stopped")
        log(rank, phase="already_early_stopped", global_step=global_step, bad_evaluations=bad_evaluations)
        if world > 1:
            dist.destroy_process_group()
        return
    if start_epoch >= args.epochs:
        restore_terminal_result("complete")
        log(rank, phase="already_complete", global_step=global_step, samples_seen=samples_seen)
        if world > 1:
            dist.destroy_process_group()
        return

    def validate_and_save(epoch, next_batch):
        nonlocal best_score, bad_evaluations, last_evaluated_step, stopped
        metrics = evaluate(model, evaluation_data, collate, args, device, rank, world, f"dev-step-{global_step}", save_predictions=False)
        score = (metrics["mean_question_macro_f1"], metrics["accuracy"])
        improved = score > best_score
        if improved:
            best_score, bad_evaluations = score, 0
        else:
            bad_evaluations += 1
        last_evaluated_step = global_step
        stopped = bool(args.early_stopping_patience and bad_evaluations >= args.early_stopping_patience)
        epoch_complete = next_batch >= len(loader)
        progress = {"epoch": epoch, "next_batch": next_batch, "global_step": global_step,
                    "samples_seen": samples_seen, "best_score": best_score, "bad_evaluations": bad_evaluations,
                    "last_evaluated_step": last_evaluated_step, "resume_geometry": resume_geometry}
        rng_states = collect_rng(world)
        global_peak_memory = torch.tensor(torch.cuda.max_memory_allocated(), device=device, dtype=torch.int64)
        if world > 1:
            dist.all_reduce(global_peak_memory, op=dist.ReduceOp.MAX)
        if rank == 0:
            record = {"epoch": epoch + 1, "next_batch": next_batch, "global_step": global_step,
                      "global_training_rows_processed": samples_seen, "dev": metrics,
                      "elapsed_seconds": time.monotonic() - started, "epoch_complete": epoch_complete,
                      "peak_allocated_bytes": int(global_peak_memory), "method": args.method,
                      "world_size": world, "train_rows": len(training_data), "dev_rows": len(evaluation_data),
                      "bad_evaluations": bad_evaluations, "backbone_trainable_parameters": backbone_trainable}
            if improved:
                save_checkpoint(args.output / "best", model, optimizer, scheduler, progress, args, record, rng_states)
            save_checkpoint(args.output / "last", model, optimizer, scheduler, progress, args, record, rng_states)
            status = "early_stopped" if stopped else ("complete" if epoch_complete and epoch + 1 >= args.epochs else "running")
            atomic_json(args.output / "result.json", {"status": status, "best_mean_question_macro_f1": best_score[0],
                        "best_accuracy": best_score[1], **record})
        log(rank, phase="dev_checkpoint", epoch=epoch + 1, global_step=global_step, next_batch=next_batch,
            best=improved, bad_evaluations=bad_evaluations, early_stopped=stopped,
            **{key: value for key, value in metrics.items() if key != "per_question"})
        barrier(world)
        # Evaluation changes the base model and adapter dropout modes.
        wrapped.train()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, trained_rows = 0.0, 0
        accumulated_global_rows = 0
        epoch_started = time.monotonic()
        for batch_index, (inputs, candidates, metadata) in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            final = (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader)
            actual_window = min(args.gradient_accumulation,
                                len(loader) - (batch_index // args.gradient_accumulation) * args.gradient_accumulation)
            sync = contextlib.nullcontext() if final or world == 1 else wrapped.no_sync()
            labels = torch.tensor([row["target"] for row in metadata], device=device)
            with sync:
                logits = wrapped(to_device(inputs, device), candidates)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                (loss / actual_window).backward()
            loss_sum += float(loss.detach()) * len(metadata)
            trained_rows += len(metadata)
            samples_seen += len(metadata) * world
            accumulated_global_rows += len(metadata) * world
            if final:
                if not first_grad_check:
                    adapter_gradient = sum(float(p.grad.detach().float().abs().sum()) for name, p in trainable
                                           if name.startswith("backbone.") and p.grad is not None)
                    missing_adapter_gradients = sum(p.grad is None for name, p in trainable if name.startswith("backbone."))
                    head_gradient = float(model.head.weight.grad.detach().float().abs().sum())
                    if not math.isfinite(head_gradient) or head_gradient <= 0:
                        raise RuntimeError("Decision head did not receive a finite nonzero gradient")
                    if args.method == "lora" and (not math.isfinite(adapter_gradient) or adapter_gradient <= 0):
                        raise RuntimeError("LoRA adapter did not receive a finite nonzero backbone gradient")
                    if args.method == "lora" and missing_adapter_gradients:
                        raise RuntimeError("Some trainable LoRA modules are disconnected from the language-model loss")
                    log(rank, phase="gradient_check", head_gradient_sum=head_gradient,
                        backbone_adapter_gradient_sum=adapter_gradient, missing_adapter_gradients=missing_adapter_gradients)
                    first_grad_check = True
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_due = bool(args.eval_every_steps and global_step % args.eval_every_steps == 0)
                previous_samples = samples_seen - accumulated_global_rows
                sample_due = bool(args.eval_every_samples and samples_seen // args.eval_every_samples > previous_samples // args.eval_every_samples)
                accumulated_global_rows = 0
                if step_due or sample_due:
                    validate_and_save(epoch, batch_index + 1)
                    if stopped:
                        break
            if (batch_index + 1) % args.log_every == 0 or batch_index + 1 == len(loader):
                log(rank, phase="train", epoch=epoch + 1, batch=batch_index + 1, batches=len(loader),
                    representative_rank_rows=trained_rows, representative_rank_loss=loss_sum / trained_rows,
                    global_step=global_step, elapsed_seconds=round(time.monotonic() - epoch_started, 2))
        if stopped:
            break
        if last_evaluated_step != global_step:
            validate_and_save(epoch, len(loader))
        if stopped:
            break
        resume_batch = 0
        log(rank, phase="epoch_complete", epoch=epoch + 1, global_step=global_step, samples_seen=samples_seen)
    log(rank, phase="early_stopped" if stopped else "complete", elapsed_seconds=round(time.monotonic() - started, 2),
        best_score=best_score, global_step=global_step, samples_seen=samples_seen)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

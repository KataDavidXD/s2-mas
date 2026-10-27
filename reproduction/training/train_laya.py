"""Native Laya observed-choice CE, RLCD and CE-continuation training.

Training and checkpoint selection use separate train/dev inputs. Final test rows
are accepted only through --evaluate-path. Continuations reuse a named SFT best.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")



def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--initialization-name", default="official-laya-original-base")
    p.add_argument("--train-path", type=Path)
    p.add_argument("--dev-path", type=Path)
    p.add_argument("--selection-path", type=Path, help="Optional independent dev evaluation of dev-best and final candidates; never pass test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoded-dir", type=Path)
    p.add_argument("--encoding-model-path", type=Path, help="Unchanged tokenizer/config source for reusing the frozen native token cache during weight continuation")
    p.add_argument("--objective", choices=("sft", "rlcd"), default="sft")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--gradient-accumulation", type=int, default=6)
    p.add_argument("--encoder-learning-rate", type=float, default=2.5e-5)
    p.add_argument("--head-learning-rate", type=float, default=1e-4)
    p.add_argument("--eval-every-samples", type=int, default=50000)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--max-length", type=int, default=0, help="0 measures a sufficient full-input native window on CPU")
    p.add_argument("--head-max-length", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--shuffle-options", action="store_true")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--evaluate-path", type=Path)
    p.add_argument("--prepare-only", action="store_true", help="CPU-only native token measurement/encoding; no model weights or CUDA")
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()
    if args.resume and args.checkpoint:
        p.error("--resume and --checkpoint are mutually exclusive")
    if args.checkpoint and not args.evaluate_path:
        p.error("--checkpoint is for evaluation; use --model-path for an explicitly named initialization")
    if args.objective == "rlcd" and args.initialization_name == "official-laya-original-base":
        p.error("Optional additional RLCD must explicitly name its matched-SFT initialization; the default control is SFT-only")
    if args.evaluate_path and not args.checkpoint:
        p.error("Evaluation requires --checkpoint")
    if not args.evaluate_path and (not args.train_path or not args.dev_path):
        p.error("Training/preparation requires --train-path and --dev-path")
    if args.selection_path and "test" in args.selection_path.name.lower():
        p.error("Test cannot be used for checkpoint selection")
    if not args.evaluate_path and any("test" in path.name.lower() for path in (args.train_path, args.dev_path)):
        p.error("Test cannot enter training or development selection")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.gradient_accumulation, args.log_every) < 1:
        p.error("Epoch, batch, accumulation and log intervals must be positive")
    return args


if __name__ == "__main__" and any(value in {"-h", "--help"} for value in __import__("sys").argv[1:]):
    arguments()


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from laya.common import collate_items
from laya.train import draw_option_order, encode_item, load_checkpoint, rlcd_loss, save_checkpoint, soft_ce_loss
from native_inputs import prepare_encoded


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def log(rank, **value):
    if rank == 0:
        print(json.dumps(value, ensure_ascii=False), flush=True)


def barrier(world):
    if world > 1:
        dist.barrier()


def setup():
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    if not torch.cuda.is_available():
        raise RuntimeError("Training/evaluation requires CUDA; --prepare-only is CPU-only")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=120))
    return rank, world, torch.device("cuda", local)


def prepared(args, split, path, rank, world):
    directory = args.encoded_dir or args.output / "encoded"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{split}.pt"
    if rank == 0:
        package = prepare_encoded(path, args.encoding_model_path or args.model_path, destination,
                                  max_len=args.max_length, head_max_len=args.head_max_length)
    barrier(world)
    if rank != 0:
        package = torch.load(destination, map_location="cpu", weights_only=False)
    if package["rows"] != len(package["items"]):
        raise ValueError("Native cache row count differs")
    return package


class Counts:
    def __init__(self):
        self.questions, self.rows, self.correct, self.loss_sum = {}, 0, 0, 0.0

    def add(self, item, predicted, loss):
        labels = list(item["q"]["crit"])
        truth = max(range(len(item["target"])), key=item["target"].__getitem__)
        meta = item["meta"]
        key = str(meta["task_family"]) + "::" + str(meta["qid"])
        q = self.questions.setdefault(key, {"labels": labels, "matrix": [[0] * len(labels) for _ in labels]})
        if q["labels"] != labels:
            raise ValueError("Question candidates differ within a version")
        q["matrix"][truth][predicted] += 1
        self.rows += 1
        self.correct += int(truth == predicted)
        self.loss_sum += float(loss)

    def merge(self, other):
        self.rows += other["rows"]
        self.correct += other["correct"]
        self.loss_sum += other["loss_sum"]
        for key, incoming in other["questions"].items():
            q = self.questions.setdefault(key, {"labels": incoming["labels"], "matrix": [[0] * len(incoming["labels"]) for _ in incoming["labels"]]})
            if q["labels"] != incoming["labels"]:
                raise ValueError("Question candidates differ across ranks")
            for i, row in enumerate(incoming["matrix"]):
                for j, count in enumerate(row):
                    q["matrix"][i][j] += count

    def result(self):
        details = {}
        for key, q in self.questions.items():
            matrix, scores = q["matrix"], []
            n = sum(map(sum, matrix))
            correct = sum(row[i] for i, row in enumerate(matrix))
            for i, row in enumerate(matrix):
                tp = row[i]
                denominator = sum(row) + sum(r[i] for r in matrix)
                scores.append(2 * tp / denominator if denominator else 0.0)
            details[key] = {"rows": n, "accuracy": correct / n, "macro_f1": sum(scores) / len(scores), "candidate_classes": len(scores)}
        return {"rows": self.rows, "accuracy": self.correct / self.rows if self.rows else 0.0,
                "mean_question_macro_f1": sum(q["macro_f1"] for q in details.values()) / len(details) if details else 0.0,
                "loss": self.loss_sum / self.rows if self.rows else 0.0,
                "questions": len(details), "per_question": details, "invalid_predictions": 0}


def forward(model, batch, device):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(*(batch[key].to(device) for key in
                           ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
    return logits.float()


def evaluate(model, tok, package, args, rank, world, device, name, predictions=False):
    model.eval()
    items, settings = package["items"], package["settings"]
    own, counts = list(range(rank, len(items), world)), Counts()
    destination = args.output / f"{name}.rank-{rank:02d}.jsonl"
    context = destination.open("w", encoding="utf-8") if predictions else contextlib.nullcontext()
    with context as handle, torch.inference_mode():
        for offset in range(0, len(own), args.eval_batch_size):
            indices = own[offset:offset + args.eval_batch_size]
            encoded = [encode_item(tok, items[i], settings["max_len"], settings["head_max_len"]) for i in indices]
            batch = collate_items([encoded], tok.pad_token_id)
            logits = forward(model, batch, device)
            mask, targets = batch["marker_mask"].to(device), batch["target"].to(device)
            if not torch.isfinite(logits[mask]).all():
                raise FloatingPointError("Nonfinite native evaluation logits at valid candidate markers")
            losses = -(torch.log_softmax(logits.masked_fill(~mask, -1e4), -1) * targets * mask).sum(-1)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Nonfinite native evaluation loss")
            predicted = logits.masked_fill(~mask, -float("inf")).argmax(-1).tolist()
            for index, pred, loss in zip(indices, predicted, losses.tolist()):
                item = items[index]
                counts.add(item, pred, loss)
                if handle is not None:
                    meta, labels = item["meta"], list(item["q"]["crit"])
                    target = max(range(len(item["target"])), key=item["target"].__getitem__)
                    handle.write(json.dumps({"index": index, "row_id": meta["row_id"],
                        "group_id": meta["group"], "task_family": meta["task_family"], "qid": meta["qid"],
                        "candidate_ids": labels, "labels": labels, "target": target,
                        "prediction": labels[pred], "truth": labels[target]}, ensure_ascii=False) + "\n")
    if world > 1:
        packages = [None] * world
        dist.all_gather_object(packages, vars(counts))
        counts = Counts()
        for package_counts in packages:
            counts.merge(package_counts)
    if counts.rows != len(items):
        raise ValueError("Evaluation row coverage differs")
    return counts.result()


def rng_states(world):
    local = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "python": random.getstate()}
    if world == 1:
        return [local]
    states = [None] * world
    dist.all_gather_object(states, local)
    return states


def checkpoint(directory, model, tok, cfg, record, optimizer=None, scheduler=None, states=None):
    temporary = directory.with_name(f".{directory.name}.pending-{time.time_ns()}")
    save_checkpoint(model, tok, cfg, str(temporary))
    atomic_json(temporary / "metrics.json", record)
    if optimizer is not None:
        torch.save({"model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "progress": record, "rng_by_rank": states}, temporary / "training.pt")
    atomic_json(temporary / "complete.json", {key: record.get(key) for key in ("global_step", "epoch", "next_batch", "train_rows")})
    if directory.exists():
        directory.rename(directory.with_name(f".{directory.name}.previous-{time.time_ns()}"))
    temporary.rename(directory)
    previous = sorted(directory.parent.glob(f".{directory.name}.previous-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in previous[1:]:
        if not old.resolve().is_relative_to(directory.parent.resolve()):
            raise ValueError("Checkpoint cleanup escaped own output directory")
        shutil.rmtree(old)


def resolve(directory, allow_official=False):
    if directory.exists() and (directory / "complete.json").exists():
        return directory
    if (allow_official and all((directory / name).is_file() for name in
                    ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"])):
        return directory
    for p in sorted(directory.parent.glob(f".{directory.name}.previous-*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if (p / "complete.json").exists():
            return p
    raise ValueError("No complete requested native checkpoint")


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        paths = [("train", args.train_path), ("dev", args.dev_path)]
        if args.selection_path:
            paths.append(("selection", args.selection_path))
        for split, path in paths:
            package = prepared(args, split, path, 0, 1)
            log(0, phase="cpu_native_prepared", split=split, rows=package["rows"], settings=package["settings"])
        return
    rank, world, device = setup()
    started = time.monotonic()
    if rank == 0:
        atomic_json(args.output / "settings.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate_path:
        loaded = resolve(args.checkpoint, allow_official=True)
        model, tok, cfg = load_checkpoint(str(loaded))
        if hasattr(model.encoder.config, "reference_compile"):
            model.encoder.config.reference_compile = False
        model.to(device)
        package = prepared(args, "evaluation", args.evaluate_path, rank, world)
        metrics = evaluate(model, tok, package, args, rank, world, device, "evaluation", predictions=True)
        if rank == 0:
            atomic_json(args.output / "evaluation.json", metrics)
        log(rank, phase="evaluation_complete", **{k: v for k, v in metrics.items() if k != "per_question"})
        barrier(world)
        if world > 1:
            dist.destroy_process_group()
        return
    train = prepared(args, "train", args.train_path, rank, world)
    dev = prepared(args, "dev", args.dev_path, rank, world)
    selection = prepared(args, "selection", args.selection_path, rank, world) if args.selection_path else None
    model, tok, cfg = load_checkpoint(str(args.model_path))
    parent_training = dict(cfg.get("training") or {})
    matched_parent = str(cfg.get("training_stage", "")).startswith("matched-native-")
    parent_rows = int(parent_training.get("rows_processed", 0)) if matched_parent else 0
    parent_unique_rows = min(len(train["items"]), parent_rows)
    parent_run = args.model_path.parent.name if matched_parent else None
    if args.objective == "rlcd" and (not matched_parent or parent_training.get("objective") != "sft"):
        raise ValueError("This RLCD continuation must start from the matched native SFT dev-best")
    if matched_parent and cfg.get("matched_control", {}).get("unique_train_rows") != len(train["items"]):
        raise ValueError("Continuation must use the same unique financial row budget as its SFT parent")
    model.requires_grad_(True)
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
    if hasattr(model.encoder.config, "reference_compile"):
        model.encoder.config.reference_compile = False
    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    stage_name = ("matched-native-rlcd" if args.objective == "rlcd" else
                  "matched-native-sft-continued" if matched_parent else "matched-native-sft")
    ancestry = {"initialization_run": parent_run,
                "initialization_selected_checkpoint_rows_processed": parent_rows,
                "initialization_selected_checkpoint_unique_rows_seen": parent_unique_rows,
                "optimizer_reset_at_stage_start": True,
                "optimizer_resume_policy": "Only this stage's last checkpoint may restore optimizer state",
                "encoder_learning_rate": args.encoder_learning_rate,
                "head_learning_rate": args.head_learning_rate,
                "encoding_model_path": str(args.encoding_model_path or args.model_path),
                "warmup_fraction": 0.05, "scheduler": "cosine-to-5-percent-of-initial-lr",
                "rlcd_parameters": {"samples": 4, "sigma_start": 0.4, "sigma_end": 0.1,
                                    "w_sph": 0.75, "w_rps": 1.0} if args.objective == "rlcd" else None}
    cfg = dict(cfg, upstream_training=cfg.get("upstream_training", cfg.get("training")),
               initialization_training=parent_training if matched_parent else None,
               max_len=max(p["settings"]["max_len"] for p in (train, dev, selection) if p),
               head_max_len=max(p["settings"]["head_max_len"] for p in (train, dev, selection) if p),
               training_stage=stage_name,
               matched_control={"initialization": args.initialization_name, "unique_train_rows": len(train["items"]),
                   "objective": args.objective, "epochs": args.epochs, "freeze_encoder": args.freeze_encoder,
                   "shuffle_options": args.shuffle_options, "checkpoint_selection": "dev macro F1, accuracy secondary",
                   "input_policy": train["settings"].get("input_policy", "full-native-input"),
                   "historical_project_financial_continuation": False, "test_used_for_selection": False,
                   **ancestry})
    groups = [p for p in model.encoder.parameters() if p.requires_grad]
    heads = [p for name, p in model.named_parameters() if not name.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": groups, "lr": args.encoder_learning_rate},
                                   {"params": heads, "lr": args.head_learning_rate}], weight_decay=0.01)
    n = len(train["items"])
    per_rank = math.ceil(n / world)
    batches = math.ceil(per_rank / args.batch_size)
    total_steps = args.epochs * math.ceil(batches / args.gradient_accumulation)
    warmup = max(1, int(total_steps * 0.05))
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    start_epoch, next_batch, global_step, samples_seen, best_score, bad = 0, 0, 0, 0, (-1.0, -1.0), 0
    geometry = {"world_size": world, "train_rows": n, "train_path": str(args.train_path), "dev_path": str(args.dev_path),
                "batch_size": args.batch_size, "accumulation": args.gradient_accumulation,
                "seed": args.seed, "objective": args.objective, "epochs": args.epochs,
                "initialization": args.initialization_name, "shuffle_options": args.shuffle_options,
                "freeze_encoder": args.freeze_encoder, "model_path": str(args.model_path),
                "encoding_model_path": str(args.encoding_model_path or args.model_path),
                "encoder_learning_rate": args.encoder_learning_rate, "head_learning_rate": args.head_learning_rate,
                "input_settings": train["settings"]}
    if args.resume:
        state = torch.load(resolve(args.output / "last") / "training.pt", map_location="cpu", weights_only=False)
        progress = state["progress"]
        if progress["geometry"] != geometry:
            raise ValueError("Resume requires the same named inputs and training geometry")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, next_batch = progress["epoch"], progress["next_batch"]
        global_step, samples_seen, best_score, bad = progress["global_step"], progress["global_training_rows_processed"], tuple(progress["best_score"]), progress["bad_evaluations"]
        if next_batch == batches:
            start_epoch, next_batch = start_epoch + 1, 0
        elif next_batch % args.gradient_accumulation:
            raise ValueError("Resume requires an optimizer-update boundary")
        rng = state["rng_by_rank"][rank]
        torch.set_rng_state(rng["torch"].cpu())
        torch.cuda.set_rng_state(rng["cuda"].cpu())
        random.setstate(rng["python"])
    wrapped = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True, broadcast_buffers=False) if world > 1 else model
    log(rank, phase="training_start", objective=args.objective, initialization=args.initialization_name,
        train_rows=n, world_size=world, epochs=args.epochs, effective_global_batch=world * args.batch_size * args.gradient_accumulation,
        encoder_trainable_parameters=sum(p.numel() for p in groups), head_trainable_parameters=sum(p.numel() for p in heads),
        equal_rank_padding_rows=per_rank * world - n)
    last_eval = global_step if args.resume else -1
    stopped = bool(args.early_stopping_patience and bad >= args.early_stopping_patience)
    gradient_checked = False

    def validate(epoch, batch_number):
        nonlocal best_score, bad, last_eval, stopped
        metric = evaluate(model, tok, dev, args, rank, world, device, "dev")
        score = (metric["mean_question_macro_f1"], metric["accuracy"])
        improved = score > best_score
        best_score, bad = (score, 0) if improved else (best_score, bad + 1)
        last_eval = global_step
        stopped = bool(args.early_stopping_patience and bad >= args.early_stopping_patience)
        record = {"epoch": epoch, "next_batch": batch_number, "global_step": global_step,
                  "global_training_rows_processed": samples_seen, "train_rows": n, "epochs": args.epochs,
                  "best_score": best_score, "bad_evaluations": bad, "geometry": geometry,
                  "dev": metric, "elapsed_seconds": time.monotonic() - started,
                  "train_input_settings": train["settings"], "dev_input_settings": dev["settings"],
                  "train_input_checks": train.get("checks"), "dev_input_checks": dev.get("checks"),
                  "objective": args.objective, "initialization": args.initialization_name,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "world_size": world,
                  "equal_rank_padding_rows_per_pass": per_rank * world - n,
                  "cumulative_training_rows_processed": parent_rows + samples_seen,
                  "stage_unique_train_rows_seen": min(n, samples_seen),
                  "cumulative_unique_train_rows_seen": max(parent_unique_rows, min(n, samples_seen)),
                  "training_stage": stage_name, **ancestry}
        checkpoint_cfg = dict(cfg, training={"updates": global_step,
                    "epochs_completed": samples_seen / (per_rank * world),
                    "world_size": world, "objective": args.objective,
                    "unique_train_qa_available": n, "rows_processed": samples_seen,
                    "fine_tuned_from_checkpoint": True,
                    "initialization_selected_checkpoint_rows_processed": parent_rows,
                    "cumulative_rows_processed": parent_rows + samples_seen})
        states = rng_states(world)
        if rank == 0:
            if improved:
                checkpoint(args.output / "dev_best", model, tok, checkpoint_cfg, record)
            checkpoint(args.output / "last", model, tok, checkpoint_cfg, record, optimizer, scheduler, states)
            atomic_json(args.output / "result.json", {"status": "running", **record})
        barrier(world)
        wrapped.train()
        log(rank, phase="dev_checkpoint", global_step=global_step, improved=improved, bad_evaluations=bad,
            **{k: v for k, v in metric.items() if k != "per_question"})

    for epoch in range(start_epoch, args.epochs):
        if stopped:
            break
        order = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        padding = per_rank * world - n
        if padding:
            order += (order * math.ceil(padding / n))[:padding]
        own = order[rank::world]
        rng = random.Random(args.seed + epoch * world + rank)
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        seen_window = 0
        for batch_number in range(batches):
            if epoch == start_epoch and batch_number < next_batch:
                if args.shuffle_options:
                    for idx in own[batch_number * args.batch_size:(batch_number + 1) * args.batch_size]:
                        draw_option_order(train["items"][idx], rng, ("choice",))
                continue
            rows = [train["items"][idx] for idx in own[batch_number * args.batch_size:(batch_number + 1) * args.batch_size]]
            settings = train["settings"]
            encoded = [encode_item(tok, row, settings["max_len"], settings["head_max_len"],
                        option_order=draw_option_order(row, rng, ("choice",)) if args.shuffle_options else None) for row in rows]
            batch = collate_items([encoded], tok.pad_token_id)
            final = (batch_number + 1) % args.gradient_accumulation == 0 or batch_number + 1 == batches
            window_start = (batch_number // args.gradient_accumulation) * args.gradient_accumulation * args.batch_size
            window_rows = min(args.gradient_accumulation * args.batch_size, per_rank - window_start)
            with contextlib.nullcontext() if final or world == 1 else wrapped.no_sync():
                logits = forward(wrapped, batch, device)
                target, mask, qtype = (batch[key].to(device) for key in ("target", "marker_mask", "qtype"))
                ce = soft_ce_loss(logits, target, mask)
                sigma = 0.4 - 0.3 * global_step / max(1, total_steps - 1)
                loss = ce if args.objective == "sft" else rlcd_loss(logits, target, mask, qtype, sigma, samples=4, w_sph=0.75, w_rps=1.0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite native training loss")
                (loss * len(rows) / window_rows).backward()
            samples_seen += len(rows) * world
            seen_window += len(rows) * world
            if final:
                if not gradient_checked:
                    enc_grad = sum(float(p.grad.detach().float().abs().sum()) for p in groups if p.grad is not None)
                    head_grad = sum(float(p.grad.detach().float().abs().sum()) for p in heads if p.grad is not None)
                    if not math.isfinite(head_grad) or head_grad <= 0 or (not args.freeze_encoder and (not math.isfinite(enc_grad) or enc_grad <= 0)):
                        raise RuntimeError("Native encoder/head did not receive finite nonzero gradients")
                    log(rank, phase="gradient_check", encoder_gradient_sum=enc_grad, head_gradient_sum=head_grad)
                    gradient_checked = True
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                due = args.eval_every_samples and samples_seen // args.eval_every_samples > (samples_seen - seen_window) // args.eval_every_samples
                seen_window = 0
                if due:
                    validate(epoch, batch_number + 1)
                    if stopped:
                        break
            if (batch_number + 1) % args.log_every == 0 or batch_number + 1 == batches:
                log(rank, phase="train", epoch=epoch + 1, batch=batch_number + 1, batches=batches,
                    global_step=global_step, global_training_rows_processed=samples_seen,
                    loss=float(loss.detach()), elapsed_seconds=round(time.monotonic() - started, 2))
        if stopped:
            break
        if last_eval != global_step:
            validate(epoch, batches)
        next_batch = 0
        if stopped:
            break
    candidates = ["dev_best"]
    if selection:
        first = json.loads((resolve(args.output / "dev_best") / "complete.json").read_text())
        last = json.loads((resolve(args.output / "last") / "complete.json").read_text())
        if first["global_step"] != last["global_step"]:
            candidates.append("last")
    scores = {}
    for candidate in candidates:
        if selection:
            from safetensors.torch import load_file
            model.load_state_dict(load_file(str(resolve(args.output / candidate) / "model.safetensors")))
            scores[candidate] = evaluate(model, tok, selection, args, rank, world, device, "selection")
        else:
            scores[candidate] = json.loads((resolve(args.output / candidate) / "metrics.json").read_text())["dev"]
    chosen = max(scores, key=lambda candidate: (scores[candidate]["mean_question_macro_f1"], scores[candidate]["accuracy"]))
    if rank == 0:
        best = args.output / "best"
        temporary = args.output / "best.pending"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(chosen, target_is_directory=True)
        if best.exists() and not best.is_symlink():
            raise ValueError("Native best path must be a managed checkpoint link")
        os.replace(temporary, best)
        latest = json.loads((resolve(args.output / "last") / "metrics.json").read_text())
        selected_record = json.loads((resolve(args.output / chosen) / "metrics.json").read_text())
        atomic_json(args.output / "result.json", {"status": "early_stopped" if stopped else "complete", **latest,
                    "best_accuracy": scores[chosen]["accuracy"], "best_mean_question_macro_f1": scores[chosen]["mean_question_macro_f1"],
                    "selected_checkpoint": chosen, "selection_split": str(args.selection_path or args.dev_path),
                    "selected_checkpoint_training_rows_processed": selected_record["global_training_rows_processed"],
                    "stage_selected_checkpoint_unique_rows_seen": min(n, selected_record["global_training_rows_processed"]),
                    "cumulative_selected_checkpoint_unique_rows_seen": max(parent_unique_rows, min(n, selected_record["global_training_rows_processed"])),
                    "selected_checkpoint_unique_rows_seen": max(parent_unique_rows, min(n, selected_record["global_training_rows_processed"])),
                    "selected_checkpoint_cumulative_training_rows_processed": parent_rows + selected_record["global_training_rows_processed"],
                    "test_used_for_selection": False, "selection": scores,
                    "elapsed_seconds": time.monotonic() - started,
                    "method": ("native-laya-rlcd-additional-pass" if args.objective == "rlcd" else
                               "native-laya-sft-continued" if matched_parent else "native-laya-sft-only")})
    barrier(world)
    log(rank, phase="complete", selected_checkpoint=chosen, global_step=global_step, samples_seen=samples_seen)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

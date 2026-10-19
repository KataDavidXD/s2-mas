"""Construct independent synthetic personas and paired original task inputs.

Python standard library only. No inference, monthly ledgers or survey rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from generate_households import generate
from assigned_knowledge import assigned_geography_context
from scenario_builder import make_request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "s2mas-independent-persona-v1-20261019"
DEFAULT_SEED = 2026101919
TARGET_INCOME_ANNUAL_PER_PERSON_CNY = 41314
TARGET_CONSUMPTION_ANNUAL_PER_PERSON_CNY = 28227
BASE_EXPECTED_INCOME_CNY = 15500
BASE_EXPECTED_SPENDING_INCOME_FRACTION = 0.9
BASE_BLOCK_SIZE = 10000


def encoded(row):
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def make_contexts():
    vocab = json.loads((ROOT / "knowledge/cc0_vocab.json").read_text(encoding="utf-8"))
    assert len(vocab["entities"]) == 38 and len(vocab["relations"]) == 66
    ids = sorted({r["object_qid"] for r in vocab["relations"] if r["subject_qid"] == "Q148" and r["property_qid"] == "P150"}, key=lambda x: int(x[1:]))
    assert len(ids) == 33
    contexts = [{"geography": assigned_geography_context(vocab, qid), "occupation": None} for qid in ids]
    return contexts


def new_households(count, seed):
    contexts = make_contexts()
    size_rng = random.Random(seed + 10000000)
    spending_scale = (TARGET_CONSUMPTION_ANNUAL_PER_PERSON_CNY / TARGET_INCOME_ANNUAL_PER_PERSON_CNY) / BASE_EXPECTED_SPENDING_INCOME_FRACTION
    index = 0
    for block_start in range(0, count, BASE_BLOCK_SIZE):
        for base in generate(min(BASE_BLOCK_SIZE, count - block_start), seed + block_start // BASE_BLOCK_SIZE):
            size = size_rng.randint(1, 5)
            old_income = base["monthly_income_cents"]
            income = round(TARGET_INCOME_ANNUAL_PER_PERSON_CNY * 100 / 12 * size * (old_income / (BASE_EXPECTED_INCOME_CNY * 100)))
            factor = income / old_income
            debt = dict(base["opening_debt"])
            debt["principal_cents"] = round(debt["principal_cents"] * factor)
            rate = debt["annual_bps"] / 120000
            principal = debt["principal_cents"]
            debt["monthly_payment_cents"] = round(principal * rate / (1 - (1 + rate) ** -debt["term_months"])) if principal else 0
            row = {
                "household_id": f"new-persona-{index:09d}",
                "record_type": "independent_synthetic_household",
                "construction_version": VERSION,
                "household_size": size,
                "currency": "CNY",
                "money_unit": "integer_cent",
                "monthly_income_cents": income,
                "essential_spending_cents": round(base["essential_spending_cents"] * factor * spending_scale),
                "adjustable_spending_cents": round(base["adjustable_spending_cents"] * factor * spending_scale),
                "opening_cash_cents": round(base["opening_cash_cents"] * factor),
                "opening_asset_cents": round(base["opening_asset_cents"] * factor),
                "opening_debt": debt,
                "eligible_task_ids": base["eligible_task_ids"],
                "knowledge_context": contexts[index % len(contexts)],
            }
            check_household(row, index)
            yield row
            index += 1


def check_household(row, index):
    assert len(row) == 14
    assert row["household_id"] == f"new-persona-{index:09d}"
    assert row["record_type"] == "independent_synthetic_household"
    assert 1 <= row["household_size"] <= 5
    for key in ["monthly_income_cents", "essential_spending_cents", "adjustable_spending_cents", "opening_cash_cents", "opening_asset_cents"]:
        assert type(row[key]) is int and row[key] >= 0
    assert row["monthly_income_cents"] > 0
    debt = row["opening_debt"]
    assert debt["annual_bps"] in (400, 800, 1200) and debt["term_months"] in (12, 24, 36)
    assert type(debt["principal_cents"]) is int and debt["principal_cents"] >= 0
    assert type(debt["monthly_payment_cents"]) is int and debt["monthly_payment_cents"] >= 0
    assert (debt["principal_cents"] == 0) == (debt["monthly_payment_cents"] == 0)
    assert row["knowledge_context"]["occupation"] is None
    assert row["knowledge_context"]["geography"]["assignment_type"] == "assigned_synthetic_context"
    assert row["eligible_task_ids"] == [f"T{i}" for i in range(1, 9)]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_requests(path, households, designs):
    pairs = count = raw_bytes = 0
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=3, fileobj=raw, mtime=0) as writer:
            for household in households:
                for task in designs:
                    for condition in ("baseline", "intervention"):
                        row = make_request(household, task, condition)
                        body = encoded(row)
                        writer.write(body)
                        raw_bytes += len(body)
                        count += 1
                    pairs += 1
    return {"file": path.name, "requests": count, "household_task_pairs": pairs,
            "households": len(households), "uncompressed_bytes": raw_bytes,
            "compressed_bytes": path.stat().st_size,
            "model_answers": 0, "monthly_ledger_results": 0}


def build(count, seed, output, shard_size, scenario_households):
    if (output / "generation_info.json").exists() or (output / "data").exists():
        raise FileExistsError("Generated data already exists; select a fresh output directory")
    available = shutil.disk_usage(output.parent).free
    minimum_space = max(100000000, count * 500)
    if available < minimum_space:
        raise OSError(f"Need at least {minimum_space} free bytes for this compressed construction")
    output.mkdir(parents=True, exist_ok=True)
    data = output / "data"
    persona_dir = data / "personas"
    persona_dir.mkdir(parents=True)
    start = time.perf_counter()
    print(json.dumps({"state": "building", "pid": os.getpid(), "requested_households": count,
                      "shard_size": shard_size, "scenario_households": min(scenario_households, count),
                      "free_disk_bytes_at_start": available}), flush=True)
    shards, selected = [], []
    totals = {"income_cents": 0, "consumption_cents": 0, "assigned_persons": 0,
              "income_per_capita_cents": 0.0, "consumption_per_capita_cents": 0.0,
              "baseline_spending_exceeds_income_households": 0,
              "baseline_spending_and_dues_exceeds_income_households": 0}
    writer = raw = None
    shard_rows = raw_bytes = 0
    try:
        for index, row in enumerate(new_households(count, seed)):
            if index % shard_size == 0:
                if writer is not None:
                    writer.close()
                    raw.close()
                    shards.append({"file": f"data/personas/{shard_path.name}", "rows": shard_rows,
                                   "first_id": first_id, "last_id": f"new-persona-{index - 1:09d}",
                                   "uncompressed_bytes": raw_bytes, "compressed_bytes": shard_path.stat().st_size})
                    dump(output / "build_status.json", {"state": "building", "pid": os.getpid(), "completed_personas": index, "completed_shards": len(shards)})
                    print(json.dumps({"completed_personas": index, "completed_shards": len(shards), "elapsed_seconds": round(time.perf_counter() - start, 2)}), flush=True)
                shard_path = persona_dir / f"personas-{len(shards):05d}.jsonl.gz"
                raw = shard_path.open("xb")
                writer = gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=3, mtime=0)
                first_id = row["household_id"]
                shard_rows = raw_bytes = 0
            body = encoded(row)
            writer.write(body)
            shard_rows += 1
            raw_bytes += len(body)
            size = row["household_size"]
            income = row["monthly_income_cents"]
            spending = row["essential_spending_cents"] + row["adjustable_spending_cents"]
            totals["income_cents"] += income
            totals["consumption_cents"] += spending
            totals["assigned_persons"] += size
            totals["income_per_capita_cents"] += income / size
            totals["consumption_per_capita_cents"] += spending / size
            totals["baseline_spending_exceeds_income_households"] += spending > income
            totals["baseline_spending_and_dues_exceeds_income_households"] += spending + row["opening_debt"]["monthly_payment_cents"] > income
            if index < scenario_households:
                selected.append(row)
        writer.close()
        raw.close()
        writer = raw = None
        shards.append({"file": f"data/personas/{shard_path.name}", "rows": shard_rows,
                       "first_id": first_id, "last_id": f"new-persona-{count - 1:09d}",
                       "uncompressed_bytes": raw_bytes, "compressed_bytes": shard_path.stat().st_size})
    finally:
        if writer is not None:
            writer.close()
        if raw is not None:
            raw.close()
    designs = json.loads((ROOT / "designs/scenario_designs.json").read_text(encoding="utf-8"))["tasks"]
    requests = write_requests(data / "scenario_inputs.jsonl.gz", selected, designs)
    small_requests = write_requests(data / "scenario_inputs_first20.jsonl.gz", selected[:20], designs)
    with (data / "personas_first20.jsonl").open("x", encoding="utf-8") as handle:
        for row in selected[:20]:
            handle.write(encoded(row).decode("utf-8"))
    info = {
        "construction_version": VERSION, "state": "complete_local_construction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "independent_synthetic_households": count, "persona_top_level_fields": 14,
        "household_id_pattern": "new-persona-{zero_based_index:09d}",
        "shards": shards, "persona_uncompressed_bytes": sum(x["uncompressed_bytes"] for x in shards),
        "persona_compressed_bytes": sum(x["compressed_bytes"] for x in shards),
        "scenario_inputs": requests, "small_example_subset": small_requests,
        "small_example_is_subset_of_formal_scenario_inputs": True,
        "scenario_families": 8, "conditions_per_household_task": 2,
        "knowledge_entities": 38, "knowledge_relations": 66,
        "knowledge_assigned_region_concepts": 33,
        "published_aggregate_input": {"source": "real-survey aggregate reference",
            "population_unit": "resident per-capita annual CNY",
            "income_cny": TARGET_INCOME_ANNUAL_PER_PERSON_CNY,
            "consumption_cny": TARGET_CONSUMPTION_ANNUAL_PER_PERSON_CNY,
            "role": "Expectation scales only; no source household record input"},
        "authored_rng_parameters": {"base_income_shape_cny": [1000, 30000], "base_expected_income_cny": BASE_EXPECTED_INCOME_CNY,
            "base_essential_fraction": [0.3, 1.1], "base_adjustable_fraction": [0.0, 0.4],
            "spending_fraction_normalizer": BASE_EXPECTED_SPENDING_INCOME_FRACTION,
            "household_size_uniform_inclusive": [1, 5], "cash_income_months_inclusive": [0, 6],
            "asset_income_months_inclusive": [0, 12], "debt_income_months_inclusive": [0, 24],
            "annual_bps_choices": [400, 800, 1200], "contract_month_choices": [12, 24, 36],
            "independent_financial_rng_block_rows": BASE_BLOCK_SIZE,
            "knowledge_assignment": "Cycle actual 33 region concepts; not population probabilities"},
        "actual_constructed_annual_per_capita_cny": {
            "mean_of_household_per_capita_income": totals["income_per_capita_cents"] / count * 12 / 100,
            "mean_of_household_per_capita_consumption": totals["consumption_per_capita_cents"] / count * 12 / 100,
            "assigned_person_weighted_income": totals["income_cents"] / totals["assigned_persons"] * 12 / 100,
            "assigned_person_weighted_consumption": totals["consumption_cents"] / totals["assigned_persons"] * 12 / 100},
        "budget_coverage": {k: v for k, v in totals.items() if "exceeds" in k},
        "elapsed_construction_seconds": time.perf_counter() - start,
        "model_calls": 0, "model_answers": 0, "monthly_ledger_executions": 0,
        "licenses": {"new_persona_and_instantiated_inputs": "S2-MAS new synthetic research data v1",
                     "wikidata": "CC0-1.0", "authored_designs_v1": "CC-BY-4.0", "python": "Apache-2.0"},
    }
    assert sum(x["rows"] for x in shards) == count
    dump(output / "generation_info.json", info)
    dump(output / "build_status.json", {"state": "complete_local_construction", "completed_personas": count, "completed_shards": len(shards), "scenario_requests": requests["requests"]})
    print(json.dumps({"state": "complete_local_construction", "households": count,
                      "shards": len(shards), "persona_compressed_bytes": info["persona_compressed_bytes"],
                      "persona_uncompressed_bytes": info["persona_uncompressed_bytes"],
                      "scenario_requests": requests["requests"],
                      "scenario_compressed_bytes": requests["compressed_bytes"],
                      "elapsed_seconds": info["elapsed_construction_seconds"]}), flush=True)
    return info


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--count", type=int, default=20)
    cli.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--shard-size", type=int, default=50000)
    cli.add_argument("--scenario-households", type=int, default=2000)
    args = cli.parse_args()
    if not 1 <= args.count <= 2000000 or not 1 <= args.shard_size <= 100000:
        raise ValueError("This version supports 1..2M households and 1..100k rows per shard")
    if not 1 <= args.scenario_households <= 2000:
        raise ValueError("Scenario companion uses at most the first 2,000 generated households")
    build(args.count, args.seed, args.output.resolve(), args.shard_size, args.scenario_households)


if __name__ == "__main__":
    main()

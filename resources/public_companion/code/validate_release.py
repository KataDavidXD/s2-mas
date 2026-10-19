"""Read saved synthetic shards and verify ordinary IDs, states and paired inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from build_release import ROOT, check_household, encoded, new_households
from scenario_builder import make_request


def lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def check_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("_cents"):
                assert type(item) is int and item >= 0
            if key.endswith("_probability"):
                assert 0 <= item <= 1
            if isinstance(item, dict) and item.get("quotation_kind") == "contract_schedule_only":
                assert item["total_scheduled_repayment_cents"] == item["principal_cents"] + item["total_scheduled_interest_cents"]
                assert item["term_months"] > 0 and item["first_due_month"] == 1
            check_values(item)
    elif isinstance(value, list):
        for item in value:
            check_values(item)


def validate(folder):
    info = json.loads((folder / "generation_info.json").read_text(encoding="utf-8"))
    expected = info["independent_synthetic_households"]
    count = 0
    first = []
    for shard in info["shards"]:
        local_count = 0
        for row in lines(folder / shard["file"]):
            check_household(row, count)
            if count < 2000:
                first.append(row)
            count += 1
            local_count += 1
        assert local_count == shard["rows"]
        assert (folder / shard["file"]).stat().st_size == shard["compressed_bytes"]
    assert count == expected
    # A bounded direct replay checks generator/version portability, without
    # content hashes or replaying any model or household monthly ledger.
    assert first[:20] == list(new_households(min(20, expected), info["seed"]))
    designs = json.loads((ROOT / "designs/scenario_designs.json").read_text(encoding="utf-8"))["tasks"]
    task_map = {t["task_id"]: t for t in designs}
    households = {h["household_id"]: h for h in first}
    result = {}
    for filename, declared in [("scenario_inputs.jsonl.gz", info["scenario_inputs"]), ("scenario_inputs_first20.jsonl.gz", info["small_example_subset"])]:
        n = 0
        groups = Counter()
        seen = set()
        previous = None
        for row in lines(folder / "data" / filename):
            rid, hid, tid, cid = row["request_id"], row["household_id"], row["task_id"], row["condition_id"]
            assert rid not in seen
            seen.add(rid)
            assert row["household"] == households[hid]
            assert row == make_request(households[hid], task_map[tid], cid)
            assert [x["option_id"] for x in row["options"]] == ["A", "B", "C"]
            assert not ({"answer", "label", "expected", "model_response", "selected_option", "monthly_ledger", "financial_outcomes"} & set(row))
            check_values(row)
            if cid == "intervention":
                assert previous["household_id"] == hid and previous["task_id"] == tid and previous["condition_id"] == "baseline"
                assert previous["household"] == row["household"] and previous["assigned_state"] == row["assigned_state"]
            previous = row
            groups[(hid, tid, cid)] += 1
            n += 1
        assert n == declared["requests"] == declared["households"] * 16
        assert len(groups) == n and all(v == 1 for v in groups.values())
        for hid in list(households)[:declared["households"]]:
            for tid in task_map:
                assert groups[(hid, tid, "baseline")] == groups[(hid, tid, "intervention")] == 1
        result[filename] = {"requests": n, "households": declared["households"], "complete_pairs": n // 2}
    return {"saved_personas": count, "sequential_unique_ordinary_ids": True,
            "valid_assigned_financial_fields": True, "first20_direct_rng_replay": True,
            "scenario_checks": result, "model_calls": 0, "monthly_ledger_executions": 0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, required=True)
    args = cli.parse_args()
    print(json.dumps(validate(args.data_dir.resolve())), flush=True)


if __name__ == "__main__":
    main()

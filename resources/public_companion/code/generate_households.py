"""Generate independent assigned financial fixtures from explicit design ranges.

Parameter ranges are illustrative coverage choices. The generator has no input
file or external data dependency. The ledger format uses integer cents.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TASK_IDS = [f"T{i}" for i in range(1, 9)]
DEFAULT_SEED = 2026101901
PARAMETERS = {
    "currency": "CNY",
    "money_unit": "integer_cent",
    "monthly_income_cny_range": [1000, 30000],
    "essential_spending_fraction_range": [0.30, 1.10],
    "adjustable_spending_fraction_range": [0.00, 0.40],
    "opening_cash_income_months_range": [0, 6],
    "opening_asset_income_months_range": [0, 12],
    "old_debt_income_months_range": [0, 24],
    "contract_annual_bps_choices": [400, 800, 1200],
    "contract_term_month_choices": [12, 24, 36],
    "parameter_interpretation": "Assigned exploratory coverage grid",
}


def generate(count: int, seed: int):
    if not 1 <= count <= 10000:
        raise ValueError("The illustrative fixture is bounded to 1..10000 records")
    rng = random.Random(seed)
    for index in range(count):
        income = rng.randint(1000, 30000) * 100
        debt = rng.randint(0, 24) * income
        annual_bps = rng.choice(PARAMETERS["contract_annual_bps_choices"])
        term = rng.choice(PARAMETERS["contract_term_month_choices"])
        rate = annual_bps / 120000
        payment = round(debt * rate / (1 - (1 + rate) ** -term)) if debt else 0
        yield {
            "household_id": f"assigned-household-{index:08d}",
            "record_type": "assigned_random_fixture",
            "monthly_income_cents": income,
            "essential_spending_cents": round(income * rng.uniform(0.30, 1.10)),
            "adjustable_spending_cents": round(income * rng.uniform(0.00, 0.40)),
            "opening_cash_cents": rng.randint(0, 6) * income,
            "opening_asset_cents": rng.randint(0, 12) * income,
            "opening_debt": {
                "principal_cents": debt,
                "annual_bps": annual_bps,
                "term_months": term,
                "monthly_payment_cents": payment,
            },
            "eligible_task_ids": TASK_IDS,
        }


def ledger_household(row: dict) -> dict:
    """Translate the assigned fixture into the financial ledger input schema."""
    debt = row["opening_debt"]
    debts = []
    if debt["principal_cents"]:
        debts.append({
            "debt_id": "assigned-opening-contract",
            "principal_cents": debt["principal_cents"],
            "annual_rate_bps": debt["annual_bps"],
            "term_months": debt["term_months"],
            "monthly_payment_cents": debt["monthly_payment_cents"],
            "first_due_month": 0,
        })
    return {
        "household_id": row["household_id"],
        "monthly_income_cents": row["monthly_income_cents"],
        "liquid_cash_cents": row["opening_cash_cents"],
        "noncash_assets_cents": row["opening_asset_cents"],
        "essential_spending_cents": row["essential_spending_cents"],
        "adjustable_spending_cents": row["adjustable_spending_cents"],
        "existing_debts": debts,
        "weight": 1.0,
        "weight_kind": "synthetic_equal",
        "profile": {},
        "provenance": {
            "record_type": "synthetic_test",
            "field_status": "all_fields_assigned_synthetic",
            "fixture_version": "independent_rng_v1",
        },
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--count", type=int, default=20)
    cli.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cli.add_argument("--format", choices=["ledger", "tasks"], default="ledger")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    rows = list(generate(args.count, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        if args.format == "tasks":
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            payload = {
                "schema_version": "consumer_credit_oneshot_v1",
                "record_type": "synthetic_test",
                "money_unit": "CNY_cents",
                "fixture_version": "independent_rng_v1",
                "seed": args.seed,
                "parameters": PARAMETERS,
                "households": [ledger_household(row) for row in rows],
            }
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"households": len(rows), "format": args.format,
                      "fixture_version": "independent_rng_v1", "model_calls": 0}))


if __name__ == "__main__":
    main()

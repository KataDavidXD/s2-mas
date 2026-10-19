"""Build original English financial decision-input examples.

The only random draws are in generate_households.py. Contract quotations and
scenario expansion are deterministic. No inference or household ledger runs.
"""
from __future__ import annotations

import argparse
import copy
import json
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path

from generate_households import DEFAULT_SEED, PARAMETERS, TASK_IDS, generate


CONDITIONS = ("baseline", "intervention")
DEFAULT_COUNT = 20


def rounded(value):
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def money(cents):
    return f"CNY {cents / 100:,.2f}"


def contract(principal, annual_bps, term_months, fee=0):
    """Quote a contract; these scheduled amounts are not realized payments."""
    assert principal >= 0 and annual_bps >= 0 and term_months > 0 and fee >= 0
    with localcontext() as ctx:
        ctx.prec = 32
        monthly_rate = Decimal(annual_bps) / Decimal(120000)
        p = Decimal(principal)
        payment = rounded(p * monthly_rate / (1 - (1 + monthly_rate) ** -term_months)) if annual_bps and principal else rounded(p / term_months)
        remaining = principal
        total = 0
        final = 0
        for month in range(1, term_months + 1):
            interest = rounded(Decimal(remaining) * monthly_rate)
            due = remaining + interest if month == term_months else min(payment, remaining + interest)
            remaining = max(0, remaining + interest - due)
            total += due
            final = due
    return {
        "principal_cents": principal,
        "annual_bps": annual_bps,
        "term_months": term_months,
        "fee_cents": fee,
        "fee_payment": "upfront_from_cash",
        "first_due_month": 1,
        "regular_payment_cents": payment,
        "final_payment_cents": final,
        "total_scheduled_repayment_cents": total,
        "total_scheduled_interest_cents": total - principal,
        "quotation_kind": "contract_schedule_only",
    }


def contract_text(q):
    return (f"principal {money(q['principal_cents'])}, nominal annual rate "
            f"{q['annual_bps'] / 100:.2f}%, {q['term_months']} months, "
            f"upfront fee {money(q['fee_cents'])}, regular monthly payment "
            f"{money(q['regular_payment_cents'])}, final payment "
            f"{money(q['final_payment_cents'])}; total scheduled repayment "
            f"{money(q['total_scheduled_repayment_cents'])} plus the upfront fee")


def make_option(option_id, text, action, **fields):
    return {"option_id": option_id, "text": text, "action": action, **fields}


def make_request(h, task, condition_id):
    task_id = task["task_id"]
    income = h["monthly_income_cents"]
    cash = h["opening_cash_cents"]
    condition = copy.deepcopy(task[condition_id])
    fixed = {"horizon_months": 12, "decision_time_month": 0}
    state = {}
    options = []

    if task_id == "T1":
        fixed.update({"income_path_applies_months": [1, 12], "opening_debt_continues": True})
        state["effective_monthly_income_cents"] = rounded(Decimal(income) * Decimal(str(condition["monthly_income_multiplier"])))
        for letter, fraction in zip("ABC", (1.0, 0.5, 0.0)):
            amount = rounded(Decimal(h["adjustable_spending_cents"]) * Decimal(str(fraction)))
            options.append(make_option(letter, f"Set monthly discretionary spending to {money(amount)} ({fraction:.0%} of the displayed starting amount).", "set_discretionary_spending", discretionary_fraction=fraction, monthly_discretionary_cents=amount))
        question = f"Choose a discretionary-spending plan for months 1–12. Monthly income will be {money(state['effective_monthly_income_cents'])}; other displayed expenses and debt contracts continue."

    elif task_id == "T2":
        fixed.update({"bill_due_month": 0, "credit_fee_cents": 0, "use_opening_cash_before_borrowing": True})
        bill = income * condition["bill_income_months"]
        gap = max(0, bill - cash)
        state.update({"one_time_bill_cents": bill, "cash_paid_toward_bill_cents": min(cash, bill), "funding_gap_cents": gap})
        options.append(make_option("A", "Use opening cash only. Leave any remaining bill unpaid and record the shortfall.", "cash_only_record_any_unpaid_bill"))
        for letter, term in (("B", 12), ("C", 24)):
            q = contract(gap, condition["assigned_credit_annual_bps"], term)
            options.append(make_option(letter, f"Use opening cash and finance the remaining gap with this contract: {contract_text(q)}.", "cash_plus_gap_loan", credit_contract=q))
        question = f"Choose a payment plan for a one-time bill of {money(bill)}, due now. Opening cash is {money(cash)}; any gap is {money(gap)}."

    elif task_id == "T3":
        fixed.update({"planned_expense_due_month": 0, "credit_annual_bps": 800, "credit_term_months": 12, "credit_fee_cents": 0})
        expense = income * condition["planned_expense_income_months"]
        limit = income * condition["line_income_months"]
        gap = max(0, expense - cash)
        state.update({"planned_expense_cents": expense, "credit_limit_cents": limit, "expense_funding_gap_cents": gap})
        for letter, draw, action, label in (("A", 0, "draw_zero", "Draw no credit"), ("B", min(gap, limit), "draw_positive_expense_gap_capped_at_line", "Draw only the expense gap, capped at the line"), ("C", limit, "draw_full_line", "Draw the full line")):
            q = contract(draw, 800, 12)
            options.append(make_option(letter, f"{label}: {money(draw)}. {contract_text(q)}.", action, draw_cents=draw, credit_contract=q))
        question = f"Choose a draw for a planned expense of {money(expense)}, due now, with opening cash {money(cash)} and an available credit line of {money(limit)}."

    elif task_id == "T4":
        fixed.update({"requested_principal_cents": income, "credit_fee_cents": 0, "principal_disbursed_month": 0})
        state["requested_principal_cents"] = income
        options.append(make_option("A", "Decline the proposed credit; do not add principal or contractual payments.", "decline_credit"))
        for letter, term, rate_key, action in (("B", 12, "short_contract_annual_bps", "accept_short_contract"), ("C", 24, "long_contract_annual_bps", "accept_long_contract")):
            q = contract(income, condition[rate_key], term)
            options.append(make_option(letter, f"Accept this contract: {contract_text(q)}.", action, credit_contract=q))
        question = f"Choose a contract for the same requested principal of {money(income)}, or decline the credit. Compare the complete quoted terms."

    elif task_id == "T5":
        fixed.update({"investment_realization_month": 12, "gain_probability": 0.5, "loss_probability": 0.5, "realization_drawn": False, "investment_fee_cents": 0})
        protected = income * condition["protected_bill_income_months"]
        available = max(0, cash - protected)
        state.update({"protected_bill_cents": protected, "investable_cash_cents": available})
        for letter, fraction in zip("ABC", (0.0, 0.1, 0.25)):
            amount = rounded(Decimal(available) * Decimal(str(fraction)))
            options.append(make_option(letter, f"Allocate {fraction:.0%} of investable cash, {money(amount)}, to the investment. Keep the remainder liquid.", "invest_available_cash_fraction", fraction=fraction, allocation_cents=amount))
        question = f"Choose an investment allocation after protecting a bill of {money(protected)}. Investable cash is {money(available)}. At month 12, the investment has a designer-assigned 50% chance of a 10% gain and a 50% chance of a 10% loss. No return is drawn in this example."

    elif task_id == "T6":
        higher = {"debt_id": "assigned-higher-rate", "due_day": 20, **contract(income, 1200, 12)}
        earlier = {"debt_id": "assigned-earlier-due", "due_day": 2, **contract(income, 400, 12)}
        full_due = higher["regular_payment_cents"] + earlier["regular_payment_cents"]
        fixed.update({"task_debts": [higher, earlier], "debt_overlay": "These two assigned contracts replace the opening-debt contract for this allocation exercise.", "assigned_repayment_cash_pool_cents": full_due})
        budget = rounded(Decimal(full_due) * Decimal(str(condition["repayment_budget_due_fraction"])))
        state.update({"repayment_budget_cents": budget, "total_current_contractual_due_cents": full_due})
        options = [
            make_option("A", "Allocate the cash budget in proportion to each current contractual due, capped at each due; assign any rounding cent to the higher-rate debt.", "allocate_proportional_to_due"),
            make_option("B", "Pay the 12% annual-rate debt first, up to its current due, then pay the 4% annual-rate debt with remaining budget.", "allocate_higher_rate_debt_first"),
            make_option("C", "Pay the debt due on day 2 first, up to its current due, then pay the debt due on day 20 with remaining budget.", "allocate_earlier_due_debt_first"),
        ]
        question = f"Allocate an assigned cash repayment budget of {money(budget)} between two debts. The 12% debt has {money(higher['regular_payment_cents'])} due on day 20; the 4% debt has {money(earlier['regular_payment_cents'])} due on day 2. Track unpaid dues separately."

    elif task_id == "T7":
        principal = max(income, h["opening_debt"]["principal_cents"])
        annual_bps = h["opening_debt"]["annual_bps"]
        current = contract(principal, annual_bps, 12)
        spare = max(0, cash - income)
        prepay = min(principal, rounded(Decimal(spare) * Decimal("0.5")))
        fixed.update({"task_current_contract": current, "debt_overlay": "This assigned remaining contract replaces the opening-debt contract for this schedule exercise.", "protected_cash_cents": income, "partial_prepayment_cents": prepay, "prepayment_fee_cents": 0})
        extension = contract(principal, annual_bps, condition["extended_remaining_term_months"], condition["extension_fee_cents"])
        requote = contract(principal - prepay, annual_bps, 12)
        state.update({"extension_contract": extension, "spare_cash_cents": spare})
        options = [
            make_option("A", f"Keep the current schedule: {contract_text(current)}.", "retain_current_schedule", credit_contract=current),
            make_option("B", f"Request the extension: {contract_text(extension)}. Activate it only if available opening cash can pay the upfront fee; otherwise retain the current schedule and record the rejected extension.", "accept_disclosed_extension", credit_contract=extension),
            make_option("C", f"Prepay {money(prepay)} from half of spare opening cash, then follow this remaining contract: {contract_text(requote)}.", "make_feasible_partial_prepayment_then_requote", prepayment_cents=prepay, spare_cash_fraction=0.5, credit_contract=requote),
        ]
        question = f"Choose a repayment schedule for an assigned remaining principal of {money(principal)} at {annual_bps/100:.2f}% annual interest. Protect one income-month of cash, {money(income)}, before any partial prepayment."

    elif task_id == "T8":
        asset = h["opening_asset_cents"]
        bill = income
        sale = rounded(Decimal(asset) * Decimal(str(condition["asset_sale_fraction_of_book_value"])))
        gap = max(0, bill - cash)
        q = contract(gap, 800, 12)
        fixed.update({"dated_payment_cents": bill, "payment_due_day": 3, "asset_book_value_cents": asset, "sale_request_day": 0, "credit_disbursement_day": 1, "credit_fee_cents": 0})
        state.update({"quoted_sale_proceeds_cents": sale, "asset_settlement_day": condition["asset_settlement_days"], "cash_payment_gap_cents": gap})
        options = [
            make_option("A", "Keep the asset. Use available opening cash for the day-3 payment and record any unpaid amount.", "retain_asset_use_cash_and_record_any_shortfall"),
            make_option("B", f"Sell the asset for {money(sale)}, arriving on day {condition['asset_settlement_days']}. Use only funds received by day 3 for the dated payment and record any remaining day-3 shortfall.", "sell_assigned_asset", sale_proceeds_cents=sale, settlement_day=condition["asset_settlement_days"]),
            make_option("C", f"Keep the asset and refinance the positive cash gap with funds available on day 1: {contract_text(q)}.", "refinance_disclosed_gap", credit_contract=q),
        ]
        question = f"Choose a route for a payment of {money(bill)}, due on day 3. Opening cash is {money(cash)}. The assigned asset has book value {money(asset)} and the sale terms shown below."
    else:
        raise ValueError(task_id)

    return {
        "request_id": f"{h['household_id']}-{task_id}-{condition_id}",
        "household_id": h["household_id"],
        "task_id": task_id,
        "condition_id": condition_id,
        "record_type": "synthetic_decision_input_example",
        "language": "en",
        "currency": "CNY",
        "money_unit": "integer_cent",
        "instruction": "Select exactly one option_id from A, B or C, using the household, assigned state and displayed condition. Return the option_id only.",
        "household": copy.deepcopy(h),
        "assigned_state": fixed,
        "condition": condition,
        "effective_state": state,
        "question": question,
        "options": options,
        "execution_rules": task["execution_rules"],
    }

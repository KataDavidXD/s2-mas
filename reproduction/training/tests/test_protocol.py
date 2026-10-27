"""Offline CPU checks using synthetic choice records."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
import merge_predictions
import paired_metrics
import prepare_data
import prepare_native

try:
    import torch
except ImportError:
    torch = None


def prepared_row():
    return {
        "row_id": "decision-row-A",
        "group_id": "household-A",
        "task_family": "payment_planning",
        "qid": "payment_choice_v1",
        "state": {"household": "A", "cash": 12, "monthly": {"income": 8, "cost": 5}},
        "question": {
            "type": "choice",
            "instructions": "Which payment would this household select?",
            "criteria": {"B": "Keep cash", "A": "Pay now"},
        },
        "candidate_ids": ["B", "A"],
        "target": 1,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CandidateInputTests(unittest.TestCase):
    def test_candidate_contains_complete_state_question_options_and_focus(self):
        row = prepared_row()
        rendered = common.prompt(row, "A")
        state_text = rendered.split("Respondent state:\n", 1)[1].split("\nQuestion:\n", 1)[0]
        options_text = rendered.split("All allowed options:\n", 1)[1].split("\nCandidate being scored:\n", 1)[0]
        focus_text = rendered.split("Candidate being scored:\n", 1)[1].split("\nCandidate compatibility", 1)[0]
        self.assertEqual(json.loads(state_text), row["state"])
        self.assertIn(row["question"]["instructions"], rendered)
        self.assertEqual(json.loads(options_text), row["question"]["criteria"])
        self.assertEqual(list(json.loads(options_text)), ["B", "A"])
        self.assertEqual(json.loads(focus_text), {"id": "A", "description": "Pay now"})

    def test_mismatched_candidate_set_is_rejected(self):
        row = prepared_row()
        row["candidate_ids"] = ["B", "C"]
        with self.assertRaises(ValueError):
            common.validate_row(row)

    @unittest.skipIf(torch is None, "PyTorch is optional for offline protocol checks")
    def test_pooling_uses_last_nonpadding_token_then_l2_normalizes(self):
        hidden = torch.tensor([
            [[1., 0.], [3., 4.], [99., 0.], [0., 99.]],
            [[1., 0.], [0., 1.], [-8., 6.], [99., 99.]],
        ])
        encoder = lambda **kwargs: SimpleNamespace(last_hidden_state=hidden)
        backbone = SimpleNamespace(model=SimpleNamespace(language_model=encoder))
        tokens = {"input_ids": torch.zeros((2, 4), dtype=torch.long),
                  "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])}
        result = common.pooled_hidden(backbone, tokens)
        torch.testing.assert_close(result, torch.tensor([[.6, .8], [-.8, .6]]))
        self.assertEqual(result.device.type, "cpu")

    @unittest.skipIf(torch is None, "PyTorch is optional for offline protocol checks")
    def test_candidate_softmax_excludes_padding_for_unequal_option_counts(self):
        logits = common.candidate_logits(torch.zeros(5), [2, 3])
        probability = torch.softmax(logits, dim=-1)
        torch.testing.assert_close(probability[0], torch.tensor([.5, .5, 0.]))
        torch.testing.assert_close(probability[1], torch.tensor([1 / 3, 1 / 3, 1 / 3]))


class PreparationTests(unittest.TestCase):
    def test_stratified_samples_are_nested_and_do_not_select_another_split(self):
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        connection.execute("CREATE TABLE rows(split TEXT,ordinal INTEGER,row_id TEXT,task_family TEXT,qid TEXT,score REAL,PRIMARY KEY(split,row_id))")
        counts = {("payment_planning", "payment_choice_v1"): 6, ("saving_planning", "saving_choice_v1"): 4}
        for ordinal in range(10):
            task_family, qid = ("payment_planning", "payment_choice_v1") if ordinal < 6 else ("saving_planning", "saving_choice_v1")
            connection.execute("INSERT INTO rows VALUES(?,?,?,?,?,?)",
                               ("train", ordinal, f"train-row-{ordinal}", task_family, qid, -ordinal))
            connection.execute("INSERT INTO rows VALUES(?,?,?,?,?,?)",
                               ("dev", ordinal + 100, f"dev-row-{ordinal}", task_family, qid, -1000 - ordinal))
        selected, allocation = prepare_data.select_nested(connection, "train", counts, [8, 4])
        self.assertEqual({size: len(ids) for size, ids in selected.items()}, {4: 4, 8: 8})
        self.assertLessEqual(selected[4], selected[8])
        self.assertLessEqual(selected[8], set(range(10)))
        self.assertEqual(allocation["4"], {"payment_planning:payment_choice_v1": 2, "saving_planning:saving_choice_v1": 2})
        self.assertEqual(selected[4], {4, 5, 8, 9})
        self.assertEqual(sum(prepare_data.quotas(8, counts).values()), 8)

    def test_native_round_trip_preserves_ordinary_ids_state_and_candidate_order(self):
        original = prepared_row()
        source = {
            "state": original["state"],
            "questions": {original["qid"]: original["question"]},
            "expected": {original["qid"]: {"choice": "A"}},
            "meta": {"row_id": "decision-row-A", "group": "household-A", "task_family": "payment_planning"},
        }
        converted = prepare_data.convert(source)
        self.assertEqual(converted, original)
        native = prepare_native.to_native(converted)
        self.assertEqual(native["meta"]["row_id"], original["row_id"])
        self.assertNotIn("shard", native["meta"])
        self.assertEqual(native["meta"]["group"], original["group_id"])
        self.assertEqual(native["state"], original["state"])
        self.assertEqual(list(native["questions"][original["qid"]]["criteria"]), ["B", "A"])
        self.assertEqual(native["expected"], {original["qid"]: "A"})

    def test_native_mapping_rejects_changed_candidate_order(self):
        row = prepared_row()
        row["candidate_ids"] = ["A", "B"]
        with self.assertRaises(ValueError):
            prepare_native.to_native(row)

    def test_preparation_requires_caller_row_id_instead_of_constructing_a_locator(self):
        original = prepared_row()
        source = {"state": original["state"], "questions": {original["qid"]: original["question"]},
                  "expected": {original["qid"]: "A"},
                  "meta": {"group": original["group_id"], "task_family": original["task_family"]}}
        with self.assertRaises(KeyError):
            prepare_data.convert(source)

    def test_duplicate_ordinary_row_id_is_rejected_within_partition(self):
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        connection.execute("CREATE TABLE rows(split TEXT,ordinal INTEGER,row_id TEXT,task_family TEXT,qid TEXT,score REAL,PRIMARY KEY(split,row_id))")
        prepare_data.insert_rows(connection, [("train", 0, "decision-A", "payment", "choice-v1", .1)])
        with self.assertRaises(ValueError):
            prepare_data.insert_rows(connection, [("train", 1, "decision-A", "payment", "choice-v1", .2)])


class PairedMetricTests(unittest.TestCase):
    def test_f1_includes_unobserved_candidates_and_invalid_keeps_denominator(self):
        result = paired_metrics.summarize(
            np.array([0, 0, 1, 1]), np.array([0, 3, 1, 0]),
            ["choice"], {"choice": ["A", "B", "C"]}, np.array([0, 3]))
        self.assertEqual((result["n"], result["valid"], result["invalid"]), (4, 3, 1))
        self.assertEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["mean_question_macro_f1"], 7 / 18)
        self.assertEqual(result["by_question"]["choice"]["class_f1"]["C"], 0.)
        self.assertEqual(result["by_question"]["choice"]["confusion_true_rows_predicted_columns_plus_invalid"],
                         [[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]])

    def test_bootstrap_resamples_whole_households_and_pairs_models(self):
        truth = np.array([0, 0, 0, 0])
        predictions = {"baseline": np.array([0, 0, 2, 2]),
                       "model": np.array([2, 2, 0, 0])}
        groups = np.array([0, 0, 1, 1])
        samples = paired_metrics.bootstrap(truth, predictions, groups, np.array([0, 2]), 128, 17)
        self.assertEqual(set(samples["baseline"][:, 0]), {0., .5, 1.})
        np.testing.assert_allclose(samples["baseline"][:, 0] + samples["model"][:, 0], 1.)
        repeated = paired_metrics.bootstrap(
            np.repeat(truth, 2), {key: np.repeat(value, 2) for key, value in predictions.items()},
            np.repeat(groups, 2), np.array([0, 2]), 128, 17)
        for name in predictions:
            np.testing.assert_allclose(samples[name], repeated[name])

    def test_prediction_pairing_rejects_duplicate_ids_and_changed_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            test = [{"qid": "choice", "labels": ["B", "A"], "truth": "A", "group": "household-A"}]
            row = {"index": 0, "qid": "choice", "labels": ["B", "A"], "truth": "A", "prediction": None}
            write_jsonl(path, [row])
            self.assertEqual(paired_metrics.read_predictions(path, test, "model"), {0: None})
            write_jsonl(path, [row, row])
            with self.assertRaises(ValueError):
                paired_metrics.read_predictions(path, test, "model")
            changed = copy.deepcopy(row)
            changed["truth"] = "B"
            write_jsonl(path, [changed])
            with self.assertRaises(ValueError):
                paired_metrics.read_predictions(path, test, "model")


class PredictionMergeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows = [{"index": index, "qid": "choice-" + str(index % 2),
                      "labels": ["A", "B"], "truth": "A", "prediction": "B"}
                     for index in range(4)]
        self.first = self.root / "evaluation.rank-0.jsonl"
        self.second = self.root / "evaluation.rank-1.jsonl"

    def test_complete_out_of_order_ranks_keep_all_indexed_predictions(self):
        write_jsonl(self.first, [self.rows[2], self.rows[0]])
        write_jsonl(self.second, [self.rows[3], self.rows[1]])
        output = self.root / "predictions.jsonl.gz"
        result = merge_predictions.merge([self.second, self.first], output, 4, 2)
        with gzip.open(output, "rt", encoding="utf-8") as source:
            actual = [json.loads(line) for line in source]
        self.assertEqual((result["rows"], result["questions"]), (4, 2))
        self.assertEqual(sorted(actual, key=lambda row: row["index"]), self.rows)

    def test_duplicate_index_is_rejected_without_overwriting_existing_output(self):
        write_jsonl(self.first, self.rows[:2])
        write_jsonl(self.second, self.rows[1:])
        output = self.root / "predictions.jsonl"
        output.write_text("previous complete output\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            merge_predictions.merge([self.first, self.second], output, 4, 2)
        self.assertEqual(output.read_text(encoding="utf-8"), "previous complete output\n")
        self.assertFalse(output.with_name(output.name + ".pending").exists())

    def test_missing_index_prevents_publishing_a_merged_result(self):
        write_jsonl(self.first, self.rows[:2])
        write_jsonl(self.second, [self.rows[3]])
        output = self.root / "predictions.jsonl"
        with self.assertRaises(ValueError):
            merge_predictions.merge([self.first, self.second], output, 4, 2)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + ".pending").exists())


if __name__ == "__main__":
    unittest.main()

"""Regression checks for the evidence checks used by the manuscript audit."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "paper_audit", Path(__file__).resolve().parents[1] / "scripts/audit_paper_numbers.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PredictionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.source = [{"sample_id": "clip", "canonical_label": "A", "signer": "S1",
                        "sign_type": "vowel", "lighting": "bright"}]
        self.prediction = {"sample_id": "clip", "true_index": "0", "predicted_index": "1",
                           "true_label": "A", "predicted_label": "B", "correct": "0",
                           "signer": "S1", "sign_type": "vowel", "lighting": "bright"}

    def test_stored_correctness_cannot_hide_wrong_prediction(self):
        audit.validate_prediction_rows([self.prediction], self.source, ["A", "B"])
        self.prediction["correct"] = "1"
        with self.assertRaisesRegex(AssertionError, "correctness flag"):
            audit.validate_prediction_rows([self.prediction], self.source, ["A", "B"])

    def test_duplicate_or_relabelled_clip_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "sample IDs"):
            audit.validate_prediction_rows([self.prediction, self.prediction], self.source, ["A", "B"])
        self.source[0]["canonical_label"] = "B"
        with self.assertRaisesRegex(AssertionError, "truth label"):
            audit.validate_prediction_rows([self.prediction], self.source, ["A", "B"])


if __name__ == "__main__":
    unittest.main()

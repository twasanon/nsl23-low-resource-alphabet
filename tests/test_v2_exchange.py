from __future__ import annotations

import unittest

from nsl23.v2_annotation import SEGMENT_FIELDS, segment_id
from nsl23.v2_exchange import merge_pass2_rows, sealed_pass2_rows


def master_row() -> dict[str, str]:
    row = {field: "" for field in SEGMENT_FIELDS}
    row.update(
        {
            "segment_id": segment_id("tape.mov", 1),
            "source_video": "tape.mov",
            "source_video_hash": "a" * 64,
            "source_path": "data/extracted/tape.mov",
            "signer": "S5",
            "class": "KA",
            "start_frame": "30",
            "end_frame": "60",
            "start_time_s": "1",
            "end_time_s": "2",
            "fps": "30",
            "duration_seconds": "10",
            "condition": "prepared",
            "lighting": "unknown",
            "capture_device": "camera",
            "sign_type": "consonant",
            "proposal_method": "motion_hold",
            "hold_index": "1",
            "n_holds_detected": "36",
            "expected_class_hint": "KA",
            "proposal_start_frame": "24",
            "proposal_end_frame": "66",
            "proposal_start_time_s": "0.8",
            "proposal_end_time_s": "2.2",
            "review_pass_1": "accepted",
            "pass1_reviewer": "Reviewer One",
            "pass1_timestamp": "2026-08-23T10:00:00Z",
            "pass1_class": "KA",
            "pass1_start_frame": "30",
            "pass1_end_frame": "60",
            "pass1_start_time_s": "1",
            "pass1_end_time_s": "2",
        }
    )
    return row


class V2ExchangeTests(unittest.TestCase):
    def test_import_rejects_nonfinite_and_negative_bounds(self) -> None:
        master = master_row()
        for changes in ({"pass2_end_time_s": "nan"}, {"pass2_end_time_s": "inf"},
                        {"pass2_start_frame": "-60", "pass2_end_frame": "-30"}):
            with self.subTest(changes=changes):
                returned = sealed_pass2_rows([master])[0]
                returned.update({"review_pass_2": "accepted", "pass2_reviewer": "Reviewer Two",
                                 "pass2_timestamp": "2026-10-11T11:00:00Z", "pass2_class": "KA",
                                 "pass2_start_frame": "30", "pass2_end_frame": "60",
                                 "pass2_start_time_s": "1", "pass2_end_time_s": "2"})
                returned.update(changes)
                with self.assertRaisesRegex(ValueError, "pass2_requires_valid_bounds"):
                    merge_pass2_rows([master], [returned], consonants={"KA"}, vowels={"A"})

    def test_sealed_export_contains_no_pass1_or_hint(self) -> None:
        exported = sealed_pass2_rows([master_row()])[0]
        self.assertEqual(exported["class"], "")
        self.assertEqual(exported["expected_class_hint"], "")
        self.assertEqual(exported["review_pass_1"], "")
        self.assertEqual(exported["pass1_class"], "")
        self.assertEqual(exported["start_time_s"], "0.8")
        self.assertEqual(exported["end_time_s"], "2.2")

    def test_import_copies_only_independent_pass2_fields(self) -> None:
        master = master_row()
        returned = sealed_pass2_rows([master])[0]
        returned.update(
            {
                "review_pass_2": "accepted",
                "pass2_reviewer": "Reviewer Two",
                "pass2_timestamp": "2026-08-23T11:00:00Z",
                "pass2_class": "KA",
                "pass2_start_frame": "31",
                "pass2_end_frame": "61",
                "pass2_start_time_s": "1.0333",
                "pass2_end_time_s": "2.0333",
            }
        )
        merged, report = merge_pass2_rows(
            [master],
            [returned],
            consonants={"KA"},
            vowels={"A"},
            reviewer="Reviewer Two",
        )
        self.assertEqual(report["imported"], 1)
        self.assertEqual(merged[0]["pass1_class"], "KA")
        self.assertEqual(merged[0]["pass2_class"], "KA")

    def test_import_rejects_proposal_tampering_and_same_reviewer(self) -> None:
        master = master_row()
        returned = sealed_pass2_rows([master])[0]
        returned["proposal_start_time_s"] = "0.1"
        with self.assertRaisesRegex(ValueError, "immutable proposal field changed"):
            merge_pass2_rows(
                [master], [returned], consonants={"KA"}, vowels={"A"}, require_complete=False
            )
        returned = sealed_pass2_rows([master])[0]
        returned.update(
            {
                "review_pass_2": "rejected",
                "pass2_reviewer": "Reviewer One",
                "pass2_timestamp": "2026-08-23T11:00:00Z",
            }
        )
        with self.assertRaisesRegex(ValueError, "independent_passes_require_two_reviewers"):
            merge_pass2_rows(
                [master], [returned], consonants={"KA"}, vowels={"A"}
            )

    def test_pass2_manual_insertion_is_added_without_a_pass1_label(self) -> None:
        master = master_row()
        returned = sealed_pass2_rows([master])[0]
        returned.update(
            {
                "review_pass_2": "rejected",
                "pass2_reviewer": "Reviewer Two",
                "pass2_timestamp": "2026-08-23T11:00:00Z",
            }
        )
        inserted = dict(returned)
        inserted.update(
            {
                "segment_id": segment_id("tape.mov", 2),
                "proposal_method": "manual_insert",
                "hold_index": "2",
                "proposal_start_frame": "90",
                "proposal_end_frame": "120",
                "proposal_start_time_s": "3",
                "proposal_end_time_s": "4",
                "start_frame": "90",
                "end_frame": "120",
                "start_time_s": "3",
                "end_time_s": "4",
                "review_pass_2": "accepted",
                "pass2_class": "KA",
                "pass2_start_frame": "90",
                "pass2_end_frame": "120",
                "pass2_start_time_s": "3",
                "pass2_end_time_s": "4",
            }
        )
        merged, report = merge_pass2_rows(
            [master],
            [returned, inserted],
            consonants={"KA"},
            vowels={"A"},
            reviewer="Reviewer Two",
        )
        self.assertEqual(report["added_manual_inserts"], 1)
        added = next(row for row in merged if row["segment_id"] == inserted["segment_id"])
        self.assertEqual(added["review_pass_1"], "")
        self.assertEqual(added["pass2_class"], "KA")

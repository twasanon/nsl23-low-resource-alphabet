from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nsl23.landmarks import palm_canonical_world_landmarks
from nsl23.v2_annotation import (
    SEGMENT_FIELDS,
    agreement_projection_sha256,
    assign_testers,
    hierarchy_sha256,
    sha256_file,
    validate_verified_row,
    validate_review_completion,
    verification_ok,
)
from nsl23.v2_features import load_feature_clip
from nsl23.v2_metrics import hierarchical_classification_metrics
from nsl23.v2_protocol import (
    load_and_validate_frozen_protocol,
    protocol_sha256,
    validate_manifest_provenance,
)


def landmark_object(points: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        landmark=[SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in points]
    )


class V2AnnotationTests(unittest.TestCase):
    def valid_row(self, source: Path) -> dict[str, str]:
        row = {field: "" for field in SEGMENT_FIELDS}
        row.update(
            {
                "segment_id": "seg_test_001",
                "source_video": "test.mov",
                "source_video_hash": sha256_file(source),
                "source_path": str(source),
                "signer": "S1",
                "class": "KA",
                "sign_type": "consonant",
                "start_frame": "0",
                "end_frame": "30",
                "start_time_s": "0",
                "end_time_s": "1",
                "fps": "30",
                "review_pass_1": "accepted",
                "pass1_reviewer": "Reviewer A",
                "pass1_timestamp": "2026-08-20T12:00:00Z",
                "pass1_class": "KA",
                "pass1_start_time_s": "0",
                "pass1_end_time_s": "1",
                "review_pass_2": "accepted",
                "pass2_reviewer": "Reviewer B",
                "pass2_timestamp": "2026-08-20T13:00:00Z",
                "pass2_class": "KA",
                "pass2_start_time_s": "0",
                "pass2_end_time_s": "1",
            }
        )
        return row

    def test_all_complete_test_candidates_stay_test(self) -> None:
        self.assertEqual(assign_testers(["S6", "S5", "S14", "S4"]), ["S5", "S14", "S4", "S6"])

    def test_nonfinite_review_bounds_and_metadata_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video.mov"
            source.write_bytes(b"video")
            for value in ("nan", "inf", "-inf"):
                with self.subTest(value=value):
                    row = self.valid_row(source)
                    row["pass2_end_time_s"] = value
                    self.assertIn("pass2_accepted_without_valid_bounds", validate_review_completion(row))
                    row = self.valid_row(source)
                    row["fps"] = value
                    self.assertIn("bad_fps", validate_verified_row(row, {"KA"}, {"A"}, 2.0))
            row = self.valid_row(source)
            self.assertIn("bad_duration", validate_verified_row(row, {"KA"}, {"A"}, float("nan")))

    def test_verified_row_requires_real_hash_and_named_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video.mov"
            source.write_bytes(b"video")
            row = self.valid_row(source)
            self.assertEqual(verification_ok(row), (True, ""))
            self.assertEqual(validate_verified_row(row, {"KA"}, {"A"}, 2.0), [])
            row["source_video_hash"] = "0" * 64
            self.assertIn(
                "source_hash_mismatch",
                validate_verified_row(row, {"KA"}, {"A"}, 2.0),
            )
            row = self.valid_row(source)
            row["pass2_reviewer"] = ""
            self.assertEqual(verification_ok(row)[1], "pass2_requires_named_reviewer")
            row = self.valid_row(source)
            row["pass2_reviewer"] = "Reviewer A"
            self.assertEqual(
                verification_ok(row)[1], "independent_passes_require_two_reviewers"
            )

    def test_adjudication_resolves_a_completed_decision_disagreement_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video.mov"
            source.write_bytes(b"video")
            row = self.valid_row(source)
            row["review_pass_2"] = "rejected"
            row["pass2_class"] = ""
            self.assertEqual(verification_ok(row)[1], "decision_disagreement")
            row.update(
                {
                    "adjudication_reviewer": "Reviewer C",
                    "adjudication_timestamp": "2026-08-20T14:00:00Z",
                    "adjudication_decision": "accepted",
                    "adjudication_class": "KA",
                    "adjudication_start_time_s": "0",
                    "adjudication_end_time_s": "1",
                }
            )
            self.assertEqual(verification_ok(row), (True, ""))
            row["review_pass_2"] = ""
            self.assertEqual(verification_ok(row)[1], "pass2_incomplete")

    def test_agreement_projection_cannot_be_changed_after_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video.mov"
            source.write_bytes(b"video")
            row = self.valid_row(source)
            before = agreement_projection_sha256([row])
            row["adjudication_decision"] = "rejected"
            self.assertEqual(agreement_projection_sha256([row]), before)
            row["pass2_class"] = "GA"
            self.assertNotEqual(agreement_projection_sha256([row]), before)
            row["review_pass_2"] = ""
            self.assertIn("pass2_incomplete", validate_review_completion(row))

    def test_v2_refuses_an_unhashed_legacy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            np.save(cache / "sample.npy", np.zeros((16, 64), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "hashed .npz"):
                load_feature_clip(
                    cache,
                    "sample",
                    "image_xyz_angle",
                    expected_protocol_sha256="a" * 64,
                    expected_source_sha256="b" * 64,
                    expected_num_frames=16,
                )


class V2MetricTests(unittest.TestCase):
    def test_all_clips_are_hierarchically_weighted(self) -> None:
        rows = [
            {"signer": "S5", "canonical_label": "A"},
            {"signer": "S5", "canonical_label": "A"},
            {"signer": "S5", "canonical_label": "B"},
            {"signer": "S14", "canonical_label": "A"},
            {"signer": "S14", "canonical_label": "B"},
            {"signer": "S14", "canonical_label": "B"},
            {"signer": "S14", "canonical_label": "B"},
        ]
        correct = np.asarray([True, True, False, True, True, True, True])
        result = hierarchical_classification_metrics(rows, correct, ["A", "B"])
        self.assertAlmostEqual(result["per_signer"]["S5"]["macro_recall"], 0.5)
        self.assertAlmostEqual(result["per_signer"]["S14"]["macro_recall"], 1.0)
        self.assertAlmostEqual(result["primary_equal_signer_macro_recall"], 0.75)
        self.assertAlmostEqual(result["clip_pooled_top1"], 6 / 7)

    def test_palm_coordinates_are_rotation_scale_translation_invariant(self) -> None:
        rng = np.random.default_rng(7)
        points = rng.normal(size=(21, 3)).astype(np.float32)
        points[0] = 0
        points[9] = [0.1, 1.0, 0.2]
        points[5] = [0.8, 0.5, 0.1]
        points[17] = [-0.7, 0.4, -0.1]
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        transformed = (points @ q.T) * 2.7 + np.array([3.0, -2.0, 5.0])
        first = palm_canonical_world_landmarks(landmark_object(points))
        second = palm_canonical_world_landmarks(landmark_object(transformed))
        np.testing.assert_allclose(first, second, atol=2e-5)


class V2ProtocolIntegrityTests(unittest.TestCase):
    def test_frozen_artifact_hashes_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            annotation_dir = root / "annotations/v2"
            manifest_dir = root / "data/manifests"
            config_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True)
            verified = annotation_dir / "verified_segments_v2.0.csv"
            coverage = annotation_dir / "coverage_v2.0.csv"
            agreement = annotation_dir / "agreement_pre_adjudication.json"
            manifest = manifest_dir / "nsl23_manifest_all.csv"
            source_inventory = manifest_dir / "v2_source_inventory_v2.0.csv"
            classes = config_dir / "nsl23_classes.json"
            for path, text in (
                (verified, "segment_id\n"),
                (coverage, "signer\n"),
                (agreement, "{}\n"),
                (manifest, "source_path\n"),
                (source_inventory, "relative_path,source_sha256,bytes\n"),
                (classes, '{"consonants": [], "vowels": []}\n'),
            ):
                path.write_text(text, encoding="utf-8")
            repo = Path(__file__).resolve().parents[1]
            protocol = json.loads(
                (repo / "configs/v2_protocol.json").read_text(encoding="utf-8")
            )
            protocol.update(
                {
                    "frozen": True,
                    "assigned_roles": {"frozen_test": ["S5", "S14"]},
                    "confirmatory_49way": True,
                    "segments_sha256": sha256_file(verified),
                    "coverage_sha256": sha256_file(coverage),
                    "agreement_sha256": sha256_file(agreement),
                    "source_inventory_sha256": sha256_file(source_inventory),
                    "manifest_sha256": sha256_file(manifest),
                    "classes_sha256": sha256_file(classes),
                    "protocol_sha256": None,
                }
            )
            protocol["hierarchy_sha256"] = hierarchy_sha256(protocol)
            protocol["protocol_sha256"] = protocol_sha256(protocol)
            protocol_path = config_dir / "v2_protocol.json"
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
            loaded = load_and_validate_frozen_protocol(protocol_path, require_confirmatory=True)
            self.assertEqual(loaded["protocol_sha256"], protocol["protocol_sha256"])
            rows = [
                {
                    "v2_stage": "stage2",
                    "v2_protocol_sha256": protocol["protocol_sha256"],
                    "v2_segments_sha256": protocol["segments_sha256"],
                    "v2_coverage_sha256": protocol["coverage_sha256"],
                    "v2_agreement_sha256": protocol["agreement_sha256"],
                    "v2_source_inventory_sha256": protocol["source_inventory_sha256"],
                    "v2_source_manifest_sha256": protocol["manifest_sha256"],
                    "v2_classes_sha256": protocol["classes_sha256"],
                }
            ]
            validate_manifest_provenance(rows, protocol, "stage2")
            verified.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "segments_sha256 mismatch"):
                load_and_validate_frozen_protocol(protocol_path)

    def test_builder_separates_final_training_from_stage4(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        classes_source = repo / "configs/nsl23_classes.json"
        labels_config = json.loads(classes_source.read_text(encoding="utf-8"))
        labels = labels_config["consonants"] + labels_config["vowels"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            verified = root / "verified.csv"
            coverage = root / "coverage.csv"
            agreement = root / "agreement.json"
            source_inventory = root / "source_inventory.csv"
            classes = root / "classes.json"
            protocol_path = root / "protocol.json"
            classes.write_text(classes_source.read_text(encoding="utf-8"), encoding="utf-8")
            coverage.write_text("signer,total_classes\nS5,49\nS14,49\n", encoding="utf-8")
            agreement.write_text("{}\n", encoding="utf-8")

            manifest_fields = [
                "source_path",
                "relative_path",
                "source_zip",
                "signer",
                "sign_type",
                "canonical_label",
                "usable_as_isolated_clip",
                "duration_seconds",
                "background_condition",
                "scene_type",
                "realworld_flag",
                "lighting",
                "indoor_outdoor",
            ]
            manifest_rows = []
            for signer in ("S1", "S2"):
                for label in labels:
                    manifest_rows.append(
                        {
                            "source_path": f"/{signer}/{label}.mov",
                            "relative_path": f"{signer}/{label}.mov",
                            "source_zip": "fixture.zip",
                            "signer": signer,
                            "sign_type": "vowel" if label in labels_config["vowels"] else "consonant",
                            "canonical_label": label,
                            "usable_as_isolated_clip": "true",
                            "duration_seconds": "1",
                            "background_condition": "prepared",
                            "scene_type": "fixture",
                            "realworld_flag": "false",
                            "lighting": "bright",
                            "indoor_outdoor": "indoor",
                        }
                    )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=manifest_fields)
                writer.writeheader()
                writer.writerows(manifest_rows)

            inventory_rows = [
                {
                    "relative_path": row["relative_path"],
                    "source_sha256": "2" * 64,
                    "bytes": "1",
                }
                for row in manifest_rows
            ]
            inventory_rows.extend(
                {
                    "relative_path": f"{signer}/aggregate.mov",
                    "source_sha256": "1" * 64,
                    "bytes": "1",
                }
                for signer in ("S3", "S5", "S14")
            )
            with source_inventory.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["relative_path", "source_sha256", "bytes"]
                )
                writer.writeheader()
                writer.writerows(inventory_rows)

            verified_rows = []
            for signer in ("S3", "S5", "S14"):
                for index, label in enumerate(labels):
                    row = {field: "" for field in SEGMENT_FIELDS}
                    row.update(
                        {
                            "segment_id": f"{signer}_{index}",
                            "source_video": f"{signer}/aggregate.mov",
                            "source_video_hash": "1" * 64,
                            "source_path": f"/{signer}/aggregate.mov",
                            "signer": signer,
                            "class": label,
                            "sign_type": "vowel" if label in labels_config["vowels"] else "consonant",
                            "start_frame": str(index * 31),
                            "end_frame": str(index * 31 + 30),
                            "start_time_s": str(index * 1.1),
                            "end_time_s": str(index * 1.1 + 1),
                            "fps": "30",
                            "condition": "prepared",
                            "review_pass_1": "accepted",
                            "pass1_reviewer": "A",
                            "pass1_timestamp": "2026-08-20T12:00:00Z",
                            "pass1_class": label,
                            "pass1_start_time_s": str(index * 1.1),
                            "pass1_end_time_s": str(index * 1.1 + 1),
                            "review_pass_2": "accepted",
                            "pass2_reviewer": "B",
                            "pass2_timestamp": "2026-08-20T13:00:00Z",
                            "pass2_class": label,
                            "pass2_start_time_s": str(index * 1.1),
                            "pass2_end_time_s": str(index * 1.1 + 1),
                            "include_in_v2": "true",
                        }
                    )
                    verified_rows.append(row)
            with verified.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=SEGMENT_FIELDS)
                writer.writeheader()
                writer.writerows(verified_rows)

            protocol = json.loads((repo / "configs/v2_protocol.json").read_text(encoding="utf-8"))
            protocol.update(
                {
                    "frozen": True,
                    "confirmatory_49way": True,
                    "segments_sha256": sha256_file(verified),
                    "coverage_sha256": sha256_file(coverage),
                    "agreement_sha256": sha256_file(agreement),
                    "source_inventory_sha256": sha256_file(source_inventory),
                    "manifest_sha256": sha256_file(manifest),
                    "classes_sha256": sha256_file(classes),
                    "protocol_sha256": None,
                }
            )
            protocol["assigned_roles"] = {
                "frozen_test": ["S5", "S14"],
                "validation": ["S3"],
                "development": ["S1", "S2"],
                "held_as_fallback": ["S4", "S6"],
                "development_augmentation": [],
                "incomplete": [],
            }
            protocol["protocol_sha256"] = protocol_sha256(protocol)
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

            def run_stage(stage: str, extra: list[str]) -> list[dict[str, str]]:
                output = root / f"{stage}.csv"
                summary = root / f"{stage}.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(repo / "scripts/build_v2_samples.py"),
                        "--protocol",
                        str(protocol_path),
                        "--manifest",
                        str(manifest),
                        "--verified",
                        str(verified),
                        "--coverage",
                        str(coverage),
                        "--agreement",
                        str(agreement),
                        "--source-inventory",
                        str(source_inventory),
                        "--classes",
                        str(classes),
                        "--stage",
                        stage,
                        "--output",
                        str(output),
                        "--summary",
                        str(summary),
                        *extra,
                    ],
                    check=True,
                    cwd=repo,
                    stdout=subprocess.DEVNULL,
                )
                with output.open(newline="", encoding="utf-8") as handle:
                    return list(csv.DictReader(handle))

            final_rows = run_stage("final_retrain", [])
            self.assertEqual({row["split"] for row in final_rows}, {"train"})
            self.assertFalse({row["signer"] for row in final_rows} & {"S5", "S14", "S4", "S6"})
            stage4_rows = run_stage("stage4_eval", [])
            self.assertEqual({row["split"] for row in stage4_rows}, {"test"})
            self.assertEqual({row["signer"] for row in stage4_rows}, {"S5", "S14"})
            stage3_rows = run_stage("stage3", ["--heldout-signer", "S3"])
            self.assertEqual({row["signer"] for row in stage3_rows if row["split"] == "test"}, {"S3"})


if __name__ == "__main__":
    unittest.main()

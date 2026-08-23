from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nsl23.freeze import canonical_snapshot_sha256, load_snapshot, verify_snapshot


class V1FreezeTests(unittest.TestCase):
    def test_snapshot_envelope_detects_tampering(self) -> None:
        payload = {
            "schema_version": 1,
            "inputs": [{"path": "x.txt", "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}],
        }
        payload["snapshot_sha256"] = canonical_snapshot_sha256(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_snapshot(path)["snapshot_sha256"], payload["snapshot_sha256"])
            payload["inputs"][0]["bytes"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "envelope hash mismatch"):
                load_snapshot(path)

    def test_repository_freeze_matches_bound_historical_tree(self) -> None:
        root = Path(__file__).resolve().parents[1]
        snapshot = load_snapshot(root / "artifacts/audit/protocol_freeze.json")
        source = json.loads(
            (root / "artifacts/audit/protocol_freeze_source.json").read_text(encoding="utf-8")
        )
        report = verify_snapshot(
            root,
            snapshot,
            mode="git",
            revision=source["git_revision"],
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["verified_inputs"], 15)

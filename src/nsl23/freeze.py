"""Verification helpers for the immutable Version 1 protocol snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_snapshot_sha256(snapshot: dict) -> str:
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(canonical)


def load_snapshot(path: Path) -> dict:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    expected = snapshot.get("snapshot_sha256")
    observed = canonical_snapshot_sha256(snapshot)
    if not expected or expected != observed:
        raise ValueError(
            f"Snapshot envelope hash mismatch: expected {expected!r}, observed {observed}"
        )
    return snapshot


def git_blob(root: Path, revision: str, relative: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def resolve_revision(root: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Cannot resolve Git revision {revision!r}")
    return result.stdout.strip()


def verify_snapshot(
    root: Path,
    snapshot: dict,
    *,
    mode: str,
    revision: str | None = None,
) -> dict:
    if mode not in {"git", "worktree"}:
        raise ValueError("mode must be 'git' or 'worktree'")
    resolved_revision = resolve_revision(root, revision or "HEAD") if mode == "git" else None
    mismatches: list[dict[str, object]] = []
    for record in snapshot["inputs"]:
        relative = record["path"]
        if mode == "git":
            data = git_blob(root, resolved_revision or "", relative)
            if data is None:
                mismatches.append({"path": relative, "reason": "missing_from_git_tree"})
                continue
        else:
            path = root / relative
            if not path.is_file():
                mismatches.append({"path": relative, "reason": "missing_from_worktree"})
                continue
            data = path.read_bytes()
        observed_size = len(data)
        observed_hash = sha256_bytes(data)
        if observed_size != record["bytes"] or observed_hash != record["sha256"]:
            mismatches.append(
                {
                    "path": relative,
                    "expected_bytes": record["bytes"],
                    "observed_bytes": observed_size,
                    "expected_sha256": record["sha256"],
                    "observed_sha256": observed_hash,
                }
            )
    return {
        "status": "pass" if not mismatches else "fail",
        "mode": mode,
        "revision": resolved_revision,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "verified_inputs": len(snapshot["inputs"]) - len(mismatches),
        "total_inputs": len(snapshot["inputs"]),
        "mismatches": mismatches,
    }

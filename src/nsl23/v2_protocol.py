"""Version 2 protocol integrity, manifest provenance, and model-spec helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .v2_annotation import hierarchy_sha256, sha256_file


def canonical_protocol(data: dict) -> str:
    payload = dict(data)
    payload["protocol_sha256"] = None
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def protocol_sha256(data: dict) -> str:
    return hashlib.sha256(canonical_protocol(data).encode("utf-8")).hexdigest()


def canonical_model_spec(data: dict) -> str:
    payload = dict(data)
    payload["model_spec_sha256"] = None
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def model_spec_sha256(data: dict) -> str:
    return hashlib.sha256(canonical_model_spec(data).encode("utf-8")).hexdigest()


def validate_hierarchy_lock(protocol: dict) -> None:
    expected = protocol.get("hierarchy_sha256")
    if not expected:
        raise ValueError("Missing hierarchy_sha256; the V2 design is not locked")
    actual = hierarchy_sha256(protocol)
    if expected != actual:
        raise ValueError(
            "hierarchy_sha256 does not match the locked fields. "
            "Do not continue after silent protocol edits."
        )


def validate_protocol_consistency(protocol: dict) -> None:
    training = protocol.get("model_training") or {}
    extraction = protocol.get("feature_extraction") or {}
    stage3 = protocol.get("stage3") or {}
    final = protocol.get("final_retrain") or {}
    estimand = protocol.get("estimand") or {}
    if int(training.get("epochs", -1)) != int(stage3.get("epochs", -2)):
        raise ValueError("Stage 3 epochs disagree with the locked model-training specification")
    if int(training.get("epochs", -1)) != int(final.get("epochs", -2)):
        raise ValueError("Final epochs disagree with the locked model-training specification")
    seeds = list(estimand.get("seeds") or [])
    if seeds != list(final.get("seeds") or []) or seeds != [17, 42, 73]:
        raise ValueError("V2 seeds must be the same locked batch 17, 42, 73")
    trials = [item.get("id") for item in stage3.get("trials") or []]
    if len(trials) != int(stage3.get("budget", -1)) or len(trials) != len(set(trials)):
        raise ValueError("Stage 3 trial registry does not match its budget")
    if int(extraction.get("frames_per_clip", -1)) <= 0:
        raise ValueError("Feature extraction must lock a positive frame count")
    if training.get("optimizer") != "AdamW":
        raise ValueError("V2 optimizer must remain AdamW")
    if protocol.get("test_assignment", {}).get("selection_uses_model_results") is not False:
        raise ValueError("Test assignment may not use model results")


def _default_artifact(protocol_path: Path, relative: str) -> Path:
    root = protocol_path.resolve().parents[1]
    return root / relative


def load_and_validate_frozen_protocol(
    protocol_path: Path,
    *,
    verified_path: Path | None = None,
    coverage_path: Path | None = None,
    agreement_path: Path | None = None,
    source_inventory_path: Path | None = None,
    manifest_path: Path | None = None,
    classes_path: Path | None = None,
    require_confirmatory: bool = False,
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_hierarchy_lock(protocol)
    validate_protocol_consistency(protocol)
    if not protocol.get("frozen"):
        raise ValueError("Version 2 protocol is not frozen. Do not train or evaluate.")

    expected_protocol_hash = protocol.get("protocol_sha256")
    if not expected_protocol_hash or expected_protocol_hash != protocol_sha256(protocol):
        raise ValueError("protocol_sha256 is missing or does not match configs/v2_protocol.json")

    artifacts = (
        (
            "segments_sha256",
            verified_path
            or _default_artifact(protocol_path, "annotations/v2/verified_segments_v2.0.csv"),
        ),
        (
            "coverage_sha256",
            coverage_path or _default_artifact(protocol_path, "annotations/v2/coverage_v2.0.csv"),
        ),
        (
            "agreement_sha256",
            agreement_path
            or _default_artifact(protocol_path, "annotations/v2/agreement_pre_adjudication.json"),
        ),
        (
            "source_inventory_sha256",
            source_inventory_path
            or _default_artifact(protocol_path, "data/manifests/v2_source_inventory_v2.0.csv"),
        ),
        (
            "manifest_sha256",
            manifest_path or _default_artifact(protocol_path, "data/manifests/nsl23_manifest_all.csv"),
        ),
        (
            "classes_sha256",
            classes_path or _default_artifact(protocol_path, "configs/nsl23_classes.json"),
        ),
    )
    for field, path in artifacts:
        expected = protocol.get(field)
        if not expected:
            raise ValueError(f"Frozen protocol is missing {field}")
        if not path.is_file():
            raise ValueError(f"Frozen artifact is missing: {path}")
        actual = sha256_file(path)
        if expected != actual:
            raise ValueError(f"{field} mismatch for {path}")

    frozen_test = protocol.get("assigned_roles", {}).get("frozen_test") or []
    if require_confirmatory:
        if protocol.get("confirmatory_49way") is not True or len(frozen_test) < 2:
            raise ValueError("No confirmatory Stage 4: fewer than two complete frozen testers")
    return protocol


def validate_manifest_provenance(rows: list[dict[str, str]], protocol: dict, stage: str) -> None:
    if not rows:
        raise ValueError("Sample manifest is empty")
    expected = {
        "v2_stage": stage,
        "v2_protocol_sha256": str(protocol["protocol_sha256"]),
        "v2_segments_sha256": str(protocol["segments_sha256"]),
        "v2_coverage_sha256": str(protocol["coverage_sha256"]),
        "v2_agreement_sha256": str(protocol["agreement_sha256"]),
        "v2_source_inventory_sha256": str(protocol["source_inventory_sha256"]),
        "v2_source_manifest_sha256": str(protocol["manifest_sha256"]),
        "v2_classes_sha256": str(protocol["classes_sha256"]),
    }
    for field, value in expected.items():
        observed = {str(row.get(field) or "") for row in rows}
        if observed != {value}:
            raise ValueError(f"V2 manifest provenance mismatch for {field}: {sorted(observed)}")


def load_and_validate_model_spec(path: Path, protocol: dict) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("frozen"):
        raise ValueError("V2 model specification is not frozen")
    if data.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise ValueError("Model specification belongs to a different V2 protocol")
    expected = data.get("model_spec_sha256")
    if not expected or expected != model_spec_sha256(data):
        raise ValueError("model_spec_sha256 is missing or invalid")
    training = protocol["model_training"]
    extraction = protocol["feature_extraction"]
    expected_fields = {
        "epochs": int(training["epochs"]),
        "episodes": int(training["episodes_per_epoch"]),
        "support": int(training["support_per_class"]),
        "query": int(training["query_per_class"]),
        "hidden_dim": int(training["hidden_dim"]),
        "layers": int(training["layers"]),
        "dropout": float(training["dropout"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "num_frames": int(extraction["frames_per_clip"]),
        "seeds": list(protocol["final_retrain"]["seeds"]),
    }
    observed = {field: data.get(field) for field in expected_fields}
    if observed != expected_fields:
        raise ValueError("Frozen model specification disagrees with the V2 protocol")
    return data

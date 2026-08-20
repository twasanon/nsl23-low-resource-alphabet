"""Load legacy and preregistered V2 landmark feature caches."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .geometry import RAW_ANGLE_FRAME_DIM, frame_dim as legacy_frame_dim, transform_clip
from .landmarks import LANDMARK_DIM


V2_FEATURE_MODES = (
    "image_xyz_angle",
    "world_xyz",
    "palm_canonical_world_xyz",
)


def cache_set_sha256(cache_dir: Path, sample_ids: list[str], *, require_npz: bool) -> str:
    """Hash an ordered sample-ID/file-hash inventory for an experiment input set."""
    entries: list[tuple[str, str]] = []
    for sample_id in sorted(set(sample_ids)):
        npz_path = cache_dir / f"{sample_id}.npz"
        npy_path = cache_dir / f"{sample_id}.npy"
        if npz_path.is_file():
            path = npz_path
        elif npy_path.is_file() and not require_npz:
            path = npy_path
        else:
            suffix = ".npz" if require_npz else ".npz or .npy"
            raise FileNotFoundError(f"Missing {suffix} cache for {sample_id}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append((sample_id, digest))
    payload = "\n".join(f"{sample_id},{digest}" for sample_id, digest in entries) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_dim(mode: str) -> int:
    if mode == "image_xyz_angle":
        return RAW_ANGLE_FRAME_DIM
    if mode in {"world_xyz", "palm_canonical_world_xyz"}:
        return LANDMARK_DIM
    return legacy_frame_dim(mode)


def load_feature_clip(
    cache_dir: Path,
    sample_id: str,
    mode: str,
    *,
    expected_protocol_sha256: str | None = None,
    expected_source_sha256: str | None = None,
    expected_num_frames: int | None = None,
) -> np.ndarray:
    npz_path = cache_dir / f"{sample_id}.npz"
    npy_path = cache_dir / f"{sample_id}.npy"
    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as packed:
            if expected_protocol_sha256 is not None:
                observed = str(packed["protocol_sha256"].item())
                if observed != expected_protocol_sha256:
                    raise ValueError(f"Cache protocol mismatch for {sample_id}")
            if expected_source_sha256:
                observed = str(packed["source_sha256"].item())
                if observed != expected_source_sha256:
                    raise ValueError(f"Cache source hash mismatch for {sample_id}")
            if expected_num_frames is not None:
                observed = int(packed["num_frames"].item())
                if observed != expected_num_frames:
                    raise ValueError(f"Cache frame-count mismatch for {sample_id}")
            if mode == "image_xyz_angle":
                array = transform_clip(packed["image_xyz"], "raw_angle")
            elif mode == "world_xyz":
                array = packed["world_xyz"].astype(np.float32, copy=True)
            elif mode == "palm_canonical_world_xyz":
                array = packed["palm_canonical_world_xyz"].astype(np.float32, copy=True)
            else:
                raise KeyError(f"Mode {mode!r} is not available in a V2 .npz cache")
    elif npy_path.is_file():
        if expected_protocol_sha256 is not None or expected_source_sha256 is not None:
            raise ValueError(
                f"V2 requires a hashed .npz cache for {sample_id}; legacy .npy is not permitted"
            )
        legacy_mode = "raw_angle" if mode == "image_xyz_angle" else mode
        array = transform_clip(np.load(npy_path, allow_pickle=False), legacy_mode)
    else:
        raise FileNotFoundError(f"No cache for {sample_id} in {cache_dir}")
    expected = feature_dim(mode)
    if array.ndim != 2 or array.shape[1] != expected:
        raise ValueError(f"Bad feature shape for {sample_id}: {array.shape}; expected (*, {expected})")
    if expected_num_frames is not None and array.shape[0] != expected_num_frames:
        raise ValueError(
            f"Bad feature length for {sample_id}: {array.shape[0]}; expected {expected_num_frames}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Feature cache contains NaN or infinity for {sample_id}")
    return array.astype(np.float32, copy=False)

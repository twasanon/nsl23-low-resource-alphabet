"""SO(3)-invariant inter-joint angles from MediaPipe 21-point hands."""

from __future__ import annotations

import numpy as np

from .landmarks import COORDS_PER_LANDMARK, LANDMARK_DIM, NUM_HAND_LANDMARKS

# 15 intra-finger flexion angles: (parent, pivot, child) along each ray from the wrist.
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
# 5 abduction / span angles at the wrist.
WRIST_TRIPLETS = (
    (1, 0, 5),
    (5, 0, 9),
    (9, 0, 13),
    (13, 0, 17),
    (1, 0, 17),
)
ANGLE_DIM = 20
ANGLE_FRAME_DIM = ANGLE_DIM + 1
RAW_ANGLE_FRAME_DIM = 63 + ANGLE_DIM + 1  # xyz without flag, angles, flag


def _joint_angle(points: np.ndarray, parent: int, pivot: int, child: int) -> float:
    u = points[parent] - points[pivot]
    v = points[child] - points[pivot]
    u_norm = float(np.linalg.norm(u))
    v_norm = float(np.linalg.norm(v))
    if u_norm < 1e-8 or v_norm < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(u, v) / (u_norm * v_norm), -1.0, 1.0))
    return float(np.arccos(cosine))


def angles_from_points(points: np.ndarray) -> np.ndarray:
    if points.shape != (NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK):
        raise ValueError(f"Expected (21, 3) points, got {points.shape}")
    values = []
    for chain in FINGER_CHAINS:
        for parent, pivot, child in zip(chain[:-2], chain[1:-1], chain[2:], strict=True):
            values.append(_joint_angle(points, parent, pivot, child))
    for parent, pivot, child in WRIST_TRIPLETS:
        values.append(_joint_angle(points, parent, pivot, child))
    return np.asarray(values, dtype=np.float32)


def points_from_frame_features(frame: np.ndarray) -> tuple[np.ndarray, float]:
    if frame.shape[-1] != LANDMARK_DIM:
        raise ValueError(f"Expected {LANDMARK_DIM}-D landmark frame, got {frame.shape}")
    points = frame[:63].reshape(NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK)
    flag = float(frame[-1])
    return points, flag


def transform_clip(clip: np.ndarray, mode: str) -> np.ndarray:
    if mode in {"xyz", "raw"}:
        return clip.astype(np.float32, copy=True)
    rows = []
    for frame in clip:
        points, flag = points_from_frame_features(frame)
        if flag < 0.5:
            angles = np.zeros(ANGLE_DIM, dtype=np.float32)
        else:
            angles = angles_from_points(points)
        flag_arr = np.array([flag], dtype=np.float32)
        if mode == "angle":
            rows.append(np.concatenate([angles, flag_arr]))
        elif mode in {"raw_angle", "xyz_angle"}:
            rows.append(np.concatenate([points.reshape(-1), angles, flag_arr]))
        else:
            raise KeyError(mode)
    return np.stack(rows, axis=0)


def frame_dim(mode: str) -> int:
    if mode in {"xyz", "raw"}:
        return LANDMARK_DIM
    if mode == "angle":
        return ANGLE_FRAME_DIM
    if mode in {"raw_angle", "xyz_angle"}:
        return RAW_ANGLE_FRAME_DIM
    raise KeyError(mode)

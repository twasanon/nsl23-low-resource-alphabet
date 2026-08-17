"""MediaPipe hand-landmark extraction and wrist-relative normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

NUM_HAND_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
LANDMARK_DIM = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK + 1  # xyz + detection flag
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9


def _select_hand(result: Any) -> tuple[Any, float] | None:
    if not result.multi_hand_landmarks:
        return None
    scored: list[tuple[Any, float]] = []
    handedness = result.multi_handedness or []
    for index, landmarks in enumerate(result.multi_hand_landmarks):
        score = 1.0
        if index < len(handedness):
            categories = getattr(handedness[index], "classification", None) or []
            if categories:
                score = float(categories[0].score)
        scored.append((landmarks, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[0]


def normalize_landmarks(hand_landmarks: Any) -> np.ndarray:
    coords = np.array(
        [[float(point.x), float(point.y), float(point.z)] for point in hand_landmarks.landmark],
        dtype=np.float32,
    )
    if coords.shape != (NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK):
        raise ValueError(f"Unexpected landmark shape {coords.shape}")
    relative = coords - coords[WRIST_INDEX]
    scale = float(np.linalg.norm(relative[MIDDLE_MCP_INDEX, :2]))
    if scale < 1e-6:
        scale = 1.0
    relative = relative / scale
    features = np.concatenate([relative.reshape(-1), np.array([1.0], dtype=np.float32)])
    return features.astype(np.float32)


def empty_frame_features() -> np.ndarray:
    return np.zeros(LANDMARK_DIM, dtype=np.float32)


def extract_clip_landmarks(frames: np.ndarray, detector: Any) -> tuple[np.ndarray, int]:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB clip (T,H,W,3); got {frames.shape}")
    rows = []
    detected = 0
    for frame in frames:
        rgb = np.ascontiguousarray(frame)
        result = detector.process(rgb)
        selected = _select_hand(result)
        if selected is None:
            rows.append(empty_frame_features())
            continue
        rows.append(normalize_landmarks(selected[0]))
        detected += 1
    return np.stack(rows, axis=0), detected


def make_detector(*, static_image_mode: bool, min_detection_confidence: float = 0.35) -> Any:
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.35,
    )


def cache_path(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / f"{sample_id}.npy"

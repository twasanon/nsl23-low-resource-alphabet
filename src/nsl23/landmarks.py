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
INDEX_MCP_INDEX = 5
PINKY_MCP_INDEX = 17


def _select_hand_index(result: Any) -> tuple[int, Any, float] | None:
    if not result.multi_hand_landmarks:
        return None
    scored: list[tuple[int, Any, float]] = []
    handedness = result.multi_handedness or []
    for index, landmarks in enumerate(result.multi_hand_landmarks):
        score = 1.0
        if index < len(handedness):
            categories = getattr(handedness[index], "classification", None) or []
            if categories:
                score = float(categories[0].score)
        scored.append((index, landmarks, score))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[0]


def _select_hand(result: Any) -> tuple[Any, float] | None:
    selected = _select_hand_index(result)
    if selected is None:
        return None
    _, landmarks, score = selected
    return landmarks, score


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


def _normalized_points(hand_landmarks: Any) -> np.ndarray:
    coords = np.array(
        [[float(point.x), float(point.y), float(point.z)] for point in hand_landmarks.landmark],
        dtype=np.float32,
    )
    if coords.shape != (NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK):
        raise ValueError(f"Unexpected landmark shape {coords.shape}")
    relative = coords - coords[WRIST_INDEX]
    scale = float(np.linalg.norm(relative[MIDDLE_MCP_INDEX]))
    if scale < 1e-6:
        scale = 1.0
    return (relative / scale).astype(np.float32)


def normalize_world_landmarks(hand_landmarks: Any) -> np.ndarray:
    points = _normalized_points(hand_landmarks)
    return np.concatenate([points.reshape(-1), np.array([1.0], dtype=np.float32)])


def palm_canonical_world_landmarks(hand_landmarks: Any) -> np.ndarray:
    """Rotate normalized world points into semantic palm axes.

    +y runs from wrist to middle MCP, +x from pinky MCP toward index MCP after
    orthogonalization, and +z completes a right-handed frame. The semantic
    index/pinky direction canonicalizes left/right hand orientation.
    """
    points = _normalized_points(hand_landmarks)
    y_axis = points[MIDDLE_MCP_INDEX]
    y_norm = float(np.linalg.norm(y_axis))
    x_hint = points[INDEX_MCP_INDEX] - points[PINKY_MCP_INDEX]
    if y_norm < 1e-6:
        return np.concatenate([points.reshape(-1), np.array([1.0], dtype=np.float32)])
    y_axis = y_axis / y_norm
    x_axis = x_hint - float(np.dot(x_hint, y_axis)) * y_axis
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-6:
        return np.concatenate([points.reshape(-1), np.array([1.0], dtype=np.float32)])
    x_axis = x_axis / x_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        return np.concatenate([points.reshape(-1), np.array([1.0], dtype=np.float32)])
    z_axis = z_axis / z_norm
    x_axis = np.cross(y_axis, z_axis)
    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    canonical = points @ basis
    return np.concatenate([canonical.reshape(-1), np.array([1.0], dtype=np.float32)]).astype(
        np.float32
    )


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


def extract_clip_landmark_views(
    frames: np.ndarray, detector: Any
) -> tuple[dict[str, np.ndarray], int]:
    """Extract the three preregistered V2 coordinate views for one clip."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB clip (T,H,W,3); got {frames.shape}")
    image_rows: list[np.ndarray] = []
    world_rows: list[np.ndarray] = []
    palm_rows: list[np.ndarray] = []
    detected = 0
    for frame in frames:
        result = detector.process(np.ascontiguousarray(frame))
        selected = _select_hand_index(result)
        if selected is None:
            image_rows.append(empty_frame_features())
            world_rows.append(empty_frame_features())
            palm_rows.append(empty_frame_features())
            continue
        index, image_landmarks, _ = selected
        image_rows.append(normalize_landmarks(image_landmarks))
        world_sets = result.multi_hand_world_landmarks or []
        if index >= len(world_sets):
            world_rows.append(empty_frame_features())
            palm_rows.append(empty_frame_features())
        else:
            world_rows.append(normalize_world_landmarks(world_sets[index]))
            palm_rows.append(palm_canonical_world_landmarks(world_sets[index]))
        detected += 1
    return (
        {
            "image_xyz": np.stack(image_rows),
            "world_xyz": np.stack(world_rows),
            "palm_canonical_world_xyz": np.stack(palm_rows),
        },
        detected,
    )


def make_detector(
    *,
    static_image_mode: bool,
    min_detection_confidence: float = 0.35,
    min_tracking_confidence: float = 0.35,
    max_num_hands: int = 2,
    model_complexity: int = 1,
) -> Any:
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def cache_path(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / f"{sample_id}.npy"

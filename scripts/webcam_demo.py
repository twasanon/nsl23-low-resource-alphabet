#!/usr/bin/env python3
"""Low-resource webcam or video-file NSL alphabet demo using MediaPipe landmarks."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from nsl23.geometry import transform_clip
from nsl23.landmarks import empty_frame_features, make_detector, normalize_landmarks
from nsl23.models import build_model


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["experiment_config"]
    labels = payload["labels"]
    model_name = payload["model_name"]
    if model_name == "proto_bilstm" and "proto_bilstm" not in config.get("models", {}):
        model_name = "landmark_bilstm"
    model = build_model(model_name, config, len(labels), int(config["num_frames"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    prototypes = payload.get("prototypes")
    if prototypes is not None:
        prototypes = prototypes.to(device)
    return model, labels, config, payload, prototypes


def features_from_bgr(frame_bgr: np.ndarray, detector) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    if not result.multi_hand_landmarks:
        return empty_frame_features()
    scored = []
    handedness = result.multi_handedness or []
    for index, landmarks in enumerate(result.multi_hand_landmarks):
        score = 1.0
        if index < len(handedness) and handedness[index].classification:
            score = float(handedness[index].classification[0].score)
        scored.append((landmarks, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return normalize_landmarks(scored[0][0])


def draw_overlay(frame: np.ndarray, text: str, fps: float) -> np.ndarray:
    vis = frame.copy()
    cv2.rectangle(vis, (8, 8), (min(vis.shape[1] - 8, 620), 78), (0, 0, 0), -1)
    cv2.putText(vis, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"{fps:.1f} FPS", (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1, cv2.LINE_AA)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoothing", type=float, default=0.4, help="EMA coefficient in [0,1); 0 disables")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, labels, config, payload, prototypes = load_checkpoint(args.checkpoint, device)
    num_frames = int(config["num_frames"])
    feature_mode = str(payload.get("feature_mode") or config.get("feature_mode", "xyz"))
    buffer: deque[np.ndarray] = deque(maxlen=num_frames)
    detector = make_detector(static_image_mode=False, min_detection_confidence=0.35)
    source = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not source.isOpened():
        raise RuntimeError("Could not open camera or video source")

    ema: np.ndarray | None = None
    last = time.perf_counter()
    fps = 0.0
    frame_index = 0
    predictions: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            frame_index += 1
            buffer.append(features_from_bgr(frame, detector))
            if len(buffer) == num_frames:
                clip_xyz = np.stack(buffer, axis=0)
                clip = torch.from_numpy(transform_clip(clip_xyz, feature_mode)).unsqueeze(0).to(device)
                with torch.no_grad():
                    if prototypes is not None:
                        features = model.forward_features(clip)
                        logits = -torch.cdist(features, prototypes)
                    else:
                        logits = model(clip)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                if args.smoothing > 0:
                    ema = probabilities if ema is None else args.smoothing * ema + (1.0 - args.smoothing) * probabilities
                    scores = ema
                else:
                    scores = probabilities
                index = int(scores.argmax())
                text = f"{labels[index]}  {100.0 * scores[index]:.1f}%"
                predictions.append(
                    {
                        "frame": frame_index,
                        "label": labels[index],
                        "confidence": float(scores[index]),
                    }
                )
            else:
                text = f"buffer {len(buffer)}/{num_frames}"
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            vis = draw_overlay(frame, text, fps)
            if args.no_window:
                if frame_index == 1 or frame_index % 15 == 0:
                    print(json.dumps({"frame": frame_index, "text": text, "fps": round(fps, 2)}), flush=True)
            else:
                cv2.imshow("NSL23 landmark demo", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        detector.close()
        source.release()
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    summary = {
        "frames": frame_index,
        "predictions": len(predictions),
        "elapsed_seconds": round(elapsed, 4),
        "fps": round(frame_index / max(elapsed, 1e-6), 2),
        "ema_fps": round(fps, 2),
        "source": str(args.video) if args.video else f"camera:{args.camera}",
        "checkpoint": str(args.checkpoint),
        "last_prediction": predictions[-1] if predictions else None,
    }
    print(json.dumps(summary), flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

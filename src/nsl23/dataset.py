"""Dataset and deterministic loader helpers for cached NSL23 clips."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .paths import resolve_repo_path


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_labels(path: Path) -> list[str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    labels = config["consonants"] + config["vowels"]
    if len(labels) != 49 or len(set(labels)) != 49:
        raise ValueError("Expected 49 unique classes")
    return labels


class CachedClipDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        labels: list[str],
        training: bool,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        self.rows = rows
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.training = training
        self.augmentation = augmentation or {}
        missing = sorted({row["canonical_label"] for row in rows} - set(labels))
        if missing:
            raise ValueError(f"Unknown labels in sample manifest: {missing}")
        for row in rows:
            if not row.get("frame_cache_path"):
                raise ValueError("Sample manifest does not contain frame_cache_path")

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, frames: torch.Tensor) -> torch.Tensor:
        if random.random() < float(self.augmentation.get("brightness_probability", 0.0)):
            limit = float(self.augmentation["brightness_delta"])
            frames = (frames + random.uniform(-limit, limit)).clamp(0.0, 1.0)
        if random.random() < float(self.augmentation.get("contrast_probability", 0.0)):
            low, high = self.augmentation["contrast_range"]
            factor = random.uniform(float(low), float(high))
            channel_mean = frames.mean(dim=(0, 2, 3), keepdim=True)
            frames = ((frames - channel_mean) * factor + channel_mean).clamp(0.0, 1.0)
        return frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        cache_path = resolve_repo_path(row["frame_cache_path"])
        array = np.load(cache_path, allow_pickle=False)
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
            raise RuntimeError(f"Invalid frame cache {cache_path}: {array.shape} {array.dtype}")
        frames = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        if self.training:
            frames = self._augment(frames)
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "frames": frames,
            "label": self.label_to_index[row["canonical_label"]],
            "sample_id": row["sample_id"],
        }


class HandCropDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        labels: list[str],
        training: bool,
        augmentation: dict[str, Any] | None = None,
        crop_dir: Path = Path("data/hand_crop_cache_primary"),
    ) -> None:
        self.rows = rows
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.training = training
        self.augmentation = augmentation or {}
        self.crop_dir = crop_dir
        missing = sorted({row["canonical_label"] for row in rows} - set(labels))
        if missing:
            raise ValueError(f"Unknown labels in sample manifest: {missing}")

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, frames: torch.Tensor) -> torch.Tensor:
        if random.random() < float(self.augmentation.get("brightness_probability", 0.0)):
            limit = float(self.augmentation["brightness_delta"])
            frames = (frames + random.uniform(-limit, limit)).clamp(0.0, 1.0)
        if random.random() < float(self.augmentation.get("contrast_probability", 0.0)):
            low, high = self.augmentation["contrast_range"]
            factor = random.uniform(float(low), float(high))
            channel_mean = frames.mean(dim=(0, 2, 3), keepdim=True)
            frames = ((frames - channel_mean) * factor + channel_mean).clamp(0.0, 1.0)
        return frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.crop_dir / f"{row['sample_id']}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, allow_pickle=False)
        frames = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        if self.training:
            frames = self._augment(frames)
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "frames": frames,
            "label": self.label_to_index[row["canonical_label"]],
            "sample_id": row["sample_id"],
        }


def make_crop_loaders(
    manifest: Path,
    labels_path: Path,
    experiment_config: dict[str, Any],
    seed: int,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader[dict[str, Any]]], list[str], dict[str, list[dict[str, str]]]]:
    rows = read_rows(manifest)
    labels = load_labels(labels_path)
    grouped = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    if not grouped["train"] or not grouped["test"]:
        raise ValueError({split: len(value) for split, value in grouped.items()})
    training = experiment_config["training"]
    augmentation = {
        key: training[key]
        for key in (
            "brightness_probability",
            "brightness_delta",
            "contrast_probability",
            "contrast_range",
        )
    }
    crop_dir = Path(experiment_config.get("crop_dir", "data/hand_crop_cache_primary"))
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            HandCropDataset(grouped["train"], labels, training=True, augmentation=augmentation, crop_dir=crop_dir),
            batch_size=int(training["batch_size"]),
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
            pin_memory=False,
        ),
        "test": DataLoader(
            HandCropDataset(grouped["test"], labels, training=False, crop_dir=crop_dir),
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }
    return loaders, labels, grouped


def resolve_cache_path(stored: str, sample_id: str, fallback_dir: Path) -> Path:
    path = Path(stored)
    if path.is_file():
        return path
    fallback = fallback_dir / f"{sample_id}.npy"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing cache for {sample_id}: tried {path} and {fallback}")


class LandmarkClipDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        labels: list[str],
        training: bool,
        noise_std: float = 0.0,
        feature_mode: str = "xyz",
    ) -> None:
        self.rows = rows
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.training = training
        self.noise_std = noise_std
        self.feature_mode = feature_mode
        missing = sorted({row["canonical_label"] for row in rows} - set(labels))
        if missing:
            raise ValueError(f"Unknown labels in sample manifest: {missing}")
        for row in rows:
            if not row.get("landmark_cache_path"):
                raise ValueError("Sample manifest does not contain landmark_cache_path")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from .geometry import transform_clip

        row = self.rows[index]
        array = np.load(
            resolve_cache_path(row["landmark_cache_path"], row["sample_id"], Path("data/landmark_cache_primary")),
            allow_pickle=False,
        )
        if array.ndim != 2:
            raise RuntimeError(f"Invalid landmark cache {row['landmark_cache_path']}: {array.shape}")
        features = torch.from_numpy(transform_clip(array, self.feature_mode))
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(features)
            noise[:, -1] = 0.0
            features = features + noise * self.noise_std
            detected = features[:, -1] > 0.5
            features[:, -1] = detected.float()
        return {
            "frames": features,
            "label": self.label_to_index[row["canonical_label"]],
            "sample_id": row["sample_id"],
            "true_label": row["canonical_label"],
            "signer": row["signer"],
            "lighting": row.get("lighting", ""),
            "sign_type": row.get("sign_type", ""),
        }


def make_loaders(
    manifest: Path,
    labels_path: Path,
    experiment_config: dict[str, Any],
    seed: int,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader[dict[str, Any]]], list[str], dict[str, list[dict[str, str]]]]:
    rows = read_rows(manifest)
    labels = load_labels(labels_path)
    grouped = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    if not grouped["train"] or not grouped["test"]:
        raise ValueError({split: len(value) for split, value in grouped.items()})
    training = experiment_config["training"]
    augmentation = {
        key: training[key]
        for key in (
            "brightness_probability",
            "brightness_delta",
            "contrast_probability",
            "contrast_range",
        )
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            CachedClipDataset(grouped["train"], labels, training=True, augmentation=augmentation),
            batch_size=int(training["batch_size"]),
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
            pin_memory=False,
        ),
        "test": DataLoader(
            CachedClipDataset(grouped["test"], labels, training=False),
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }
    if grouped["validation"]:
        loaders["validation"] = DataLoader(
            CachedClipDataset(grouped["validation"], labels, training=False),
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders, labels, grouped


def make_landmark_loaders(
    manifest: Path,
    labels_path: Path,
    experiment_config: dict[str, Any],
    seed: int,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader[dict[str, Any]]], list[str], dict[str, list[dict[str, str]]]]:
    rows = read_rows(manifest)
    labels = load_labels(labels_path)
    grouped = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    if not grouped["train"] or not grouped["test"]:
        raise ValueError({split: len(value) for split, value in grouped.items()})
    training = experiment_config["training"]
    noise_std = float(training.get("landmark_noise_std", 0.0))
    feature_mode = str(experiment_config.get("feature_mode", "xyz"))
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            LandmarkClipDataset(
                grouped["train"], labels, training=True, noise_std=noise_std, feature_mode=feature_mode
            ),
            batch_size=int(training["batch_size"]),
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
            pin_memory=False,
        ),
        "test": DataLoader(
            LandmarkClipDataset(grouped["test"], labels, training=False, feature_mode=feature_mode),
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }
    if grouped["validation"]:
        loaders["validation"] = DataLoader(
            LandmarkClipDataset(grouped["validation"], labels, training=False, feature_mode=feature_mode),
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders, labels, grouped

"""Fair MobileNetV2-based spatial and temporal baselines for NSL23."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class FrozenMobileNetV2(nn.Module):
    """ImageNet-pretrained frame encoder with frozen weights and batch norms."""

    output_dim = 1280

    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for parameter in self.parameters():
            parameter.requires_grad = False
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenMobileNetV2":
        # The frozen encoder must not update running batch-normalization state.
        super().train(False)
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.features(frames)
            return self.pool(features).flatten(1)


class SingleFrameCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = FrozenMobileNetV2()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        middle = clips[:, clips.shape[1] // 2]
        features = self.encoder(middle)
        return self.head(self.dropout(features))


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        projection_dim: int = 256,
        hidden_dim: int = 128,
        layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = FrozenMobileNetV2()
        self.projection = nn.Linear(self.encoder.output_dim, projection_dim)
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = clips.shape
        features = self.encoder(clips.reshape(batch * time, channels, height, width))
        projected = self.projection(features).reshape(batch, time, -1)
        sequence, _ = self.lstm(projected)
        pooled = self.norm(sequence).mean(dim=1)
        return self.head(self.dropout(pooled))


class CNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int = 16,
        projection_dim: int = 256,
        heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 512,
        encoder_dropout: float = 0.0,
        head_dropout: float = 0.2,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        if projection_dim % heads:
            raise ValueError("projection_dim must be divisible by heads")
        self.encoder = FrozenMobileNetV2()
        self.projection = nn.Linear(self.encoder.output_dim, projection_dim)
        self.position = nn.Parameter(torch.zeros(num_frames, projection_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=encoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(projection_dim, num_classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = clips.shape
        if time > self.position.shape[0]:
            raise ValueError(f"Received {time} frames but positional table has {self.position.shape[0]}")
        features = self.encoder(clips.reshape(batch * time, channels, height, width))
        sequence = self.projection(features).reshape(batch, time, -1)
        sequence = sequence + self.position[:time].unsqueeze(0)
        encoded = self.norm(self.transformer(sequence))
        pooled = encoded.mean(dim=1)
        return self.head(self.dropout(pooled))


def build_model(name: str, config: dict[str, Any], num_classes: int, num_frames: int) -> nn.Module:
    model_config = config["models"][name]
    if name == "single_frame_cnn":
        return SingleFrameCNN(num_classes=num_classes, dropout=model_config["dropout"])
    if name == "cnn_bilstm":
        return CNNBiLSTM(
            num_classes=num_classes,
            projection_dim=model_config["projection_dim"],
            hidden_dim=model_config["hidden_dim"],
            layers=model_config["layers"],
            dropout=model_config["dropout"],
        )
    if name == "cnn_transformer":
        return CNNTransformer(
            num_classes=num_classes,
            num_frames=num_frames,
            projection_dim=model_config["projection_dim"],
            heads=model_config["heads"],
            layers=model_config["layers"],
            feedforward_dim=model_config["feedforward_dim"],
            encoder_dropout=model_config["encoder_dropout"],
            head_dropout=model_config["head_dropout"],
            norm_first=model_config["norm_first"],
        )
    raise KeyError(name)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }

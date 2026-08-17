#!/usr/bin/env python3
"""Generate deterministic, publication-ready figures from audited NSL23 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


MODEL_ORDER = [
    "single_frame_cnn",
    "cnn_bilstm",
    "cnn_transformer",
    "landmark_mlp",
    "landmark_bilstm",
]
MODEL_LABELS = {
    "single_frame_cnn": "Single-frame CNN",
    "cnn_bilstm": "CNN–BiLSTM",
    "cnn_transformer": "CNN–Transformer",
    "landmark_mlp": "Landmark MLP",
    "landmark_bilstm": "Landmark BiLSTM",
}
COLORS = {
    "single_frame_cnn": "#0072B2",
    "cnn_bilstm": "#D55E00",
    "cnn_transformer": "#009E73",
    "landmark_mlp": "#CC79A7",
    "landmark_bilstm": "#000000",
    "bright": "#E69F00",
    "dark": "#6A5ACD",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    sns.set_style("whitegrid", {"grid.color": "#E6E6E6", "grid.linewidth": 0.6})


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def box(ax: plt.Axes, x: float, y: float, width: float, height: float, text: str, color: str, *, fontsize: float = 9) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.10),
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color="#444444"))


def study_flow(audit_dir: Path, output_dir: Path) -> None:
    manifest = json.loads((audit_dir / "nsl23_manifest_summary.json").read_text(encoding="utf-8"))
    folds = json.loads((audit_dir / "isolated_twofold_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.04, 0.68, 0.23, 0.17, f"NSL23 release\n{manifest['videos']} video files", "#0072B2", fontsize=10)
    box(ax, 0.38, 0.68, 0.24, 0.17, f"Isolated videos\n{manifest['isolated_videos']} files", "#56B4E9", fontsize=10)
    box(ax, 0.73, 0.68, 0.23, 0.17, f"Primary set\n{manifest['primary_usable_isolated_clips']} original clips\n49 classes · 7 signers", "#009E73", fontsize=9.5)
    arrow(ax, (0.27, 0.765), (0.38, 0.765))
    arrow(ax, (0.62, 0.765), (0.73, 0.765))
    ax.text(0.325, 0.84, "exclude 23 aggregate\n(no timestamps)", ha="center", va="bottom", fontsize=8, color="#555555")
    ax.text(0.675, 0.84, "exclude 232 isolated\ncropped/preprocessed files", ha="center", va="bottom", fontsize=8, color="#555555")

    box(ax, 0.08, 0.30, 0.34, 0.18, "Fold 1\ntrain: 277 clips (6 signers)\ntest: S1, 98 clips", "#E69F00")
    box(ax, 0.58, 0.30, 0.34, 0.18, "Fold 2\ntrain: 277 clips (6 signers)\ntest: S2, 98 clips", "#CC79A7")
    arrow(ax, (0.81, 0.68), (0.25, 0.48))
    arrow(ax, (0.84, 0.68), (0.75, 0.48))
    ax.text(0.50, 0.205, "5 models × 3 fixed seeds per fold = 30 runs", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(0.50, 0.115, "Each held-out signer contributes all 49 classes in both bright and dark indoor conditions.", ha="center", va="center", fontsize=8.5)
    ax.text(0.50, 0.045, "Aggregate and cropped/preprocessed categories overlap by two files; exclusions are shown sequentially.", ha="center", va="center", fontsize=7.5, color="#555555")
    save_figure(fig, output_dir, "figure_1_study_flow")


def architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "RGB video models (control)", ha="left", va="center", fontsize=10, weight="bold")
    box(ax, 0.02, 0.72, 0.16, 0.16, "16 RGB frames\n224 × 224", "#0072B2", fontsize=8)
    box(ax, 0.22, 0.72, 0.20, 0.16, "Frozen MobileNetV2\n1,280-D / frame", "#56B4E9", fontsize=8)
    arrow(ax, (0.18, 0.80), (0.22, 0.80))
    box(ax, 0.46, 0.84, 0.30, 0.10, "Middle frame → linear", COLORS["single_frame_cnn"], fontsize=8)
    box(ax, 0.46, 0.71, 0.30, 0.10, "Projection → BiLSTM → mean", COLORS["cnn_bilstm"], fontsize=8)
    box(ax, 0.46, 0.58, 0.30, 0.10, "Projection + positions → Transformer", COLORS["cnn_transformer"], fontsize=8)
    arrow(ax, (0.42, 0.80), (0.46, 0.89))
    arrow(ax, (0.42, 0.80), (0.46, 0.76))
    arrow(ax, (0.42, 0.80), (0.46, 0.63))
    box(ax, 0.80, 0.71, 0.17, 0.16, "49-way\nsoftmax", "#333333", fontsize=8)
    arrow(ax, (0.76, 0.89), (0.80, 0.83))
    arrow(ax, (0.76, 0.76), (0.80, 0.79))
    arrow(ax, (0.76, 0.63), (0.80, 0.75))

    ax.text(0.02, 0.50, "Hand-landmark models (webcam path)", ha="left", va="center", fontsize=10, weight="bold")
    box(ax, 0.02, 0.22, 0.16, 0.16, "Same 16 frames\nuint8 RGB", "#0072B2", fontsize=8)
    box(ax, 0.22, 0.22, 0.22, 0.16, "MediaPipe Hands\n21 × xyz + flag\n64-D / frame", "#CC79A7", fontsize=8)
    arrow(ax, (0.18, 0.30), (0.22, 0.30))
    box(ax, 0.48, 0.32, 0.28, 0.12, "Flatten 1,024-D → MLP", COLORS["landmark_mlp"], fontsize=8)
    box(ax, 0.48, 0.16, 0.28, 0.12, "64-D sequence → BiLSTM → mean", COLORS["landmark_bilstm"], fontsize=8)
    arrow(ax, (0.44, 0.30), (0.48, 0.38))
    arrow(ax, (0.44, 0.30), (0.48, 0.22))
    box(ax, 0.80, 0.22, 0.17, 0.16, "49-way\nsoftmax", "#333333", fontsize=8)
    arrow(ax, (0.76, 0.38), (0.80, 0.34))
    arrow(ax, (0.76, 0.22), (0.80, 0.28))
    ax.text(
        0.50,
        0.04,
        "Wrist-relative xy scaling uses the middle-MCP distance. Landmark coordinates are the only input to the webcam demo.",
        ha="center",
        va="center",
        fontsize=8,
    )
    save_figure(fig, output_dir, "figure_2_architectures")


def training_curves(results_dir: Path, output_dir: Path) -> None:
    data = pd.read_csv(results_dir / "training_histories_all.csv")
    metrics = [("train_loss", "Training loss"), ("train_accuracy", "Training accuracy"), ("train_macro_f1", "Training macro-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65))
    for ax, (column, title) in zip(axes, metrics, strict=True):
        for model in MODEL_ORDER:
            model_rows = data[data["model"] == model]
            grouped = model_rows.groupby("epoch")[column]
            mean = grouped.mean()
            sd = grouped.std(ddof=1)
            epochs = mean.index.to_numpy()
            values = mean.to_numpy()
            spread = sd.fillna(0).to_numpy()
            ax.plot(epochs, values, label=MODEL_LABELS[model], color=COLORS[model], linewidth=1.7)
            ax.fill_between(epochs, values - spread, values + spread, color=COLORS[model], alpha=0.13, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 30)
        if column != "train_loss":
            ax.set_ylim(0, 1)
    axes[0].set_ylabel("Mean across folds and seeds")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output_dir, "figure_4_training_curves")


def primary_performance(results_dir: Path, output_dir: Path) -> None:
    pooled = pd.read_csv(results_dir / "pooled_out_of_fold_by_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    rng = np.random.default_rng(20260815)
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Top-1 accuracy", "Macro-F1"], strict=True):
        for index, model in enumerate(MODEL_ORDER):
            values = pooled.loc[pooled["model"] == model, metric].to_numpy(dtype=float)
            mean = values.mean()
            sd = values.std(ddof=1)
            ax.bar(index, mean, color=mpl.colors.to_rgba(COLORS[model], 0.72), edgecolor=COLORS[model], width=0.62)
            ax.errorbar(index, mean, yerr=sd, color="#222222", capsize=3, linewidth=1, zorder=4)
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(index + jitter, values, color="#111111", s=18, zorder=5, linewidth=0.3, edgecolor="white")
            ax.text(index, mean + sd + 0.012, f"{mean:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[name].replace(" ", "\n") for name in MODEL_ORDER], fontsize=7)
        ax.set_ylim(0, 0.40)
        ax.set_title(title)
        ax.set_ylabel("Score" if ax is axes[0] else "")
    fig.text(0.5, 0.01, "Bars: mean over three seeds after pooling the two held-out folds (196 clips/seed); error bars: ±1 SD; points: seeds.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output_dir, "figure_3_primary_performance")


def lighting_performance(results_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "condition_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    metrics = [("accuracy_mean", "accuracy_sd", "Top-1 accuracy"), ("macro_f1_mean", "macro_f1_sd", "Macro-F1")]
    for ax, (mean_col, sd_col, title) in zip(axes, metrics, strict=True):
        width = 0.34
        x = np.arange(len(MODEL_ORDER))
        for offset_index, lighting in enumerate(["bright", "dark"]):
            values = []
            errors = []
            for model in MODEL_ORDER:
                row = summary[(summary["model"] == model) & (summary["lighting"].str.lower() == lighting)]
                if len(row) != 1:
                    raise ValueError(f"Expected one condition summary for {model}/{lighting}, found {len(row)}")
                values.append(float(row.iloc[0][mean_col]))
                errors.append(float(row.iloc[0][sd_col]))
            positions = x + (offset_index - 0.5) * width
            ax.bar(positions, values, width=width, label=lighting.capitalize(), color=COLORS[lighting], alpha=0.78)
            ax.errorbar(positions, values, yerr=errors, fmt="none", color="#222222", capsize=2.5, linewidth=0.9)
        ax.set_xticks(x, [MODEL_LABELS[name].replace(" ", "\n") for name in MODEL_ORDER], fontsize=7)
        ax.set_ylim(0, 0.40)
        ax.set_title(title)
        ax.set_ylabel("Score" if ax is axes[0] else "")
    axes[1].legend(frameon=False, loc="upper right")
    fig.text(0.5, 0.01, "Mean ± SD over three seeds; each condition contains 98 out-of-fold clips (49 classes × two signers) per seed.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output_dir, "figure_5_lighting_performance")


def confusion_heatmap(results_dir: Path, output_dir: Path, model: str, stem: str, title: str) -> None:
    matrix = pd.read_csv(
        results_dir / f"confusion_{model}_mean_normalized.csv",
        index_col=0,
        keep_default_na=False,
    )
    fig, ax = plt.subplots(figsize=(8.1, 7.2))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="mako",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Mean row-normalized proportion", "shrink": 0.75},
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90, labelsize=5.5)
    ax.tick_params(axis="y", labelrotation=0, labelsize=5.5)
    save_figure(fig, output_dir, stem)


def per_class_recall(results_dir: Path, output_dir: Path, model: str, stem: str, title: str, color: str) -> None:
    data = pd.read_csv(results_dir / "per_class_summary.csv", keep_default_na=False)
    data = data[data["model"] == model].sort_values("class_index")
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6), gridspec_kw={"height_ratios": [36, 13]})
    for ax, sign_type in zip(axes, ["consonant", "vowel"], strict=True):
        subset = data[data["sign_type"] == sign_type]
        x = np.arange(len(subset))
        ax.bar(x, subset["recall_mean"], color=color, alpha=0.78, width=0.78)
        ax.errorbar(x, subset["recall_mean"], yerr=subset["recall_sd"], fmt="none", ecolor="#333333", capsize=1.5, linewidth=0.7)
        ax.set_xticks(x, subset["label"], rotation=90)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall")
        ax.set_title(sign_type.capitalize())
    axes[-1].set_xlabel("NSL23 class")
    fig.suptitle(title, y=1.01, fontsize=10)
    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required result files are missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=Path("artifacts/audit"))
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--design-only", action="store_true")
    args = parser.parse_args()
    configure_style()
    study_flow(args.audit_dir, args.output_dir)
    architecture(args.output_dir)
    if args.design_only:
        print(json.dumps({"status": "pass", "figures": 2, "output_dir": str(args.output_dir)}))
        return
    required = [
        args.results_dir / "training_histories_all.csv",
        args.results_dir / "pooled_out_of_fold_by_seed.csv",
        args.results_dir / "condition_summary.csv",
        args.results_dir / "confusion_cnn_transformer_mean_normalized.csv",
        args.results_dir / "confusion_landmark_bilstm_mean_normalized.csv",
        args.results_dir / "per_class_summary.csv",
    ]
    require_files(required)
    training_curves(args.results_dir, args.output_dir)
    primary_performance(args.results_dir, args.output_dir)
    lighting_performance(args.results_dir, args.output_dir)
    confusion_heatmap(
        args.results_dir,
        args.output_dir,
        "landmark_bilstm",
        "figure_6_landmark_bilstm_confusion",
        "Landmark BiLSTM confusion matrix averaged over three seeds and two held-out folds",
    )
    per_class_recall(
        args.results_dir,
        args.output_dir,
        "landmark_bilstm",
        "figure_7_landmark_bilstm_per_class_recall",
        "Landmark BiLSTM per-class recall (mean ± SD over three seeds)",
        COLORS["landmark_bilstm"],
    )
    confusion_heatmap(
        args.results_dir,
        args.output_dir,
        "cnn_transformer",
        "figure_8_transformer_confusion",
        "CNN–Transformer confusion matrix averaged over three seeds and two held-out folds",
    )
    print(json.dumps({"status": "pass", "figures": 8, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()

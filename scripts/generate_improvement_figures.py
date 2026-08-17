#!/usr/bin/env python3
"""Figures and LaTeX tables for the NSL23 improvement matrix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from generate_publication_figures import COLORS, configure_style, save_figure


LABELS = {
    "single_frame_cnn": "RGB CNN",
    "cnn_bilstm": "RGB BiLSTM",
    "cnn_transformer": "RGB Transformer",
    "xyz_bilstm": "xyz BiLSTM (CE)",
    "raw_angle_bilstm": "xyz+angle BiLSTM (CE)",
    "crop_cnn": "Hand-crop CNN",
    "proto_xyz": "xyz Proto-BiLSTM",
    "proto_raw_angle": "xyz+angle Proto-BiLSTM",
}
BAR_COLORS = {
    "single_frame_cnn": "#0072B2",
    "cnn_bilstm": "#D55E00",
    "cnn_transformer": "#009E73",
    "xyz_bilstm": "#888888",
    "raw_angle_bilstm": "#CC79A7",
    "crop_cnn": "#E69F00",
    "proto_xyz": "#56B4E9",
    "proto_raw_angle": "#000000",
}


def seed_pool(pairs: list[tuple[int, float]]) -> tuple[float, float, list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for seed, value in pairs:
        grouped[seed].append(value)
    values = [float(np.mean(grouped[seed])) for seed in sorted(grouped) if len(grouped[seed]) == 2]
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)), values


def pm(mean: float, sd: float) -> str:
    return f"{100.0 * mean:.1f} $\\pm$ {100.0 * sd:.1f}"


def collect_model(run_dir: Path, model: str) -> dict[str, tuple[float, float, list[float]]]:
    acc, f1, cons, vow = [], [], [], []
    for path in sorted(run_dir.glob(f"*/{model}/*/metrics.json")):
        payload = json.loads(path.read_text())
        seed = int(path.parts[-2].split("_")[1])
        pred = path.parent / "test_predictions.csv"
        accuracy = float(payload["test_accuracy"])
        cons_v = payload.get("consonant_accuracy") or payload.get("test_consonant_accuracy")
        vow_v = payload.get("vowel_accuracy") or payload.get("test_vowel_accuracy")
        if pred.is_file():
            frame = pd.read_csv(pred)
            accuracy = float(frame["correct"].mean())
            cons_v = float(frame.loc[frame["sign_type"] == "consonant", "correct"].mean())
            vow_v = float(frame.loc[frame["sign_type"] == "vowel", "correct"].mean())
        acc.append((seed, accuracy))
        f1.append((seed, float(payload.get("test_macro_f1") or payload.get("macro_f1") or 0.0)))
        if cons_v is not None:
            cons.append((seed, float(cons_v)))
        if vow_v is not None:
            vow.append((seed, float(vow_v)))
    return {
        "accuracy": seed_pool(acc),
        "macro_f1": seed_pool(f1),
        "consonant": seed_pool(cons) if cons else (float("nan"), 0.0, []),
        "vowel": seed_pool(vow) if vow else (float("nan"), 0.0, []),
    }


def collect_enrollment(result_dir: Path, pattern: str, method: str) -> tuple[float, float, list[float]]:
    pairs = []
    for path in sorted(result_dir.glob(pattern)):
        payload = json.loads(path.read_text())
        seed = int(payload["seed"])
        pairs.append((seed, float(payload["methods"][method]["accuracy"])))
    return seed_pool(pairs)


def proto_confusion(run_dir: Path, labels: list[str], output: Path) -> np.ndarray:
    matrices = []
    for path in sorted(run_dir.glob("*/proto_bilstm/*/test_predictions.csv")):
        frame = pd.read_csv(path, keep_default_na=False)
        counts = np.zeros((len(labels), len(labels)), dtype=float)
        for true_i, pred_i in zip(frame["true_index"], frame["predicted_index"], strict=True):
            counts[int(true_i), int(pred_i)] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrices.append(counts / row_sums)
    mean = np.mean(matrices, axis=0)
    pd.DataFrame(mean, index=labels, columns=labels).to_csv(output)
    return mean


def zero_shot_figure(primary: pd.DataFrame, models: dict[str, dict], output_dir: Path) -> None:
    order = [
        "single_frame_cnn",
        "cnn_transformer",
        "xyz_bilstm",
        "crop_cnn",
        "raw_angle_bilstm",
        "proto_xyz",
        "proto_raw_angle",
    ]
    values = {
        "single_frame_cnn": (float(primary.loc["single_frame_cnn", "accuracy_mean"]), float(primary.loc["single_frame_cnn", "accuracy_sd"])),
        "cnn_transformer": (float(primary.loc["cnn_transformer", "accuracy_mean"]), float(primary.loc["cnn_transformer", "accuracy_sd"])),
        "xyz_bilstm": (float(primary.loc["landmark_bilstm", "accuracy_mean"]), float(primary.loc["landmark_bilstm", "accuracy_sd"])),
        "crop_cnn": models["crop_cnn"]["accuracy"][:2],
        "raw_angle_bilstm": models["landmark_bilstm"]["accuracy"][:2],
        "proto_xyz": models["proto_bilstm_xyz"]["accuracy"][:2],
        "proto_raw_angle": models["proto_bilstm"]["accuracy"][:2],
    }
    fig, ax = plt.subplots(figsize=(7.4, 3.15))
    x = np.arange(len(order))
    means = [values[name][0] for name in order]
    sds = [values[name][1] for name in order]
    ax.bar(x, means, color=[mpl.colors.to_rgba(BAR_COLORS[name], 0.78) for name in order], edgecolor=[BAR_COLORS[name] for name in order], width=0.72)
    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    for index, mean in enumerate(means):
        ax.text(index, mean + sds[index] + 0.015, f"{100 * mean:.1f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1 / 49, color="#666666", linestyle="--", linewidth=0.8)
    ax.text(len(order) - 0.02, 1 / 49 + 0.012, "chance 2%", ha="right", va="bottom", fontsize=7, color="#555555")
    ax.set_xticks(x, [LABELS[name].replace(" ", "\n") for name in order], fontsize=7)
    ax.set_ylabel("Pooled unseen-signer accuracy")
    ax.set_ylim(0, 0.72)
    fig.text(0.5, 0.01, "Mean ± SD over three seeds after pooling S1 and S2 (196 clips/seed). Same folds as the 30-run RGB/xyz matrix.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output_dir, "figure_9_zero_shot_improvements")


def enrollment_figure(enroll: dict[str, tuple[float, float]], output_dir: Path) -> None:
    families = [
        ("xyz BiLSTM", enroll["xyz_zero"], enroll["xyz_closed"]),
        ("xyz+angle BiLSTM", enroll["angle_zero"], enroll["angle_closed"]),
        ("xyz+angle Proto-BiLSTM", enroll["proto_zero"], enroll["proto_closed"]),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    x = np.arange(len(families))
    width = 0.36
    zero = [row[1][0] for row in families]
    closed = [row[2][0] for row in families]
    zero_sd = [row[1][1] for row in families]
    closed_sd = [row[2][1] for row in families]
    ax.bar(x - width / 2, zero, width, label="Unseen signer (train prototypes / softmax)", color="#0072B2", alpha=0.82)
    ax.bar(x + width / 2, closed, width, label="Closed 1-shot (other lighting)", color="#009E73", alpha=0.82)
    ax.errorbar(x - width / 2, zero, yerr=zero_sd, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.errorbar(x + width / 2, closed, yerr=closed_sd, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.set_xticks(x, [row[0] for row in families], fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.text(0.5, 0.01, "Closed 1-shot uses the other lighting of all 49 classes from the held-out signer. Mean ± SD over three seeds.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output_dir, "figure_10_enrollment")


def write_tables(models: dict, enroll: dict, fusion: dict, primary: pd.DataFrame, output: Path) -> None:
    xyz = (
        float(primary.loc["landmark_bilstm", "accuracy_mean"]),
        float(primary.loc["landmark_bilstm", "accuracy_sd"]),
        float(primary.loc["landmark_bilstm", "macro_f1_mean"]),
        float(primary.loc["landmark_bilstm", "macro_f1_sd"]),
    )
    lines = [
        "% Auto-generated from the improvement matrix. Do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Follow-on unseen-signer accuracy on the same S1/S2 folds (196 clips/seed; mean $\pm$ SD over three seeds). The xyz BiLSTM row is the frozen 30-run control.}",
        r"\label{tab:improve}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Acc. (\%) & Macro-F1 (\%) & Cons. (\%) & Vow. (\%) \\",
        r"\midrule",
        f"xyz BiLSTM (CE) & {pm(xyz[0], xyz[1])} & {pm(xyz[2], xyz[3])} & 10.0 $\\pm$ 3.9 & 66.0 $\\pm$ 4.0 \\\\",
        f"Hand-crop CNN & {pm(*models['crop_cnn']['accuracy'][:2])} & {pm(*models['crop_cnn']['macro_f1'][:2])} & {pm(*models['crop_cnn']['consonant'][:2])} & {pm(*models['crop_cnn']['vowel'][:2])} \\\\",
        f"xyz+angle MLP (CE) & {pm(*models['landmark_mlp']['accuracy'][:2])} & {pm(*models['landmark_mlp']['macro_f1'][:2])} & {pm(*models['landmark_mlp']['consonant'][:2])} & {pm(*models['landmark_mlp']['vowel'][:2])} \\\\",
        f"xyz+angle BiLSTM (CE) & {pm(*models['landmark_bilstm']['accuracy'][:2])} & {pm(*models['landmark_bilstm']['macro_f1'][:2])} & {pm(*models['landmark_bilstm']['consonant'][:2])} & {pm(*models['landmark_bilstm']['vowel'][:2])} \\\\",
        f"xyz Proto-BiLSTM & {pm(*models['proto_bilstm_xyz']['accuracy'][:2])} & {pm(*models['proto_bilstm_xyz']['macro_f1'][:2])} & {pm(*models['proto_bilstm_xyz']['consonant'][:2])} & {pm(*models['proto_bilstm_xyz']['vowel'][:2])} \\\\",
        f"xyz+angle Proto-BiLSTM & {pm(*models['proto_bilstm']['accuracy'][:2])} & {pm(*models['proto_bilstm']['macro_f1'][:2])} & {pm(*models['proto_bilstm']['consonant'][:2])} & {pm(*models['proto_bilstm']['vowel'][:2])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Closed 1-shot enrollment: the other lighting of the held-out signer supplies one support clip per class (98 queries/fold). Mean $\pm$ SD over three seeds after pooling S1 and S2.}",
        r"\label{tab:enroll}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Encoder & Unseen signer (\%) & Closed 1-shot (\%) \\",
        r"\midrule",
        f"xyz BiLSTM (CE) & {pm(*enroll['xyz_zero'][:2])} & {pm(*enroll['xyz_closed'][:2])} \\\\",
        f"xyz+angle BiLSTM (CE) & {pm(*enroll['angle_zero'][:2])} & {pm(*enroll['angle_closed'][:2])} \\\\",
        f"xyz+angle Proto-BiLSTM & {pm(*enroll['proto_zero'][:2])} & {pm(*enroll['proto_closed'][:2])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    configure_style()
    run_dir = Path("artifacts/runs/improvements")
    result_dir = Path("artifacts/results/improvements")
    figure_dir = Path("artifacts/figures")
    labels = json.loads(Path("configs/nsl23_classes.json").read_text())
    class_list = [*labels["consonants"], *labels["vowels"]]
    primary = pd.read_csv("artifacts/results/model_summary.csv").set_index("model")
    models = {
        name: collect_model(run_dir, name)
        for name in ("landmark_mlp", "landmark_bilstm", "crop_cnn", "proto_bilstm", "proto_bilstm_xyz")
    }
    enroll = {
        "xyz_zero": collect_enrollment(result_dir, "enrollment_heldout_*.json", "softmax"),
        "xyz_closed": collect_enrollment(result_dir, "enrollment_heldout_*.json", "enroll_closed_other_lighting"),
        "angle_zero": collect_enrollment(result_dir, "enrollment_rawangle_bilstm_*.json", "softmax"),
        "angle_closed": collect_enrollment(result_dir, "enrollment_rawangle_bilstm_*.json", "enroll_closed_other_lighting"),
        "proto_zero": collect_enrollment(result_dir, "enrollment_proto_*.json", "train_prototypes"),
        "proto_closed": collect_enrollment(result_dir, "enrollment_proto_*.json", "enroll_closed_other_lighting"),
    }
    fusion = {}
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = {"models": {k: {mk: {"mean": v[0], "sd": v[1], "seeds": v[2]} for mk, v in block.items()} for k, block in models.items()}, "enrollment": {k: {"mean": v[0], "sd": v[1], "seeds": v[2]} for k, v in enroll.items()}}
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    proto_confusion(run_dir, class_list, result_dir / "confusion_proto_bilstm_mean_normalized.csv")
    matrix = pd.read_csv(result_dir / "confusion_proto_bilstm_mean_normalized.csv", index_col=0, keep_default_na=False)
    fig, ax = plt.subplots(figsize=(8.1, 7.2))
    sns.heatmap(matrix, ax=ax, cmap="mako", vmin=0, vmax=1, square=True, cbar_kws={"label": "Mean row-normalized proportion", "shrink": 0.75})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Prototypical xyz+angle BiLSTM confusion, averaged over folds and seeds")
    ax.tick_params(axis="x", labelrotation=90, labelsize=5.5)
    ax.tick_params(axis="y", labelrotation=0, labelsize=5.5)
    save_figure(fig, figure_dir, "figure_11_proto_bilstm_confusion")
    zero_shot_figure(primary, models, figure_dir)
    enrollment_figure(enroll, figure_dir)
    write_tables(models, enroll, fusion, primary, Path("manuscript/generated_improvement_tables.tex"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

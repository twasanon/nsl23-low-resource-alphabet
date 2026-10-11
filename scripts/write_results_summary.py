#!/usr/bin/env python3
"""Write a human-readable summary directly from audited result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ORDER = ("single_frame_cnn", "cnn_bilstm", "cnn_transformer", "landmark_mlp", "landmark_bilstm")
LABEL = {
    "single_frame_cnn": "Single-frame CNN",
    "cnn_bilstm": "CNN–BiLSTM",
    "cnn_transformer": "CNN–Transformer",
    "landmark_mlp": "Landmark MLP",
    "landmark_bilstm": "Landmark BiLSTM",
}


def pm(mean: float, sd: float) -> str:
    return f"{100 * mean:.1f}% ± {100 * sd:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--runtime", type=Path, default=Path("artifacts/runtime/runtime_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/RESULTS_SUMMARY.md"))
    args = parser.parse_args()

    model = pd.read_csv(args.results_dir / "model_summary.csv").set_index("model")
    fold = pd.read_csv(args.results_dir / "fold_summary.csv").set_index(["model", "fold"])
    condition = pd.read_csv(args.results_dir / "condition_summary.csv")
    condition["lighting"] = condition["lighting"].str.lower()
    condition = condition.set_index(["model", "lighting"])
    sign_type = pd.read_csv(args.results_dir / "sign_type_summary.csv").set_index(["model", "sign_type"])
    paired = pd.read_csv(args.results_dir / "paired_differences_summary.csv").set_index(["comparison", "metric"])
    confusions = pd.read_csv(
        args.results_dir / "top_confusions_landmark_bilstm.csv", keep_default_na=False
    ).head(10)
    runtime = pd.read_csv(args.runtime).set_index("model")

    lines = [
        "# NSL23 frozen primary results",
        "",
        "All values below are generated from the complete 30-run matrix (five models, two folds, three seeds). Each seed pools the two held-out folds (196 unique clips), and architecture summaries report the mean and sample SD over seeds.",
        "",
        "## Primary performance",
        "",
        "| Model | Accuracy | Macro precision | Macro recall | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        row = model.loc[name]
        lines.append(
            f"| {LABEL[name]} | {pm(row.accuracy_mean, row.accuracy_sd)} | "
            f"{pm(row.macro_precision_mean, row.macro_precision_sd)} | "
            f"{pm(row.macro_recall_mean, row.macro_recall_sd)} | {pm(row.macro_f1_mean, row.macro_f1_sd)} |"
        )
    best_accuracy = model.accuracy_mean.idxmax()
    best_f1 = model.macro_f1_mean.idxmax()
    lines.extend(
        [
            "",
            f"Highest mean accuracy: **{LABEL[best_accuracy]}** ({pm(model.loc[best_accuracy, 'accuracy_mean'], model.loc[best_accuracy, 'accuracy_sd'])}).",
            f"Highest mean macro-F1: **{LABEL[best_f1]}** ({pm(model.loc[best_f1, 'macro_f1_mean'], model.loc[best_f1, 'macro_f1_sd'])}).",
            "",
            "Equal class support makes pooled accuracy and macro recall numerically identical. No inferential significance claim is made from three seeds and two held-out signers.",
            "",
            "## Seed-matched descriptive differences",
            "",
            "| Comparison | Accuracy difference, percentage points | Macro-F1 difference, percentage points |",
            "|---|---:|---:|",
        ]
    )
    for comparison, label in (
        ("cnn_transformer_minus_single_frame_cnn", "CNN–Transformer minus single-frame CNN"),
        ("cnn_transformer_minus_cnn_bilstm", "CNN–Transformer minus CNN–BiLSTM"),
        ("cnn_bilstm_minus_single_frame_cnn", "CNN–BiLSTM minus single-frame CNN"),
        ("landmark_bilstm_minus_single_frame_cnn", "Landmark BiLSTM minus single-frame CNN"),
        ("landmark_bilstm_minus_cnn_transformer", "Landmark BiLSTM minus CNN–Transformer"),
        ("landmark_bilstm_minus_landmark_mlp", "Landmark BiLSTM minus landmark MLP"),
    ):
        acc = paired.loc[(comparison, "accuracy")]
        f1 = paired.loc[(comparison, "macro_f1")]
        lines.append(
            f"| {label} | {100 * acc.difference_mean:+.1f} ± {100 * acc.difference_sd:.1f} | "
            f"{100 * f1.difference_mean:+.1f} ± {100 * f1.difference_sd:.1f} |"
        )
    lines.extend(
        [
            "",
            "Differences are paired by seed after pooling the same held-out clips. They are descriptive, not hypothesis-test estimates.",
            "",
            "## Held-out signers",
            "",
            "| Model | S1 accuracy | S1 macro-F1 | S2 accuracy | S2 macro-F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ORDER:
        s1 = fold.loc[(name, "heldout_s1")]
        s2 = fold.loc[(name, "heldout_s2")]
        lines.append(
            f"| {LABEL[name]} | {pm(s1.test_accuracy_mean, s1.test_accuracy_sd)} | "
            f"{pm(s1.test_macro_f1_mean, s1.test_macro_f1_sd)} | "
            f"{pm(s2.test_accuracy_mean, s2.test_accuracy_sd)} | {pm(s2.test_macro_f1_mean, s2.test_macro_f1_sd)} |"
        )

    lines.extend(["", "## Recording condition", "", "| Model | Bright accuracy | Dark accuracy | Bright macro-F1 | Dark macro-F1 |", "|---|---:|---:|---:|---:|"])
    for name in ORDER:
        bright = condition.loc[(name, "bright")]
        dark = condition.loc[(name, "dark")]
        lines.append(
            f"| {LABEL[name]} | {pm(bright.accuracy_mean, bright.accuracy_sd)} | "
            f"{pm(dark.accuracy_mean, dark.accuracy_sd)} | "
            f"{pm(bright.macro_f1_mean, bright.macro_f1_sd)} | {pm(dark.macro_f1_mean, dark.macro_f1_sd)} |"
        )

    lines.extend(["", "## Sign type", "", "| Model | Consonant accuracy | Vowel accuracy | Consonant macro-F1 | Vowel macro-F1 |", "|---|---:|---:|---:|---:|"])
    for name in ORDER:
        consonant = sign_type.loc[(name, "consonant")]
        vowel = sign_type.loc[(name, "vowel")]
        lines.append(
            f"| {LABEL[name]} | {pm(consonant.accuracy_mean, consonant.accuracy_sd)} | "
            f"{pm(vowel.accuracy_mean, vowel.accuracy_sd)} | "
            f"{pm(consonant.macro_f1_supported_classes_mean, consonant.macro_f1_supported_classes_sd)} | "
            f"{pm(vowel.macro_f1_supported_classes_mean, vowel.macro_f1_supported_classes_sd)} |"
        )

    lines.extend(["", "## Most frequent Landmark BiLSTM off-diagonal predictions", "", "| True label | Predicted label | Count across folds and seeds | Repeated true-class predictions |", "|---|---|---:|---:|"])
    for row in confusions.itertuples():
        lines.append(
            f"| {row.true_label} | {row.predicted_label} | {int(row.count_across_seeds_and_folds)} | "
            f"{int(row.repeated_true_class_predictions)} |"
        )

    lines.extend(["", "## Batch-one runtime", "", "| Model | Model-only median (p95), ms | End-to-end median (p95), ms | End-to-end clips/s |", "|---|---:|---:|---:|"])
    for name in ORDER:
        if name not in runtime.index:
            continue
        row = runtime.loc[name]
        lines.append(
            f"| {LABEL[name]} | {row.model_only_median_ms:.1f} ({row.model_only_p95_ms:.1f}) | "
            f"{row.end_to_end_median_ms:.1f} ({row.end_to_end_p95_ms:.1f}) | {row.end_to_end_clips_per_second:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Training excludes the evaluated signer within each fold, but S1/S2 results informed follow-on choices. This is exploratory development evidence, not a confirmatory unseen-signer test. It does not establish signer-population generalization, outdoor/prepared-condition robustness, continuous-sign recognition, streaming performance, or state-of-the-art status. Clip-level bootstrap intervals in the machine-readable results are descriptive and cannot replace signer-level uncertainty.",
            "",
        ]
    )
    followon_path = args.results_dir / "improvements/summary.json"
    if followon_path.is_file():
        followon = json.loads(followon_path.read_text(encoding="utf-8"))
        lines.extend(["## Follow-on matrix (same development folds)", "",
                      "| Model | Accuracy | Macro-F1 | Consonants | Vowels |",
                      "|---|---:|---:|---:|---:|"])
        control = model.loc["landmark_bilstm"]
        control_signs = sign_type.loc["landmark_bilstm"]
        lines.append(
            f"| xyz BiLSTM (CE, frozen control) | {pm(control.accuracy_mean, control.accuracy_sd)} | "
            f"{pm(control.macro_f1_mean, control.macro_f1_sd)} | "
            f"{pm(control_signs.loc['consonant', 'accuracy_mean'], control_signs.loc['consonant', 'accuracy_sd'])} | "
            f"{pm(control_signs.loc['vowel', 'accuracy_mean'], control_signs.loc['vowel', 'accuracy_sd'])} |"
        )
        for key, label in (("crop_cnn", "Hand-crop CNN"), ("landmark_bilstm", "xyz+angle BiLSTM (CE)"),
                           ("proto_bilstm_xyz", "xyz Proto-BiLSTM"), ("proto_bilstm", "xyz+angle Proto-BiLSTM")):
            scores = followon["models"][key]
            cells = [pm(scores[metric]["mean"], scores[metric]["sd"])
                     for metric in ("accuracy", "macro_f1", "consonant", "vowel")]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.extend(["", "Closed 1-shot uses one other-lighting support clip for every class of the held-out signer:", ""])
        for key, label in (("xyz_closed", "xyz BiLSTM"), ("angle_closed", "xyz+angle BiLSTM"),
                           ("proto_closed", "prototypical encoder")):
            scores = followon["enrollment"][key]
            lines.append(f"- {label}: {pm(scores['mean'], scores['sd'])}.")
        lines.extend(["", "Do not report `enroll_replace_class` as 1-shot accuracy.", ""])
        alternatives = json.loads((args.results_dir / "improvements/alternative_protocols.json").read_text(encoding="utf-8"))
        lines.extend(["## Same prototypical encoder, different evaluation tasks", "",
                      "| Task | Chance | Score |", "|---|---:|---:|"])
        for key, label in (("top1", "49-way top-1, unseen development signer"),
                           ("top3", "49-way top-3, unseen development signer"),
                           ("top5", "49-way top-5, unseen development signer"),
                           ("consonant_36way", "36-way consonants, unseen development signer"),
                           ("five_way_unseen", "5-way restriction, full training-class means (not 1-shot)"),
                           ("five_way_same_signer", "5-way 1-shot, same signer")):
            scores = alternatives[key]
            lines.append(f"| {label} | {100 * alternatives['chance'][key]:.1f}% | {pm(scores['mean'], scores['sd'])} |")
        lines.extend(["", "Do not quote five-class or top-5 scores as 49-way top-1 alphabet accuracy.", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

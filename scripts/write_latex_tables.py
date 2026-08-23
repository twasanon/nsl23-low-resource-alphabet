#!/usr/bin/env python3
"""Write LaTeX result tables from audited CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ORDER = ("single_frame_cnn", "cnn_bilstm", "cnn_transformer", "landmark_mlp", "landmark_bilstm")
LABEL = {
    "single_frame_cnn": "Single-frame CNN",
    "cnn_bilstm": "CNN--BiLSTM",
    "cnn_transformer": "CNN--Transformer",
    "landmark_mlp": "Landmark MLP",
    "landmark_bilstm": "Landmark BiLSTM",
}


def pm(mean: float, sd: float, *, percent: bool = True) -> str:
    scale = 100.0 if percent else 1.0
    return f"{scale * mean:.1f} $\\pm$ {scale * sd:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--runtime", type=Path, default=Path("artifacts/runtime/runtime_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("manuscript/generated_tables.tex"))
    args = parser.parse_args()

    model = pd.read_csv(args.results_dir / "model_summary.csv").set_index("model")
    fold = pd.read_csv(args.results_dir / "fold_summary.csv").set_index(["model", "fold"])
    condition = pd.read_csv(args.results_dir / "condition_summary.csv")
    condition["lighting"] = condition["lighting"].str.lower()
    condition = condition.set_index(["model", "lighting"])
    sign_type = pd.read_csv(args.results_dir / "sign_type_summary.csv").set_index(["model", "sign_type"])
    runtime = pd.read_csv(args.runtime).set_index("model")

    lines = [
        "% Auto-generated from artifacts/results. Do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pooled unseen-signer performance (196 clips/seed; mean $\pm$ SD over three seeds). Chance accuracy for 49 classes is $2.0\%$.}",
        r"\label{tab:primary}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Acc. (\%) & Macro-P (\%) & Macro-R (\%) & Macro-F1 (\%) \\",
        r"\midrule",
    ]
    for name in ORDER:
        row = model.loc[name]
        lines.append(
            f"{LABEL[name]} & {pm(row.accuracy_mean, row.accuracy_sd)} & "
            f"{pm(row.macro_precision_mean, row.macro_precision_sd)} & "
            f"{pm(row.macro_recall_mean, row.macro_recall_sd)} & "
            f"{pm(row.macro_f1_mean, row.macro_f1_sd)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Held-out signer accuracy and macro-F1 (98 clips/fold; mean $\pm$ SD over three seeds).}",
            r"\label{tab:folds}",
            r"\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Model & S1 acc. (\%) & S1 F1 (\%) & S2 acc. (\%) & S2 F1 (\%) \\",
            r"\midrule",
        ]
    )
    for name in ORDER:
        s1 = fold.loc[(name, "heldout_s1")]
        s2 = fold.loc[(name, "heldout_s2")]
        lines.append(
            f"{LABEL[name]} & {pm(s1.test_accuracy_mean, s1.test_accuracy_sd)} & "
            f"{pm(s1.test_macro_f1_mean, s1.test_macro_f1_sd)} & "
            f"{pm(s2.test_accuracy_mean, s2.test_accuracy_sd)} & "
            f"{pm(s2.test_macro_f1_mean, s2.test_macro_f1_sd)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Descriptive bright versus dark accuracy (98 pooled clips/condition/seed).}",
            r"\label{tab:lighting}",
            r"\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Model & Bright acc. (\%) & Dark acc. (\%) & Bright F1 (\%) & Dark F1 (\%) \\",
            r"\midrule",
        ]
    )
    for name in ORDER:
        bright = condition.loc[(name, "bright")]
        dark = condition.loc[(name, "dark")]
        lines.append(
            f"{LABEL[name]} & {pm(bright.accuracy_mean, bright.accuracy_sd)} & "
            f"{pm(dark.accuracy_mean, dark.accuracy_sd)} & "
            f"{pm(bright.macro_f1_mean, bright.macro_f1_sd)} & "
            f"{pm(dark.macro_f1_mean, dark.macro_f1_sd)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Consonant versus vowel performance on the pooled out-of-fold clips. Accuracy uses all clips in the named sign type; F1 is macro-averaged only over classes supported within that sign-type subset (36 consonants or 13 vowels), not over all 49 classes.}",
            r"\label{tab:signtype}",
            r"\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Model & Cons. acc. (\%) & Vowel acc. (\%) & Cons. F1 (\%) & Vowel F1 (\%) \\",
            r"\midrule",
        ]
    )
    for name in ORDER:
        consonant = sign_type.loc[(name, "consonant")]
        vowel = sign_type.loc[(name, "vowel")]
        lines.append(
            f"{LABEL[name]} & {pm(consonant.accuracy_mean, consonant.accuracy_sd)} & "
            f"{pm(vowel.accuracy_mean, vowel.accuracy_sd)} & "
            f"{pm(consonant.macro_f1_supported_classes_mean, consonant.macro_f1_supported_classes_sd)} & "
            f"{pm(vowel.macro_f1_supported_classes_mean, vowel.macro_f1_supported_classes_sd)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Batch-one clip latency. RGB end-to-end includes original-video decode on Apple M4 MPS. Landmark end-to-end is MediaPipe Hands on 16 cached frames plus the CPU classifier. Streaming FPS is reported separately.}",
            r"\label{tab:runtime}",
            r"\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Model & Device & Model-only (ms) & End-to-end (ms) & Clips/s \\",
            r"\midrule",
        ]
    )
    device = {"single_frame_cnn": "MPS", "cnn_bilstm": "MPS", "cnn_transformer": "MPS", "landmark_bilstm": "CPU"}
    for name in ORDER:
        if name not in runtime.index:
            continue
        row = runtime.loc[name]
        lines.append(
            f"{LABEL[name]} & {device.get(name, '')} & "
            f"{row.model_only_median_ms:.1f} ({row.model_only_p95_ms:.1f}) & "
            f"{row.end_to_end_median_ms:.1f} ({row.end_to_end_p95_ms:.1f}) & "
            f"{row.end_to_end_clips_per_second:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

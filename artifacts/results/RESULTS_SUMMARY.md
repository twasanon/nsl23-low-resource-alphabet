# NSL23 frozen primary results

All values below are generated from the complete 18-run matrix. Each seed pools the two held-out folds (196 unique clips), and architecture summaries report the mean and sample SD over seeds.

## Primary performance

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Single-frame CNN | 5.8% ± 0.8% | 3.6% ± 0.6% | 5.8% ± 0.8% | 3.5% ± 0.2% |
| CNN–BiLSTM | 7.0% ± 2.1% | 5.2% ± 5.0% | 7.0% ± 2.1% | 4.2% ± 2.1% |
| CNN–Transformer | 6.3% ± 0.8% | 5.7% ± 1.6% | 6.3% ± 0.8% | 4.4% ± 1.3% |

Highest mean accuracy: **CNN–BiLSTM** (7.0% ± 2.1%).
Highest mean macro-F1: **CNN–Transformer** (4.4% ± 1.3%).

Equal class support makes pooled accuracy and macro recall numerically identical. No inferential significance claim is made from three seeds and two held-out signers.

## Seed-matched descriptive differences

| Comparison | Accuracy difference, percentage points | Macro-F1 difference, percentage points |
|---|---:|---:|
| CNN–Transformer minus single-frame CNN | +0.5 ± 1.0 | +0.9 ± 1.2 |
| CNN–Transformer minus CNN–BiLSTM | -0.7 ± 1.8 | +0.2 ± 2.2 |
| CNN–BiLSTM minus single-frame CNN | +1.2 ± 1.5 | +0.6 ± 1.9 |

Differences are paired by seed after pooling the same held-out clips. They are descriptive, not hypothesis-test estimates.

## Held-out signers

| Model | S1 accuracy | S1 macro-F1 | S2 accuracy | S2 macro-F1 |
|---|---:|---:|---:|---:|
| Single-frame CNN | 6.5% ± 0.6% | 2.8% ± 0.9% | 5.1% ± 1.0% | 2.9% ± 1.3% |
| CNN–BiLSTM | 8.2% ± 2.0% | 5.0% ± 3.2% | 5.8% ± 2.4% | 2.0% ± 1.1% |
| CNN–Transformer | 5.1% ± 1.0% | 3.7% ± 1.3% | 7.5% ± 1.2% | 3.0% ± 1.4% |

## Recording condition

| Model | Bright accuracy | Dark accuracy | Bright macro-F1 | Dark macro-F1 |
|---|---:|---:|---:|---:|
| Single-frame CNN | 5.8% ± 1.6% | 5.8% ± 0.6% | 3.2% ± 1.1% | 3.0% ± 0.7% |
| CNN–BiLSTM | 7.1% ± 1.0% | 6.8% ± 3.3% | 3.6% ± 1.6% | 4.0% ± 3.2% |
| CNN–Transformer | 7.1% ± 0.0% | 5.4% ± 1.6% | 4.5% ± 0.3% | 3.0% ± 1.5% |

## Sign type

| Model | Consonant accuracy | Vowel accuracy | Consonant macro-F1 | Vowel macro-F1 |
|---|---:|---:|---:|---:|
| Single-frame CNN | 4.2% ± 0.7% | 10.3% ± 2.2% | 2.8% ± 0.5% | 10.9% ± 2.6% |
| CNN–BiLSTM | 5.6% ± 1.8% | 10.9% ± 2.9% | 4.2% ± 1.8% | 8.8% ± 3.5% |
| CNN–Transformer | 4.9% ± 0.7% | 10.3% ± 2.9% | 4.0% ± 0.8% | 9.5% ± 2.6% |

## Most frequent CNN–Transformer off-diagonal predictions

| True label | Predicted label | Count across folds and seeds | Repeated true-class predictions |
|---|---|---:|---:|
| AM | AH | 6 | 12 |
| A | AH | 4 | 12 |
| BHA | AH | 4 | 12 |
| CHA | AH | 4 | 12 |
| GYA | AH | 4 | 12 |
| II | T_SHA | 4 | 12 |
| JHA | AH | 4 | 12 |
| KHA | AH | 4 | 12 |
| NA | AH | 4 | 12 |
| NAA | RI | 4 | 12 |

## Batch-one runtime

| Model | Model-only median (p95), ms | End-to-end median (p95), ms | End-to-end clips/s |
|---|---:|---:|---:|
| Single-frame CNN | 4.0 (4.2) | 939.0 (2097.8) | 1.06 |
| CNN–BiLSTM | 49.5 (50.2) | 962.9 (2126.2) | 1.04 |
| CNN–Transformer | 49.7 (50.1) | 970.9 (2162.6) | 1.03 |

## Interpretation boundary

These are strict unseen-signer results for S1 and S2 under the available complete 49-class bright/dark tests. They do not establish signer-population generalization, outdoor/prepared-condition robustness, continuous-sign recognition, streaming performance, or state-of-the-art status. Clip-level bootstrap intervals in the machine-readable results are descriptive and cannot replace signer-level uncertainty.

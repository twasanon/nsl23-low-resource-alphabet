# NSL23 frozen primary results

All values below are generated from the complete 30-run matrix (five models, two folds, three seeds). Each seed pools the two held-out folds (196 unique clips), and architecture summaries report the mean and sample SD over seeds.

## Primary performance

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Single-frame CNN | 5.8% ± 0.8% | 3.6% ± 0.6% | 5.8% ± 0.8% | 3.5% ± 0.2% |
| CNN–BiLSTM | 7.0% ± 2.1% | 5.2% ± 5.0% | 7.0% ± 2.1% | 4.2% ± 2.1% |
| CNN–Transformer | 6.3% ± 0.8% | 5.7% ± 1.6% | 6.3% ± 0.8% | 4.4% ± 1.3% |
| Landmark MLP | 18.0% ± 2.1% | 10.3% ± 1.5% | 18.0% ± 2.1% | 11.1% ± 1.8% |
| Landmark BiLSTM | 24.8% ± 4.0% | 17.1% ± 5.6% | 24.8% ± 4.0% | 16.8% ± 3.8% |

Highest mean accuracy: **Landmark BiLSTM** (24.8% ± 4.0%).
Highest mean macro-F1: **Landmark BiLSTM** (16.8% ± 3.8%).

Equal class support makes pooled accuracy and macro recall numerically identical. No inferential significance claim is made from three seeds and two held-out signers.

## Seed-matched descriptive differences

| Comparison | Accuracy difference, percentage points | Macro-F1 difference, percentage points |
|---|---:|---:|
| CNN–Transformer minus single-frame CNN | +0.5 ± 1.0 | +0.9 ± 1.2 |
| CNN–Transformer minus CNN–BiLSTM | -0.7 ± 1.8 | +0.2 ± 2.2 |
| CNN–BiLSTM minus single-frame CNN | +1.2 ± 1.5 | +0.6 ± 1.9 |
| Landmark BiLSTM minus single-frame CNN | +19.0 ± 3.7 | +13.2 ± 3.9 |
| Landmark BiLSTM minus CNN–Transformer | +18.5 ± 4.6 | +12.4 ± 5.1 |
| Landmark BiLSTM minus landmark MLP | +6.8 ± 1.9 | +5.6 ± 2.8 |

Differences are paired by seed after pooling the same held-out clips. They are descriptive, not hypothesis-test estimates.

## Held-out signers

| Model | S1 accuracy | S1 macro-F1 | S2 accuracy | S2 macro-F1 |
|---|---:|---:|---:|---:|
| Single-frame CNN | 6.5% ± 0.6% | 2.8% ± 0.9% | 5.1% ± 1.0% | 2.9% ± 1.3% |
| CNN–BiLSTM | 8.2% ± 2.0% | 5.0% ± 3.2% | 5.8% ± 2.4% | 2.0% ± 1.1% |
| CNN–Transformer | 5.1% ± 1.0% | 3.7% ± 1.3% | 7.5% ± 1.2% | 3.0% ± 1.4% |
| Landmark MLP | 18.7% ± 3.1% | 11.6% ± 2.5% | 17.3% ± 1.0% | 10.1% ± 0.7% |
| Landmark BiLSTM | 25.9% ± 2.6% | 17.4% ± 2.0% | 23.8% ± 5.6% | 15.0% ± 4.8% |

## Recording condition

| Model | Bright accuracy | Dark accuracy | Bright macro-F1 | Dark macro-F1 |
|---|---:|---:|---:|---:|
| Single-frame CNN | 5.8% ± 1.6% | 5.8% ± 0.6% | 3.2% ± 1.1% | 3.0% ± 0.7% |
| CNN–BiLSTM | 7.1% ± 1.0% | 6.8% ± 3.3% | 3.6% ± 1.6% | 4.0% ± 3.2% |
| CNN–Transformer | 7.1% ± 0.0% | 5.4% ± 1.6% | 4.5% ± 0.3% | 3.0% ± 1.5% |
| Landmark MLP | 21.4% ± 2.0% | 14.6% ± 2.1% | 12.7% ± 2.5% | 8.8% ± 1.2% |
| Landmark BiLSTM | 27.6% ± 3.5% | 22.1% ± 4.6% | 18.4% ± 3.1% | 14.3% ± 4.1% |

## Sign type

| Model | Consonant accuracy | Vowel accuracy | Consonant macro-F1 | Vowel macro-F1 |
|---|---:|---:|---:|---:|
| Single-frame CNN | 4.2% ± 0.7% | 10.3% ± 2.2% | 2.8% ± 0.5% | 10.9% ± 2.6% |
| CNN–BiLSTM | 5.6% ± 1.8% | 10.9% ± 2.9% | 4.2% ± 1.8% | 8.8% ± 3.5% |
| CNN–Transformer | 4.9% ± 0.7% | 10.3% ± 2.9% | 4.0% ± 0.8% | 9.5% ± 2.6% |
| Landmark MLP | 7.2% ± 2.4% | 48.1% ± 1.9% | 5.6% ± 2.5% | 45.0% ± 2.8% |
| Landmark BiLSTM | 10.0% ± 3.9% | 66.0% ± 4.0% | 8.4% ± 4.0% | 64.2% ± 2.7% |

## Most frequent Landmark BiLSTM off-diagonal predictions

| True label | Predicted label | Count across folds and seeds | Repeated true-class predictions |
|---|---|---:|---:|
| BHA | AH | 12 | 12 |
| NA | YAN | 12 | 12 |
| RA | AH | 12 | 12 |
| TA | RI | 12 | 12 |
| NAA | PHA | 10 | 12 |
| GYA | A | 9 | 12 |
| HA | U | 9 | 12 |
| T_SHA | AH | 9 | 12 |
| MA | AH | 8 | 12 |
| PA | RI | 8 | 12 |

## Batch-one runtime

| Model | Model-only median (p95), ms | End-to-end median (p95), ms | End-to-end clips/s |
|---|---:|---:|---:|
| Single-frame CNN | 4.0 (4.2) | 939.0 (2097.8) | 1.06 |
| CNN–BiLSTM | 49.5 (50.2) | 962.9 (2126.2) | 1.04 |
| CNN–Transformer | 49.7 (50.1) | 970.9 (2162.6) | 1.03 |
| Landmark BiLSTM | 0.3 (0.5) | 345.7 (441.2) | 2.89 |

## Interpretation boundary

These are strict unseen-signer results for S1 and S2 under the available complete 49-class bright/dark tests. They do not establish signer-population generalization, outdoor/prepared-condition robustness, continuous-sign recognition, streaming performance, or state-of-the-art status. Clip-level bootstrap intervals in the machine-readable results are descriptive and cannot replace signer-level uncertainty.

## Follow-on matrix (same folds, not a replacement)

See `artifacts/results/improvements/summary.json`. Pooled unseen-signer accuracy:

| Model | Accuracy | Macro-F1 | Consonants | Vowels |
|---|---:|---:|---:|---:|
| xyz BiLSTM (CE, frozen control) | 24.8% ± 4.0% | 16.8% ± 3.8% | 10.0% ± 3.9% | 66.0% ± 4.0% |
| Hand-crop CNN | 19.2% ± 1.3% | 15.9% ± 0.7% | 15.5% ± 0.8% | 29.5% ± 2.9% |
| xyz+angle BiLSTM (CE) | 37.9% ± 2.4% | 30.1% ± 2.6% | 28.7% ± 4.5% | 63.5% ± 3.8% |
| xyz Proto-BiLSTM | 55.8% ± 1.8% | 50.6% ± 2.4% | 58.3% ± 2.5% | 48.7% ± 5.9% |
| xyz+angle Proto-BiLSTM | 59.4% ± 3.3% | 54.7% ± 4.2% | 61.8% ± 5.7% | 52.6% ± 4.8% |

Closed 1-shot (other lighting of the held-out signer): xyz BiLSTM 76.2% ± 4.2%; xyz+angle BiLSTM 78.2% ± 2.8%; prototypical encoder 73.0% ± 1.5%. Do not report `enroll_replace_class` as 1-shot accuracy.

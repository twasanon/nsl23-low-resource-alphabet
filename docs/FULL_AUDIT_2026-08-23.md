# NSL23 full paper and repository audit — 23 August 2026

## Bottom line

The Version 1 paper is a defensible **exploratory development study**, not a confirmatory estimate for new signers. Its central accuracy result remains correct: the xyz+angle prototypical BiLSTM achieves **59.4% ± 3.3%** accuracy on 196 pooled S1/S2 out-of-fold clips per seed. Dataset counts, folds, accuracies, signer-specific results, enrollment, alternative exams, confusion claims, runtime receipts, and bibliography linkage all trace to real artifacts.

The supplied outside audit was valuable and mostly correct, but it missed one numerical aggregation defect: follow-on macro-F1 had been averaged across fold-level macro-F1 values even though the methods said fold predictions were pooled first. Macro-F1 is nonlinear. The scripts and paper now compute it from the 196 pooled predictions per seed. This changes follow-on macro-F1 values but **does not change any accuracy**, including the 59.4% headline.

The only scientific blocker that remains is the intended one: Version 2 still has zero human reviews. **Do not train Version 2.** Complete two independent human passes, agreement, adjudication, coverage, and freeze first.

## Audit scope and evidence layers

This audit used five layers:

1. Direct inspection of the manuscript, protocol, direction dashboard, claim ledger, reviewer instructions, configuration, loaders, models, training, evaluation, and freeze gates.
2. Live strict verification of 630 manifest rows, 375 primary clips, 375 RGB caches, both 277/98 folds, class completeness, lighting balance, and source existence.
3. Re-aggregation from all per-run prediction and metric files, rather than trusting manuscript tables.
4. Byte-level verification of the dated V1 snapshot against a historical Git tree and MD5 recomputation of all four official dataset ZIPs.
5. Compilation and page-by-page visual inspection of the paper PDF, plus unit and integrity tests.

No model was retrained. Retraining would take hours and would add a cross-hardware determinism question; primitive-level outputs, checkpoints, predictions, metrics, and aggregation logic were audited instead.

Machine-readable evidence is in `artifacts/audit/full_paper_audit.json`. The repeatable command is:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_paper_numbers.py
```

## Evaluation of the supplied external audit

### Correct findings

- The 375-clip primary sample, 277/98 folds, S1/S2 held-out identities, 49 classes, and bright/dark balance are correct.
- All main accuracy results, enrollment results, alternative-exam results, stable confusion claims, training-saturation claims, parameter counts, and runtime receipts are supported.
- The 59.4% result is development evidence because S1/S2 informed follow-on representation choices. Its ±3.3% is sample SD across three seed scores, not a confidence interval for signers.
- The old notebook claim, stale root `results/`, duplicate notebook, unused transformer figures, and machine-specific input paths needed cleanup.
- Table 4 silently changed the macro-F1 class set; the caption needed to say that consonant/vowel F1 uses only the 36/13 supported classes.
- Deterministic-algorithm settings do not guarantee bitwise identity across PyTorch releases, hardware, and devices.
- Version 2 is correctly gated on independent human review and frozen test signers.

### Corrections and additions

| Supplied audit statement | Independent finding |
|---|---|
| Five frozen files mismatch the current tree. | At the audited current state, eight of 15 differ because additional V2/portability edits occurred. This is expected live-tree drift. More importantly, **all 15 match commit `0eb1ff20337b584a5a7e4b06d5e4d5a350915fc7` exactly**. |
| No standalone freeze verifier. | `scripts/verify_protocol_freeze.py` already existed, but it only checked the live tree. It now verifies the historical tree by default and preserves worktree mode as a drift diagnostic. |
| Raw-angle fusion SD is about 2.9 points. | Pooling the two equal-size folds within each seed gives 41.33%, 40.82%, and 42.35%; therefore the correct result is **41.5% ± 0.8%**. The larger value came from treating six fold scores as the summary unit. |
| Follow-on macro-F1 values are exact. | They were fold-averaged rather than prediction-pooled. Correct pooled values are shown below. This was the one material numerical issue missed by that audit. |
| WLASL pages 1448–1458. | The manuscript bibliography’s **1459–1469** is the correct official WACV range. |
| Archive hashes were not recomputed. | This audit recomputed all four ZIP MD5 values successfully before their duplicate local copies were removed. |
| Nine V2 tests pass. | After adding historical-freeze and reviewer-exchange tests, **15 tests** pass. Additional final checks are listed below. |

## Corrected quantitative results

### Frozen primary matrix

Each seed pools 196 unique predictions from S1 and S2 before metrics are computed. The table is unchanged.

| Model | Accuracy | Pooled macro-F1 |
|---|---:|---:|
| Single-frame CNN | 5.8% ± 0.8% | 3.5% ± 0.2% |
| CNN–BiLSTM | 7.0% ± 2.1% | 4.2% ± 2.1% |
| CNN–Transformer | 6.3% ± 0.8% | 4.4% ± 1.3% |
| Landmark MLP (xyz) | 18.0% ± 2.1% | 11.1% ± 1.8% |
| Landmark BiLSTM (xyz) | 24.8% ± 4.0% | 16.8% ± 3.8% |

### Follow-on matrix after correcting macro-F1 pooling

| Model | Accuracy (unchanged) | Correct pooled macro-F1 |
|---|---:|---:|
| Hand-crop CNN | 19.2% ± 1.3% | **16.9% ± 0.9%** |
| xyz+angle MLP (cross entropy) | 22.4% ± 3.1% | **15.6% ± 2.7%** |
| xyz+angle BiLSTM (cross entropy) | 37.9% ± 2.4% | **31.5% ± 3.4%** |
| xyz prototypical BiLSTM | 55.8% ± 1.8% | **53.4% ± 2.8%** |
| xyz+angle prototypical BiLSTM | 59.4% ± 3.3% | **57.4% ± 3.6%** |

For the final row, pooled macro-F1 by seed is 57.84%, 60.69%, and 53.56%. The arithmetic mean is 57.36% and the sample SD (`ddof=1`) is 3.59 points.

### Secondary exams

- Equal-weight proto + crop fusion: 59.7% ± 3.6%.
- Equal-weight raw-angle + crop fusion: **41.5% ± 0.8%**.
- Closed 1-shot other-lighting support: xyz 76.2% ± 4.2%; raw-angle 78.2% ± 2.8%; proto 73.0% ± 1.5%.
- Same proto encoder: top-3 85.5% ± 6.0%; top-5 91.3% ± 4.0%; unseen 36-way consonants 65.7% ± 3.6%; unseen 5-way 1-shot 90.6% ± 2.5%; same-signer 5-way 93.3% ± 1.5%.

These are separate grading conventions. None replaces the 49-way unseen-signer top-1 development result.

## Freeze and chronology audit

The original `protocol_freeze.json` was created before the first full primary training run but recorded `git_revision: null`. Its envelope hash is valid. A search across repository history found one exact tree:

- Snapshot SHA-256: `512c9ded600038a0bf310d820a13310ccecf3c0b3a9df90f92be42549c8221e5`
- Matching commit: `0eb1ff20337b584a5a7e4b06d5e4d5a350915fc7`
- Human-readable tag: `v1-protocol-freeze`
- Result: 15/15 input byte lengths and SHA-256 hashes match.

The dated snapshot remains the authority because the commit binding was reconstructed later. The tag is a convenience pointer, not retroactive preregistration. Later worktree drift is expected and explicitly reported rather than hidden.

## Data and path audit

- 630 source videos are present and decode-audited in the manifest.
- 607 files are non-aggregate, 23 are aggregate tapes, 234 are cropped/preprocessed, and 375 original isolated clips form V1.
- Active manifests and reviewer inputs now use repository-relative paths; loaders resolve them against the repository root.
- Historical metric/environment receipts retain their original machine paths because rewriting historical evidence would be inappropriate.
- A case-only S13 filename mismatch in `tapes.csv` was fixed; it would have failed on a case-sensitive Linux filesystem.

## Storage audit and cleanup

The folder was approximately 23 GiB because it held both 9.6 GiB of official ZIPs and 9.7 GiB of extracted videos, plus caches and tracked checkpoints.

Before cleanup, every ZIP’s byte size and MD5 matched `archive_checksums.json`, and all 630 extracted source paths existed. The four duplicate ZIPs were then removed, saving 10,335,577,763 bytes (9.626 GiB). They are recoverable from DOI `10.5281/zenodo.10478554`; extracted videos were retained. The receipt is `artifacts/audit/storage_cleanup.json`.

Tracked smoke checkpoints/ZIPs, the stale root `results/` tree, one byte-identical notebook copy, and six unused transformer-figure files were removed. This reduces the checkout and prevents future smoke-output growth without rewriting Git history.

## Reviewer-workflow audit

Current V2 state remains:

- 21 aggregate tapes, 22.13 source minutes, about 3.47 GiB uncompressed.
- 691 motion proposals.
- Zero pass-1 decisions, zero pass-2 decisions, zero verified segments.
- S5, S14, S4, and S6 remain permanently sealed from development.

The local interface already hid pass-1 values visually, but an outside reviewer receiving the master CSV could inspect them. The new export/import path fixes this:

- Export refuses incomplete pass 1.
- Pass-2 CSV removes pass-1 decision, reviewer, timestamp, class, bounds, tape-order hint, adjudication, and final inclusion data.
- A sidecar binds proposal identity, original bounds, pass-1 state, class configuration, and hierarchy hash.
- Import rejects tampering, missing proposals, leaked pass-1 fields, invalid classes/bounds, changed protocol/classes, and same-person double review.
- Only pass-2 fields are copied into the master.
- Pass-2 detector-miss insertions are retained but still require pass 1 before freeze.
- The web server binds to localhost and must not be exposed directly to the internet.

## Manuscript and reference audit

- Every BibTeX key is cited and every citation resolves to a BibTeX entry (18/18).
- Dataset, WLASL, MS-ASL, AUTSL, MediaPipe, prototypical-network, optimizer, metric, and bootstrap citations support the claims for which they are used.
- The manuscript correctly separates isolated 49-way classification from spotting, continuous recognition, and live webcam accuracy.
- It now states that MPS/PyTorch determinism is not a cross-version or cross-device bitwise guarantee.
- Table 4 now defines its supported-class macro-F1 denominator.
- Figures 3 and 5 were regenerated with non-overlapping model labels and visually checked.
- Raw-angle fusion now includes the correct SD.
- The data-availability section points to portable manifests, the V1 snapshot verifier, and the V2 protocol.

## Remaining limitations, not repairable by code

1. Only S1 and S2 provide complete isolated-alphabet V1 evaluation seats, and both informed follow-on development.
2. Three seeds estimate optimization variability poorly and say nothing about signer-population variability.
3. The observed proto accuracy differs by roughly eight points between S1 and S2; signer variation is visibly larger than a simplistic reading of seed SD.
4. Closed 1-shot uses the other lighting from the same signer/session, not a second day or independent enrollment session.
5. The task is manually bounded isolated-letter classification. It is not sign spotting, continuous signing, sentence recognition, or deployment evidence.
6. The webcam result is a speed check on recorded streams, not live recognition accuracy.
7. Version 2 can make a confirmatory 49-way claim only if at least two predeclared test-hierarchy signers obtain complete verified coverage.

## Publication readiness verdict

**Version 1:** numerically and reproducibly suitable for submission as an exploratory/negative-results paper after journal-specific formatting and authorship/blinding decisions. Do not remove its development qualifiers. A venue may still reject it for sample size or scope; that is a scientific-positioning risk, not a hidden integrity defect.

**Version 2:** protocol-ready and reviewer-ready, but not data-ready and not train-ready. The next permitted scientific action is reviewer calibration followed by pass 1 and an independent pass 2. If fewer than two complete frozen testers are recovered, publish the annotation resource and V1 exploration without a confirmatory 49-way Stage 4 claim.

## Repeated checks required before any submission

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_protocol_freeze.py
PYTHONPATH=src .venv/bin/python scripts/verify_primary_protocol.py --verify-v1-freeze
PYTHONPATH=src .venv/bin/python scripts/audit_paper_numbers.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=src .venv/bin/python scripts/validate_primary_claims.py
```

Then rebuild the LaTeX PDF and visually inspect every page. After human annotation begins, also run the agreement, coverage, and freeze validators described in `annotations/v2/ANNOTATION_GUIDE.md`.

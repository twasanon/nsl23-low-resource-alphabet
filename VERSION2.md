# NSL23 Version 2 protocol

Status: Stage 1 in progress (2026-08-19). Version 1 is frozen. **Do not train any Version 2 classifier until pass-2 coverage is frozen.** S5 and S14 stay sealed until Stage 4.

The current manuscript (`manuscript/nsl23_low_resource.tex`) is **Version 1**: an honest ablation on 375 original isolated clips with S1 and S2 held out.

## Version 1 freeze (do not retune)

Same 375 clips, same `heldout_s1` / `heldout_s2` folds, three seeds. S1 and S2 have **98 isolated clips each** (49 classes × {bright, dark}), not 49 clips.

| Method | 49-way unseen signer |
|---|---:|
| Full-frame CNN | 5.8% ± 0.8% |
| CNN–BiLSTM | 7.0% ± 2.1% |
| CNN–Transformer | 6.3% ± 0.8% |
| Landmark BiLSTM xyz | 24.8% ± 4.0% |
| + angles, cross entropy | 37.9% ± 2.4% |
| Prototypical xyz | 55.8% ± 1.8% |
| Prototypical xyz+angles | **59.4% ± 3.3%** |

S1/S2 influenced representation choice. They are **development** signers for Version 2, not untouched final testers. That is why they are the right Stage 2 diagnostic set: the 59.4% number was already measured on them.

## Research question (Version 2)

How accurately can a lightweight hand-centric system recognize all 49 NSL alphabet signs from a signer unseen during training?

The headline answer is the **Stage 4** number on frozen testers, produced once. Stage 2 only asks whether extra verified training signers move the old S1/S2 exam.

## Locked order

1. Finish pass-1 / pass-2 annotation. No training.
2. Generate the coverage table (`python scripts/report_v2_coverage.py`).
3. Apply the completeness-only test hierarchy: **S5, S14**, then **S4**, then **S6**. Never choose testers from accuracy.
4. Validation is **S3** if S3 is complete and not a tester. No other validation fallback is defined.
5. Freeze: `python scripts/freeze_v2_protocol.py` writes hashed `verified_segments_v2.0.csv`, `coverage_v2.0.csv`, and `configs/v2_protocol.json`. From this point, test annotation and splits are immutable. Only then are unused fallbacks released into development.
6. **Stage 2 (data only):** exact current prototypical xyz+angle BiLSTM (`scripts/train_protonet.py`, `feature_mode=raw_angle`) on expanded development training data. Evaluate on the **same S1/S2 held-out isolated clips** as Version 1. Compare to 59.4%. Do **not** score S5, S14, or any fallback tester.
7. **Stage 3:** geometry / metric / Poudel / architecture ablations on development + validation only. Never look at frozen-test predictions or accuracy.
8. Freeze architecture, features, preprocessing, hyperparameters, checkpoint selection, and inference.
9. Optionally retrain that frozen specification on all permitted non-test data, including the former validation signer.
10. **Stage 4:** evaluate the frozen testers **exactly once**. Report each tester and the pooled 49-way result. S5 vs S14 is descriptive, not a causal environment effect.
11. Only afterward: leave-one-signer-out, top-k, enrollment, webcam timing. None of those may change the frozen model.

## Stage 1 — Dataset, not the network

Annotate the **21 original** aggregate tapes (23 files minus 2 cropped copies). Crops may later augment training; they are never extra samples.

Motion/pause detection may **propose** cuts. Equal-duration bins are never labels. The detector **keeps every hold**; it does not drop to the longest 13 or 36. Sequential `expected_class_hint` values are a navigation aid only. Dynamic vowels may live **between** holds; the accepted interval is the complete gesture, including motion. Watch the video window. Insert missed signs when a hold was not proposed. Do not force an incomplete tape to 13 or 36 classes.

Every automatic row starts with `include_in_v2=false` and an empty `class`. Nothing is labelled because it was a pause.

Two visual passes:

1. Pass 1: watch a video window around the proposal (±2–3 s), edit start/end (store **frames and seconds**), set class, mark accept / reject / incomplete / ambiguous.
2. Pass 2: later re-review with pass-1 accept/reject hidden. Same person is allowed; a second NSL-knowledgeable reviewer should check every signer in the test hierarchy (**S5, S14, S4, S6**).

A row enters modeling only when `review_pass_2=accepted`, `include_in_v2=true`, and `class` is a canonical 49-way label.

Failed or incomplete tapes stay in `annotations/v2/tape_outcomes.csv`. If visual review recovers 34 of 36 consonants on S14, S14 is **not** complete, even if a real-world tester would be desirable.

The number **572** is an annotation ceiling, not recovered data. The paper may say only: “Manual annotation recovered *N* verified isolated gesture segments from 21 original aggregate recordings.”

Motion-based segmentation produced more or fewer candidate holds than the expected alphabet count for every aggregate recording. Combined with visual inspection showing variable sign and transition durations, that rules out automatic equal-duration labeling. Candidate boundaries are review proposals only; all final segments are manually verified.

## Test hierarchy (completeness only)

Preferred frozen testers: **S5** (prepared camera) and **S14** (unprepared / real-world), if both have 49 verified classes.

If one fails completeness, replace it using **S4 → S6**, subject only to full verified 49-class coverage.

Until `freeze_v2_protocol.py` succeeds:

- S1 and S2 are development (isolated 375 clips).
- S3 is provisional validation if complete, otherwise incomplete.
- **S4 and S6 are `held_as_fallback`, never development.** They cannot be used to tune coordinates, losses, or checkpoints, because they might still become testers.
- Consonant-only people (S7, S8, S10, S11, S12, and verified S9/S13 tapes) may train consonants; they are never 49-way testers.

After freeze, unused fallbacks (complete, but not selected as testers) are released to development. A signer who became a tester never enters development.

Report Stage 4 S5, S14 (or whoever was frozen), and pooled accuracy as descriptive. Do not treat a real-world signer scoring lower as a causal “real-world penalty”: signer and condition change together.

## Stage 2 — Same 59.4% method, S1/S2 exam

This stage isolates extra signer data. It is not the paper’s final test.

- Train: Version 1 training clips plus verified aggregate segments from **development** signers only (no frozen testers, no held fallbacks).
- Evaluate: the original `heldout_s1` / `heldout_s2` isolated clips.
- Model: unchanged prototypical xyz+angle BiLSTM.

If 59.4% rises, the gain is extra verified training diversity, not a new architecture. If it stays near 59.4%, the representation still needs work. Either result is useful. Neither result may be computed on S5/S14.

## Stage 3 — Ablations on development + validation

Only after Stage 2. Never open frozen-test predictions.

1. image xyz (control)
2. world xyz
3. palm-canonical world coordinates
4. joint angles
5. bone vectors / fingertip distances
6. motion features
7. leave-one-out or supervised-contrastive metric learning
8. Poudel 54k pretraining, if available
9. ST-GCN / graph model
10. RGB+geometry fusion only if the crop branch is strong

Then freeze the specification. Stage 4 is one evaluation.

## Commands (Stage 1)

```bash
python scripts/list_v2_tapes.py
python scripts/propose_aggregate_segments.py --from-cache
python scripts/migrate_v2_segment_schema.py
python scripts/review_aggregates.py --port 8765
python scripts/report_v2_coverage.py
python scripts/freeze_v2_protocol.py
```

Do not commit unreviewed motion guesses. After freeze, commit `verified_segments_v2.0.csv`, `coverage_v2.0.csv`, and `configs/v2_protocol.json`.

<!-- V2_FREEZE_HASHES -->
## Freeze hashes

Not frozen. `created_before_any_v2_training` remains true in `configs/v2_protocol.json` until Stage 2 starts; hashes are filled by `scripts/freeze_v2_protocol.py`.

| Artifact | SHA-256 |
|---|---|
| annotations/v2/verified_segments_v2.0.csv | *(not frozen)* |
| annotations/v2/coverage_v2.0.csv | *(not frozen)* |
| configs/v2_protocol.json | *(not frozen)* |
<!-- /V2_FREEZE_HASHES -->

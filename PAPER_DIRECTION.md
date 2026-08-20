# NSL23 paper direction

**Present status (20 August 2026):** Version 1 is frozen. Version 2 is the scientific direction. **Do not train Version 2.** The repository contains 691 motion proposals, zero completed review passes, zero verified segments, and no frozen V2 sample manifests.

The protocol and executable gates are now aligned. Human annotation is the immediate gate, but it must be performed by people who can identify the NSL alphabet. See [`annotations/v2/ANNOTATION_GUIDE.md`](annotations/v2/ANNOTATION_GUIDE.md).

## What you personally need to do

You do not need to label clips merely because you own the project. You should label them only if you genuinely know the 49 NSL alphabet signs. Otherwise, recruit qualified reviewers and coordinate the process.

Before the real review:

1. Create the disposable S3 calibration pilot with `python scripts/make_v2_review_pilot.py`.
2. Run pilot pass 1 and pass 2 with qualified reviewers.
3. If the boundary policy changes, update the protocol while the real review count is still zero, refresh the hierarchy hash, and commit/tag that lock.
4. Do not merge pilot decisions into `annotations/v2/segments.csv`.

Real pass 1:

```bash
python scripts/review_aggregates.py --mode pass1 --reviewer "REVIEWER A" --port 8765
```

Real pass 2, with a different person for every tape:

```bash
python scripts/review_aggregates.py --mode pass2 --reviewer "REVIEWER B" --port 8765
```

Pass 2 starts from the original proposal and hides pass-1 decisions, class, bounds, and the tape-order class hint. Both reviewers must also sweep each full source tape for detector misses. Save the hash-bound pre-adjudication agreement report before resolving conflicts.

## Locked scientific decisions

| Decision | Lock |
|---|---|
| V1 59.4% ± 3.3% | Exploratory development evidence on S1/S2, not an untouched confirmation. |
| Test candidates | S5, S14, S4, S6 remain sealed permanently. Every candidate with 49 verified classes becomes a tester; none is released into development. |
| Minimum Stage 4 | At least two complete frozen testers. Otherwise publish the V1 exploration plus annotation resource without a confirmatory 49-way result. |
| Development | S1/S2; S3 is validation if complete. If S3 is incomplete, Stage 3 is skipped and its verified non-test clips may augment training; other incomplete non-test signers may also contribute verified/original clips. |
| Stage 2 | Existing prototypical image xyz+angle model; S1/S2 isolated development diagnostic only. |
| Stage 3 | Exactly three geometry trials, 40 epochs, seeds 17/42/73, LOSO S1/S2/S3. Select mean per-signer macro-recall. Skip if S3 is incomplete. |
| Final retrain | Required train-only batch on every permitted non-test sample, including S3. It cannot load tester rows. |
| Stage 4 | Evaluation-only batch from the three frozen checkpoints. No retraining or adaptation. |
| Primary endpoint | Use all verified clips. Average clips within class, 49 classes within signer, frozen signers within seed, then report mean ± sample SD over seeds. |
| Uncertainty | Seed SD describes optimization variation only, not signer-population uncertainty. |
| Task | 49-way classification of an isolated, manually bounded gesture from an unseen signer. Not spotting, continuous recognition, or webcam accuracy. |
| Annotation staffing | Two different qualified reviewers for every proposal; a named adjudicator resolves completed-pass disagreements only. |
| Feature cache | 16 frames, 224 px, MediaPipe 0.10.14 and detector settings are locked; every source and cache is hash checked. |
| Model training | BiLSTM and AdamW settings, 40 epochs, 40 episodes, 5 support, 2 query, and seeds 17/42/73 are locked. |

Hierarchy lock: `09a2b7b2cb3a580c6cf3e3cd0c952a04ec4a793863ab27212c0cdeb55c5ab009` in `configs/v2_protocol.json`. This local hash should be anchored by a Git commit/tag before real annotation begins.

## Stop conditions

- No V2 training, prototype construction, feature comparison, or checkpoint selection until `configs/v2_protocol.json` has `"frozen": true`.
- Never use `configs/signer_split.json` for V2.
- Never train on S5, S14, S4, or S6, even when one is incomplete.
- Never use model results to choose which signer becomes test.
- Never force detector proposals to equal 13 vowels or 36 consonants.
- Never run Stage 4 when `confirmatory_49way` is false.
- Never describe Stage 4 as population-level evidence from signers; the fixed test cohort remains small.

## Route to the publishable Version 2 paper

1. Reviewer calibration on disposable S3 proposals.
2. Lock the final annotation policy in Git before real outcomes.
3. Independent full pass 1 and pass 2.
4. Save the hash-bound pre-adjudication agreement artifact; adjudicate disagreements.
5. Coverage, frozen source inventory, and protocol hashes.
6. Build every stage manifest with `scripts/prepare_v2_manifests.py`.
7. Build the frozen three-view landmark cache with `scripts/extract_v2_landmarks.py`.
8. Stage 2 diagnostic.
9. Three-trial Stage 3, or the preregistered skip when S3 is incomplete.
10. Freeze `configs/v2_model_spec.json`.
11. Train three final checkpoints on non-test data only.
12. Run the evaluation-only Stage 4 batch once.
13. Write the V2 manuscript around Stage 4, including annotation agreement and per-tester results.
14. Only afterward run enrollment, top-k, broader LOSO, or webcam engineering analyses.

## Current paper

`manuscript/nsl23_low_resource.pdf` remains the Version 1 exploratory report on 375 clips. Its text now explicitly states that S1/S2 informed follow-on representation choices. It can support the project history, but the intended publishable headline is the pre-specified Version 2 Stage 4 endpoint—or no confirmatory headline if the annotation coverage gate fails.

Detailed protocol: [`VERSION2.md`](VERSION2.md). Historical V1 rationale: [`PAPER_DIRECTION_V1.md`](PAPER_DIRECTION_V1.md). Frozen V1 experiment: [`PROTOCOL.md`](PROTOCOL.md).

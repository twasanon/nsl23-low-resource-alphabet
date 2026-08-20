# NSL23 Version 2 protocol

Status: pre-annotation gate, 20 August 2026. Version 1 is frozen. **Do not train Version 2.** Dashboard: [`PAPER_DIRECTION.md`](PAPER_DIRECTION.md). Human instructions: [`annotations/v2/ANNOTATION_GUIDE.md`](annotations/v2/ANNOTATION_GUIDE.md).

The current manuscript is a Version 1 exploratory report. Its 59.4% result is development evidence because S1/S2 informed follow-on representation choices. The Version 2 headline is the pre-specified Stage 4 endpoint, or there is no confirmatory 49-way headline.

## Exact task

Stage 4 measures:

> 49-way classification given an isolated, manually bounded gesture from an unseen signer.

It does not measure automatic spotting, continuous recognition, sentence recognition, or live webcam accuracy. Webcam FPS is an engineering measurement performed only after Stage 4.

NSL23 reports 1,205 gestures represented inside 630 videos. That is not 1,205 independently labeled isolated clips. Version 1 used 375 audited original isolated clips. The aggregate tapes have no official temporal labels.

## Human annotation gate

The 691 motion proposals are navigation aids, not labels. The expected 572 gestures are a ceiling, not a target that must be forced.

- Pass 1 is performed by a named NSL-knowledgeable reviewer. It may show a non-authoritative tape-order hint.
- Pass 2 starts from the original proposal. It hides pass-1 decision, class, bounds, and the tape-order class hint.
- Each accepted pass stores reviewer identity, UTC timestamp, canonical class, and bounds in seconds and frames.
- A segment is included only when both passes accept the same class with temporal IoU ≥ 0.5, or after recorded adjudication.
- Every proposal requires two different named reviewers; this includes development and rejected proposals.
- Adjudication may resolve a completed-pass decision, class, or bounds disagreement, but it cannot replace a missing pass.
- Freeze requires two completed passes on every proposal, canonical class/sign-type agreement or recorded adjudication, valid timestamps, real source-file SHA-256, positive bounds, frame/time consistency, bounds inside duration, and no overlap.
- Pre-adjudication decision agreement, class agreement, and temporal IoU are saved and reported.

Boundary rule: start at the first intentional movement into the sign; include the complete handshape or dynamic motion; end after its final intentional hold or motion and before transition to the next sign.

The operator should not label clips unless they can identify the NSL alphabet. Recruit qualified reviewers rather than guessing from the expected order.

Both reviewers sweep the full source tape for missed detector proposals. A pass-1 manual insertion is class-hidden and second-verified in pass 2, remains flagged as `manual_insert`, and is not claimed as independently discovered twice.

## Locked feature and training specification

Each frozen manual window is sampled at 16 uniform temporal bin centers and decoded at 224 pixels. MediaPipe Hands 0.10.14 runs in static-image mode with complexity 1, at most two hands, and 0.35 detection/tracking thresholds. The selected image and world landmark lists remain index-paired. Frozen source hashes, protocol hashes, and frame counts are embedded in every V2 `.npz`; legacy `.npy` caches are rejected.

Every geometry trial uses the same one-layer, 64-unit-per-direction BiLSTM, layer normalization and temporal mean pooling. Prototypical training is locked to AdamW (learning rate 3e-4, weight decay 1e-4), 40 epochs, 40 episodes per epoch, 5 support and 2 query examples per class, and deterministic seeds 17/42/73. These values are checked by the training entry point, the frozen model specification, the checkpoint, and the Stage 4 evaluator.

## Frozen test assignment

The sealed candidate set is S5, S14, S4, and S6. It was declared before annotation outcomes.

Every candidate with 49 verified classes becomes a frozen tester. Candidate signers are never released into development, even when incomplete. This preserves every available unseen signer and prevents a post-coverage decision to trade test signers for training data.

Stage 4 requires at least two complete frozen testers. If fewer than two are complete, freeze and publish the annotation resource but do not make a confirmatory 49-way claim.

S1/S2 are development. S3 is validation if complete. If S3 is incomplete, Stage 3 is skipped and its verified clips may augment training. Other incomplete signers may contribute verified or original isolated clips only when they are outside the sealed test hierarchy. This rule was locked before annotation outcomes.

## Stage 2: data-only diagnostic

Model: the Version 1 prototypical BiLSTM using wrist-centered image xyz plus 20 joint angles.

For each S1/S2 fold, the held-out signer is excluded from every training source. Validation and all test candidates are excluded. Evaluation uses only the original isolated clips of that held-out signer.

This remains development evidence and never opens Stage 4 labels.

## Stage 3: three preregistered geometry trials

Run only when complete S3 is frozen as validation. Every trial uses the same encoder, prototypical loss, 40 epochs, seeds 17/42/73, and LOSO folds S1/S2/S3.

1. `image_xyz_angle`: wrist-centered, palm-scale-normalized image xyz plus 20 joint angles and detection flag.
2. `world_xyz`: MediaPipe world coordinates, wrist-centered and scaled by wrist-to-middle-MCP distance, plus detection flag.
3. `palm_canonical_world_xyz`: `world_xyz` rotated into semantic palm axes: wrist→middle MCP is +y, pinky MCP→index MCP is orthogonalized +x, and +z completes a right-handed frame.

Selection metric: mean across seeds and the three held-out signers of per-signer 49-class macro-recall using every evaluation clip. Tie-break: mean clip-pooled top-1, then prefer the control.

If S3 is incomplete, skip Stage 3 and select `image_xyz_angle`. Poudel pretraining, ST-GCN, metric learning, RGB fusion, top-k, and enrollment are outside this budget.

## Model-spec freeze and final retraining

After Stage 3, or its preregistered skip, freeze `configs/v2_model_spec.json`. That file binds the chosen feature representation, 40 epochs, model settings, seeds, Stage 3 summary hash, and frozen protocol hash.

Final retraining is train-only. It uses all permitted non-test data, including S3, for seeds 17, 42, and 73. The final manifest contains no test rows, and the training script refuses any test row. Checkpoints store the protocol, model-spec, training-manifest, and segment hashes.

## Stage 4 endpoint

Stage 4 is evaluation-only. It loads the three frozen checkpoints and cannot train.

For each seed:

1. Use every verified clip of each frozen tester.
2. Within each signer and class, calculate recall across that class's clips.
3. Average the 49 classes equally to obtain per-signer macro-recall.
4. Average frozen signers equally to obtain the seed's primary endpoint.

Report the mean and sample SD of the primary endpoint across seeds 17/42/73.

The seed SD describes optimization variation only. It is not a confidence interval for a population of signers.

Required companions:

- Each tester's macro-recall and clip-pooled top-1, mean ± sample SD across seeds.
- Overall clip-pooled top-1 as a secondary endpoint.
- Per-seed results and predictions.
- The number of clips contributing to each tester.

Signer and recording condition are confounded. Prepared/unprepared differences are descriptive, never causal. Even with multiple frozen testers, this remains a small fixed-cohort evaluation and does not estimate a population of signers.

## Executable route

### 1. Calibration and real annotation

```bash
python scripts/make_v2_review_pilot.py
python scripts/review_aggregates.py --mode pass1 --reviewer "REVIEWER A" --segments annotations/v2/pilot_segments.csv --port 8765
python scripts/review_aggregates.py --mode pass2 --reviewer "REVIEWER B" --segments annotations/v2/pilot_segments.csv --port 8765

python scripts/review_aggregates.py --mode pass1 --reviewer "REVIEWER A" --port 8765
python scripts/review_aggregates.py --mode pass2 --reviewer "REVIEWER B" --port 8765
python scripts/report_v2_annotation_agreement.py --output annotations/v2/agreement_pre_adjudication.json
python scripts/review_aggregates.py --mode adjudicate --reviewer "ADJUDICATOR" --port 8765
python scripts/report_v2_coverage.py
python scripts/freeze_v2_protocol.py
```

### 2. Frozen manifests and features

```bash
python scripts/prepare_v2_manifests.py
python scripts/extract_v2_landmarks.py --manifests data/manifests/v2/*.csv
```

Every generated manifest embeds the protocol, annotation-agreement, source-inventory, and artifact hashes. Cache files embed the protocol and source-video hashes.

### 3. Development stages

```bash
python scripts/run_v2_stage2.py
```

If S3 is complete:

```bash
python scripts/run_v2_stage3.py
python scripts/freeze_v2_model_spec.py --stage3-summary artifacts/v2/stage3_summary.json
```

If S3 is incomplete:

```bash
python scripts/freeze_v2_model_spec.py --skip-stage3
```

### 4. Final train-only batch and one Stage 4 evaluation

```bash
python scripts/run_v2_final_training.py

python scripts/evaluate_v2_stage4.py \
  --samples data/manifests/v2/stage4_eval.csv \
  --final-train-manifest data/manifests/v2/final_retrain.csv \
  --checkpoints \
    artifacts/v2/final/seed_17/best_checkpoint.pt \
    artifacts/v2/final/seed_42/best_checkpoint.pt \
    artifacts/v2/final/seed_73/best_checkpoint.pt \
  --output-dir artifacts/v2/stage4
```

The evaluator refuses an existing Stage 4 output directory and writes a started/completed receipt. No rerun should be used to adapt the model.

## Fail-closed guarantees

- `configs/signer_split.json` is obsolete for V2.
- V2 manifests require a frozen protocol and matching segment, coverage, source-manifest, class, and protocol hashes.
- Freeze requires every proposal to complete both independent reviews and binds the pre-adjudication agreement report and source-video inventory by SHA-256.
- The training gate is mandatory whenever V2 provenance fields are present.
- Every test-candidate signer is excluded from training and validation.
- Stage 3 uses explicit LOSO test folds; validation rows are never silently ignored.
- Final retraining accepts train rows only.
- Stage 4 accepts test rows only and frozen train-only checkpoints only.
- Required 49-class coverage is checked per evaluation signer.

<!-- V2_FREEZE_HASHES -->
## Freeze hashes

Not frozen. The hierarchy lock below was created before annotation outcomes. Artifact hashes are populated by `freeze_v2_protocol.py`.

| Artifact | SHA-256 |
|---|---|
| hierarchy lock | `09a2b7b2cb3a580c6cf3e3cd0c952a04ec4a793863ab27212c0cdeb55c5ab009` |
| annotations/v2/verified_segments_v2.0.csv | *(not frozen)* |
| annotations/v2/coverage_v2.0.csv | *(not frozen)* |
| annotations/v2/agreement_pre_adjudication.json | *(not frozen)* |
| data/manifests/v2_source_inventory_v2.0.csv | *(not frozen)* |
| data/manifests/nsl23_manifest_all.csv | *(not frozen)* |
| configs/nsl23_classes.json | *(not frozen)* |
| configs/v2_protocol.json canonical payload (`protocol_sha256=null`) | *(not frozen)* |
<!-- /V2_FREEZE_HASHES -->

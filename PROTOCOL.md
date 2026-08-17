# NSL23 Primary Benchmark Protocol

Status: frozen before primary model training on 2026-08-15.

## Research question

On original, individually labelled NSL23 alphabet clips, does temporal aggregation with a BiLSTM or Transformer improve unseen-signer classification over a middle-frame MobileNetV2 baseline when the spatial encoder, input sampling, training schedule, and evaluation labels are held constant?

## Dataset provenance and audit

- Source: NSL23 Zenodo version 2, DOI `10.5281/zenodo.10478554`, CC BY 4.0.
- All four archives and the supplied dataset-detail document are verified against the byte counts and MD5 checksums in the Zenodo record.
- The release contains 630 decodable videos representing 1,205 claimed gesture instances: 607 individually labelled video files and 23 aggregate recordings.
- The lossless manifest retains every source file and its original filename, signer, source archive, capture attributes, video metadata, decoding status, canonical label where applicable, and explicit exclusion reason.
- The primary sample set contains 375 original individually labelled clips from S1, S2, S7, S8, S10, S11, and S12. It excludes 234 cropped or preprocessed files and all 23 aggregate recordings.
- Aggregate recordings are excluded because the release supplies no per-gesture timestamps. Visual review of ordered equal-duration candidate bins showed cumulative timing drift and, in at least one source, an apparently incomplete or shifted sequence. Equal-duration slicing is therefore not treated as ground truth.

## Class ontology

The fixed ontology has 49 labels: 36 consonants and 13 vowels. Filename aliases are signer-aware where the release uses two transliteration schemes. No unknown class is added. No test row is discarded because its class is absent from training. Each primary fold is asserted to contain all 49 labels in both train and test partitions.

## Signer-held-out design

Two pre-specified folds are used:

1. `heldout_s1`: train on S2, S7, S8, S10, S11, and S12; test on S1.
2. `heldout_s2`: train on S1, S7, S8, S10, S11, and S12; test on S2.

Each fold has 277 training clips and 98 test clips. Each held-out signer contributes one original bright/normal and one original dark clip for every class, so each test fold contains 49 bright and 49 dark clips. There is no validation set and no checkpoint selection. Hyperparameters are common to all models, training lasts exactly 30 epochs, and only the final checkpoint is evaluated.

The two held-out signers are the only release participants with original individually labelled clips for all 49 classes. Other individually labelled signers contribute consonants only. Consequently, the design supports strict unseen-signer testing over the complete alphabet without assigning fabricated timestamps to aggregate videos, but its signer-level external validity is limited to two held-out participants.

## Input pipeline

- Sixteen temporal-bin-center frames are sampled from each clip.
- Frames are resized with aspect ratio preserved and padded to 224 × 224 with black pixels; no geometric stretch is used.
- RGB values are normalized with ImageNet mean and standard deviation.
- Training-only brightness and contrast perturbations are sampled once per clip and applied identically to all 16 frames.
- Horizontal flipping is disabled because it can alter sign semantics.
- The frame cache stores `uint8` arrays derived from the original clips and is keyed by deterministic sample IDs.

## Models and fairness controls

All three models use the same ImageNet-1K V2 MobileNetV2 frame encoder. The encoder and its batch-normalization state are frozen in every model.

- Single-frame CNN: classifies the middle sampled frame with dropout 0.2.
- CNN–BiLSTM: projects 1,280-dimensional frame features to 256 dimensions, uses a one-layer bidirectional LSTM with 128 hidden units per direction, mean-pools time, and applies dropout 0.2.
- CNN–Transformer: projects frame features to 256 dimensions, adds learned positional embeddings, applies two Transformer encoder layers with four heads and a 512-dimensional feed-forward block, mean-pools time, and applies head dropout 0.2. Encoder attention dropout is zero for deterministic MPS-compatible scaled-dot-product attention.

The models have 62,769, 736,305, and 1,399,345 trainable parameters, respectively. Parameter counts are reported so temporal gains are not presented as architecture-neutral.

## Optimization

- Seeds: 17, 42, 73.
- PyTorch deterministic algorithms are enabled; cuDNN benchmarking is disabled, and the CUDA workspace setting is fixed when CUDA is used.
- Batch size: 8.
- Epochs: 30, fixed.
- Loss: inverse-frequency class-weighted cross entropy without label smoothing. Weights are normalized to mean one within each training fold; this compensates for the release's two training clips per vowel versus up to seven per consonant.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Schedule: cosine annealing over all 30 epochs.
- Test policy: the final checkpoint is evaluated once per fold/model/seed after the protocol is frozen.

## Outcomes

Primary descriptive outcomes are top-1 accuracy and macro F1. Macro precision, macro recall, weighted F1, fixed-order per-class metrics, and both count and row-normalized confusion matrices are also retained. Metrics always use the explicit 49-class order.

Results are first reported for each held-out signer and seed. Out-of-fold predictions are pooled across S1 and S2 only after both folds complete. Mean and sample standard deviation across seeds are reported. Stratified clip bootstrap intervals are supplied as within-sample uncertainty summaries; they do not overcome the limitation of having two held-out signers.

Bright/normal versus dark performance is a pre-specified descriptive breakdown with 49 clips per condition per fold. It is not a causal illumination experiment because condition and recording details are not independently randomized.

## Runtime measurement

Runtime is measured on the local Apple M4 MPS device with batch size one after warm-up and explicit device synchronization. Model-only latency and end-to-end decode/preprocess/transfer/inference latency are reported separately as median and 95th percentile milliseconds per clip. Throughput is reported as clips per second. The study does not claim streaming real-time operation unless a separately implemented sliding-window benchmark supports that claim.

## Landmark addendum (2026-08-17)

Status: frozen after the RGB matrix. Same 375 clips, same S1/S2 folds, same seeds, same 30-epoch final-checkpoint rule.

- Detector: MediaPipe Hands 0.10.14 on the cached 16 RGB frames (`static_image_mode=True`).
- Features: 21 xyz landmarks, wrist-subtracted, divided by the middle-MCP xy length, plus a detection flag (64-D / frame). Misses are zeros. No clip in the primary set had zero detections (5,868 / 6,000 frames detected).
- Models: `landmark_mlp` (301,617 parameters) and `landmark_bilstm` (73,137 parameters). Training noise std 0.02 on spatial coordinates only.
- Device: CPU.
- Webcam: `scripts/webcam_demo.py` uses video-mode Hands and a 16-frame deque. Streaming FPS is wall-clock frames processed, not RGB clip throughput.

## Improvement addendum (2026-08-17)

Same 375 clips and S1/S2 folds. Does not replace the frozen xyz 30-run matrix.

- Features: `raw_angle` concatenates 63 xyz, 20 inter-joint angles (15 finger + 5 wrist span), and the detection flag (84-D / frame).
- Cross-entropy: `landmark_mlp` and `landmark_bilstm` with `configs/experiment_improvements.json`, 30 fixed epochs.
- Prototypical: `scripts/train_protonet.py`, 40 epochs, 40 episodes/epoch, 5-shot/2-query with replacement, nearest **training** prototype at test time. xyz ablation uses `--feature-mode xyz`.
- Crops: MediaPipe boxes on the RGB cache → `data/hand_crop_cache_primary/` (not in git). Frozen MobileNetV2 middle-frame CNN.
- Enrollment: `enroll_closed_other_lighting` is the 49-way 1-shot protocol. Do not headline `enroll_replace_class`.
- Webcam default: prototypical xyz+angle checkpoint, nearest class mean.


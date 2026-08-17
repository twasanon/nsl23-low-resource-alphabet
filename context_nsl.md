# NSL23 CNN–Transformer Research Paper: Complete Project Handoff Context

**Purpose of this document:**  
This is the authoritative handoff for the next AI agent that will take the Nepali Sign Language (NSL) recognition project from its current draft state to a **scientifically defensible, reproducible, publication-ready paper** for the **National College of Computer Studies Research Journal (NCCSRJ), Nepal**.

The next agent should not treat the current paper or notebook as finished. Both contain useful work, but there are important implementation, dataset, methodology, evaluation, formatting, and claim-validation issues that must be resolved before submission.

The goal is not merely to make the paper look polished. The goal is to ensure that **every claim in the manuscript is backed by an experiment, every equation matches the actual implementation, every table/figure is generated from real results, the notebook runs end-to-end in Google Colab, the dataset is handled correctly, the references are accurate, and the final manuscript follows the target journal’s actual submission rules.**

---

# 1. High-Level Project Goal

The research project currently has the working title:

> **Bridging the Silent Gap: Real-Time Nepali Sign Language Recognition Using a Hybrid CNN–Transformer Architecture**

The central research idea is:

- Use the public **NSL23 Nepali Sign Language video dataset**.
- Recognize isolated NSL alphabet signs from video.
- Use a **MobileNetV2 CNN** to extract spatial information from individual frames.
- Use a **Transformer encoder** to model temporal information across the sampled video frames.
- Compare the proposed CNN–Transformer against meaningful baselines.
- Use signer-independent evaluation so the system is tested on people who were not used for training.
- Analyze robustness under different recording conditions.
- Measure actual inference performance on a **Google Colab NVIDIA T4 GPU**.
- Provide the full reproducible implementation through GitHub rather than printing source code in the manuscript.

The paper is intended to be an **application-oriented computer vision / machine learning paper** with Nepal/NSL relevance.

The paper must **not** claim success before the experiments are actually run.

---

# 2. Current Files / Artifacts

The important working files created during this project are:

## Paper

Current latest working draft:

- `nsl_cnn_transformer_ieee_FINAL.tex`
- `nsl_cnn_transformer_ieee_FINAL.pdf`

The LaTeX draft already contains:

- title and abstract
- introduction
- related work
- dataset discussion
- robust indexing description
- CNN–Transformer methodology
- mathematical formulation
- training / preprocessing discussion
- evaluation metrics
- signer-independent evaluation discussion
- runtime discussion
- Tables II / III placeholders
- code availability section
- bibliography

However, **this is still a draft and should not be submitted as-is**.

## Notebook provided by the user

The notebook currently attached for the next phase is:

- `NSL23_CNN_Transformer_Colab_Benchmarks_NSLS23Robust.ipynb`

This notebook should be treated as the **starting implementation**, not as known-good code.

An earlier generated version also existed:

- `NSL23_CNN_Transformer_Colab_Benchmarks_RobustAndConditions.ipynb`

That version contained basic condition-wise evaluation logic, but the attached `NSLS23Robust` notebook should remain the main reference because it contains the newer robust indexing changes. If the other notebook is available, condition-analysis logic can be inspected and merged, but it must be verified rather than copied blindly.

## Dataset reference

Zenodo record:

- `https://zenodo.org/records/10478554`
- DOI: `10.5281/zenodo.10478554`

Primary dataset paper:

- J. Sunuwar, S. Borah, and A. Kharga, **“NSL23 dataset for alphabets of Nepali sign language,”** *Data in Brief*, vol. 53, 110080, 2024.
- DOI: `10.1016/j.dib.2024.110080`
- Public full text is available through PubMed Central.

A dataset-detail Word file was also previously provided:

- `Details Dataset.docx`

---

# 3. CRITICAL: Target Journal Requirements

Before editing the manuscript, verify the journal guidelines again online because submission rules can change.

As verified on **August 16, 2026**, the official NCCSRJ submission page states the following:

Journal submission information:

`https://www.nepjol.info/index.php/nccsrj/about/submissions`

The journal currently requires:

- manuscript in **Microsoft Word**
- English
- approximately **3,000–5,000 words**
- **Times New Roman, 12 pt**
- **double spaced**
- figures and tables placed in the text
- **APA reference format**
- DOI included where available
- title, abstract, keywords, introduction/background, methodology, results/discussion, conclusion, references, tables and figures
- normally no more than 5,000 words
- approximately **5–7 keywords**
- a separate cover page containing author details
- the author’s name should not appear in the anonymized article body

## Important conflict with the current draft

The current paper is:

- IEEEtran LaTeX
- two-column IEEE style
- numbered IEEE citations such as `[1]`
- currently contains the author block directly in the manuscript

That is **not the target journal’s stated submission format**.

The user originally requested IEEE style because it is common in computer science, but the official journal requirements must take priority for final submission.

### Required final strategy

It is fine to continue using LaTeX as an internal working source while experiments are being completed.

Before submission, create:

1. **A journal-formatted Word manuscript**:
   - Times New Roman 12
   - double spaced
   - APA references
   - 3,000–5,000 words
   - author information removed from blinded manuscript

2. **A separate cover/title page**:
   - title
   - author name
   - affiliation
   - abstract
   - keywords
   - any other information required by the journal

3. A PDF proof may also be generated for visual review, but the official submission file should follow the journal requirement.

### Blind-review / GitHub concern

The current paper plans to link a public GitHub repository. A personal GitHub account may reveal author identity. Before submission:

- verify whether the journal expects the code link to be anonymized during review;
- if needed, provide an anonymized repository/supplementary archive during review and replace it with the permanent public GitHub link after acceptance.

Do not accidentally defeat manuscript anonymization through a personal repository URL.

---

# 4. Current Scientific Hypothesis

The proposed scientific hypothesis should be framed approximately as:

> A lightweight CNN–Transformer video classifier can improve signer-independent NSL alphabet recognition over a static single-frame CNN and a recurrent CNN–LSTM/BiLSTM baseline while retaining practical inference throughput.

This hypothesis requires **direct evidence**.

The most defensible contribution is not “Transformers are novel.” CNN–Transformer hybrids are already established in vision.

The novelty should instead be framed around:

- application to **NSL23**
- a reproducible video-classification benchmark
- signer-independent evaluation
- comparison against temporal and non-temporal baselines
- detailed analysis of NSL23 environmental conditions
- practical deployment-oriented runtime evaluation
- potentially the first carefully documented CNN–Transformer benchmark for this exact NSL23 task/protocol, **if literature review verifies that statement**

Never call the architecture globally novel simply because it combines CNNs and Transformers.

---

# 5. DO NOT TRUST THE OLD 93.5% RESULT

Earlier drafts of this project mentioned:

> “93.5% accuracy”  
> “state-of-the-art performance”

Those numbers were conceptual / placeholder values and **were not produced by a completed NSL23 experiment**.

The current final LaTeX draft correctly uses blank result placeholders instead of 93.5%.

The next agent must follow this rule:

> **Never reintroduce 93.5%, “state-of-the-art,” or any other performance number unless it is produced by the actual final experiment and stored in a reproducible results artifact.**

Even if the CNN–Transformer outperforms the internal baselines, it should normally be written as:

> “The proposed model outperformed the evaluated baselines under our signer-independent NSL23 protocol.”

Do not use “state-of-the-art” unless a literature review establishes comparable published results on the same dataset, task, and evaluation protocol.

Results from other NSL datasets cannot be treated as directly comparable SOTA evidence.

---

# 6. NSL23 Dataset: Facts That Matter

Verified dataset facts:

- 630 total videos
- 1,205 total gesture instances represented within those videos
- 516 consonant videos representing 971 consonant gestures
- 114 vowel videos representing 234 vowel gestures
- 36 consonant classes
- 13 vowel classes
- 49 alphabet classes overall
- 14 volunteers (`S1`–`S14`)
- 5 experienced/native NSL users and 9 beginners
- video files are primarily `.mov`
- different:
  - resolutions
  - frame rates
  - durations
  - lighting conditions
  - backgrounds
- prepared environment = controlled black background
- unprepared environment = uncontrolled background
- real-world indoor/outdoor cases exist
- bright/dark conditions are also represented
- consonants are described as **static gestures**
- vowel set contains **static and dynamic gestures**
- some videos contain a single sign
- some experienced signers performed all 36 consonants or all 13 vowels in one continuous video
- aggregate videos can range from roughly 14 seconds to several minutes
- individual character clips are generally around 1–5 seconds
- some cropped/preprocessed duplicates are included

## Dataset location nuance

NSL is directly relevant to Nepal and Nepali-speaking communities.

However, the NSL23 dataset itself was collected through **Sikkim Manipal Institute of Technology in Sikkim, India**, not in Nepal.

Therefore the paper must not say that the experimental videos were collected “in Nepal.”

Acceptable wording:

> “The dataset represents NSL under diverse indoor/outdoor and prepared/unprepared recording conditions relevant to practical deployment.”

Avoid unsupported wording such as:

> “The model was validated in real Nepal environments.”

The social/application motivation is Nepal-specific; the data collection location is not Nepal.

---

# 7. Fundamental Dataset Problem the Next Agent Must Resolve FIRST

This is the most important scientific issue in the entire project.

**1,205 gestures does NOT mean 1,205 separately labeled one-sign video files.**

The dataset has **630 videos**, and several videos contain an entire sequence of all consonants or vowels.

The current model is an **isolated-sign classifier** that assumes:

> one input video clip → one class label

Therefore long “all consonants” / “all vowels” videos cannot simply be assigned one class.

The current robust notebook filters long aggregate videos out.

That creates a serious consequence:

- a significant fraction of the 1,205 gestures disappears from the training/evaluation dataset;
- several experienced signers supplied their signs mainly in aggregate videos;
- the effective signer count and condition coverage can shrink substantially;
- the experiment may no longer represent the full 14-signer / 49-class NSL23 dataset.

## This must be audited empirically after download

Create a machine-readable manifest of **every video** before any training.

Recommended columns:

- `source_path`
- `source_zip`
- `signer`
- `sign_type` (`consonant` / `vowel`)
- `raw_filename`
- `canonical_label`
- `is_aggregate`
- `is_cropped`
- `background_condition`
- `realworld_flag`
- `lighting`
- `indoor_outdoor`
- `fps`
- `frame_count`
- `duration_seconds`
- `width`
- `height`
- `usable_as_isolated_clip`
- `reason_excluded`

Save:

- `nsl23_manifest_all.csv`

Then produce counts:

- total videos
- isolated videos
- aggregate videos
- videos by signer
- videos by label
- videos by condition
- videos by cropped/original
- exact number of usable isolated clips after filtering
- exact number of signers represented after filtering
- exact number of classes represented after filtering

These counts must become the numbers reported in the final paper.

## Do not claim 14-signer signer-independent recognition unless the training data really contains usable labeled clips from 14 signers.

---

# 8. Aggregate Videos: Required Decision

After auditing the dataset, choose one scientifically valid approach.

## Preferred approach if feasible: segment aggregate recordings

If aggregate videos contain all alphabet signs in a known order:

1. determine whether trustworthy gesture-boundary annotations exist;
2. if not, create gesture boundaries;
3. save the derived annotations in a file such as:

`segments.csv`

with:

- source video
- signer
- class label
- start frame/time
- end frame/time
- condition
- reviewer/verification status

### Important

Do **not** divide every aggregate video into 36 or 13 equal-duration blocks and assume those blocks are correct labels unless manual verification proves that the signing timing is sufficiently uniform.

A publishable dataset transformation requires validated segment boundaries.

A reasonable workflow is:

- detect candidate movement/pause boundaries using motion energy;
- use canonical alphabet order as a guide only if the dataset documentation explicitly confirms the order;
- visually verify every segment;
- save final annotations in the GitHub repository.

This may itself become a useful reproducibility contribution.

## If reliable segmentation is not feasible

Then explicitly define the paper as using an **isolated-clip subset of NSL23**.

Report:

- exact number of clips
- exact number of classes
- exact number of signers
- exact conditions

Do not continue to describe results as if all 1,205 gestures were used.

### Scientific tradeoff

If the isolated subset cannot support a credible signer-independent 49-class experiment, the research scope may need to change.

Do not hide this limitation.

---

# 9. CRITICAL Cropped-Video Issue

The current notebook has a function that says:

> if cropped videos exist, use cropped videos exclusively.

This is problematic.

The dataset paper states that cropped/preprocessed repetitions were created only for a subset of volunteers, particularly `S1`, `S2`, and `S3`.

If the notebook globally selects only files with `"cropped"` in their path, the effective dataset may collapse to only a few signers.

That would destroy the intended signer-independent experiment.

## Required fix

Do **not** use global `prefer_cropped=True` for the primary experiment.

Recommended primary policy:

- use original/raw isolated clips/segments;
- avoid duplicated cropped versions in the primary dataset;
- keep cropped versions only for an optional ablation or separate experiment.

If cropping is desired, perform the **same preprocessing pipeline** consistently on every sample rather than selectively using pre-cropped duplicates from only a subset of signers.

---

# 10. Environment / Condition Taxonomy Must Be Correct

The paper currently talks about:

- Prepared
- Unprepared
- Real-world

These should be handled carefully.

According to the dataset description:

- **Prepared** means controlled black background.
- **Unprepared** means no black background / uncontrolled background.
- **RealWorld** scenes are effectively a specific kind of unprepared indoor/outdoor capture.
- Bright and Dark are **lighting attributes**, not necessarily independent dataset classes.

Therefore “Prepared / Unprepared / Real-world” can overlap conceptually.

If Table III needs three mutually exclusive rows, define them explicitly:

1. **Prepared**
2. **Indoor Unprepared (non-RealWorld)**
3. **RealWorld Unprepared**

Do not double-count RealWorld examples in both Unprepared and RealWorld rows.

Prefer storing independent metadata fields:

- `background_condition`: prepared / unprepared
- `scene_type`: normal_indoor / realworld_indoor / realworld_outdoor
- `lighting`: bright / dark / normal / natural

Then define the table grouping from those fields.

## Condition confounding warning

NSL23 conditions are not evenly distributed across signers.

Certain signers appear only in certain conditions.

Therefore a condition-wise comparison on a signer-independent test set may also reflect differences between signers.

The paper must not imply a causal “background effect” unless the comparison is properly controlled.

Condition Table III should be described as **descriptive robustness analysis**, with:

- number of test clips per condition
- number of classes per condition
- signer IDs contributing to each condition

If the held-out signer split does not contain enough examples in each condition, do not fabricate the table. Redesign the split or weaken the analysis.

---

# 11. Current Attached Notebook: Static Audit

Notebook:

`NSL23_CNN_Transformer_Colab_Benchmarks_NSLS23Robust.ipynb`

It contains approximately:

- environment imports
- Zenodo download
- robust indexing functions
- signer-aware split
- video loader
- MobileNetV2 + Transformer
- training
- test evaluation
- single-video inference
- sample-grid generation
- baseline models
- benchmark suite

The notebook’s Python code is syntactically parsable after ignoring Colab shell commands.

However, it is **NOT runnable end-to-end in its current state**.

## Blocking bug 1: vowels are not downloaded

Current cell:

```python
DOWNLOAD_WHICH = ["consonant_p1", "consonant_p2", "consonant_p3"]
```

It does not include:

```python
"vowel"
```

Therefore the notebook cannot currently train the claimed 49-class consonant + vowel model.

Required:

```python
DOWNLOAD_WHICH = [
    "consonant_p1",
    "consonant_p2",
    "consonant_p3",
    "vowel",
]
```

for the final full experiment.

For a smoke test a subset is fine, but the final result must explicitly use the intended dataset.

## Blocking bug 2: `items` is used before being created

The robust indexing cell defines:

```python
collect_videos(...)
```

but in the attached notebook it does not subsequently execute:

```python
items = collect_videos(...)
```

The next cell calls:

```python
show_random_samples(items, ...)
```

so the notebook will raise `NameError: items is not defined`.

This must be fixed immediately.

## Blocking methodological issue 3: cropped-only behavior

As discussed above, `prefer_cropped=True` is not acceptable as the primary strategy.

## Blocking methodological issue 4: aggregate videos are discarded

The notebook filters:

- `ALL`
- `NSL`
- `CONSONANT`
- `VOWEL`
- long videos >10 sec

This is useful for isolated clips, but the manuscript cannot continue pretending the full 1,205 gestures and 14 signers remain after filtering.

After filtering, print and save the actual dataset statistics.

## Blocking issue 5: unknown signers

`_extract_signer_from_path()` returns `UNK` if no signer ID is found.

Do not allow `UNK` samples into the main signer-independent split as one artificial “signer group.”

Instead:

- resolve the signer from the documented folder structure;
- if still unresolved, exclude from the primary experiment and log it;
- report the count.

## Blocking issue 6: silently dropping unseen labels

The notebook creates class mapping from training labels and then drops validation/test examples with labels not seen in training.

That can make the test set artificially easier and invalidate reported accuracy.

Do not silently drop legitimate test classes.

Instead:

- construct the dataset and split so every evaluated class is represented in training;
- predefine the canonical class set;
- assert coverage;
- if coverage fails, choose a valid signer split or change the dataset strategy;
- log any genuinely malformed label separately.

## Blocking issue 7: GroupShuffleSplit does not guarantee class coverage

With only 14 signers, a 10% grouped split is effectively selecting a small number of signers, not a precise 10% of clips.

The paper must report:

- exact train signer IDs
- exact validation signer IDs
- exact test signer IDs
- number of clips/classes in each split

Save the split assignment:

`split_manifest.csv`

Do not rely only on seed `42` and then describe it vaguely.

## Blocking issue 8: attached notebook does not contain the final condition-wise section

A later notebook version contained a simple `infer_condition_from_path()` implementation, but the currently attached robust notebook does not contain the full Prepared/Unprepared/RealWorld result export.

Implement condition metadata directly in the manifest rather than bolting it on at the end.

---

# 12. Notebook Evaluation Bugs / Improvements

## Classification report

Current:

```python
classification_report(
    y_true,
    y_pred,
    target_names=[idx2label[i] for i in range(NUM_CLASSES)]
)
```

This may fail or misalign if one or more classes are absent from the test subset.

Use explicit labels:

```python
labels=np.arange(NUM_CLASSES)
```

and:

```python
zero_division=0
```

## Confusion matrix

Use:

```python
confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)
```

so the class ordering is fixed and reproducible.

Store both:

- raw counts
- row-normalized confusion matrix

## Macro-F1

Be explicit about which classes are included.

For the main 49-class test, ideally every class should have support.

For condition subsets, macro-F1 can be misleading if some classes are absent. Save:

- number of classes represented
- class support
- exact `labels=` policy

Describe that policy in the paper.

---

# 13. Runtime Benchmark Is Currently Not Adequate for the “Real-Time” Claim

The current benchmark function runs a preloaded batch through the model repeatedly.

That measures **model throughput**, not true end-to-end real-time inference.

It excludes:

- video decoding
- frame selection
- resize
- normalization
- host-to-GPU transfer
- postprocessing
- streaming buffer logic

Additionally, the current `ms/frame` calculation is misleading for the single-frame baseline because it divides runtime by all `T` frames even though that model only evaluates the middle frame.

## Required runtime metrics

On Colab T4, report at least:

### A. Model-only batch-size-1 latency

- use one clip at a time
- warm GPU
- synchronize CUDA
- use CUDA events or careful `perf_counter + synchronize`
- report median and preferably p95
- units: **ms/clip**

### B. Model-only throughput

- specify batch size
- report clips/s
- optionally report CNN-frame-equivalent throughput, but label it clearly

### C. End-to-end clip inference

Measure:

1. decode / frame sampling
2. preprocessing
3. device transfer
4. model forward
5. softmax / prediction

Report end-to-end ms/clip.

### D. Streaming inference

If the title retains **“Real-Time”**, implement the sliding-window method described in the paper:

- deque/buffer of latest `T` frames
- inference every `stride` frames
- optional exponential moving average:

\[
\bar p_t = \alpha \bar p_{t-1} + (1-\alpha)p_t
\]

Test it on a continuous video stream or video-file simulation.

Do not call batched clip throughput “webcam FPS.”

If the system is not demonstrated to operate fast enough end-to-end, remove or soften **“Real-Time”** from the title and claims.

---

# 14. Current Model Architecture

The current implementation is:

## Frame encoder

- ImageNet-pretrained MobileNetV2
- classifier removed
- convolutional features
- global average pooling
- output dimension 1280

## Projection

- linear projection:
  - 1280 → 512

## Temporal encoder

- Transformer encoder
- `d_model = 512`
- `nhead = 8`
- 4 encoder layers
- feed-forward dimension = 2048
- dropout = 0.1
- GELU activation
- batch-first PyTorch encoder

## Pooling

- mean pool across time

## Classification

- linear layer:
  - 512 → `NUM_CLASSES`

## Input

- `NUM_FRAMES = 16`
- RGB
- 224 × 224
- ImageNet normalization

This is a reasonable baseline architecture, but it must be validated experimentally.

---

# 15. IMPORTANT Math / Implementation Mismatch in Current Paper

The paper currently presents **sinusoidal positional encoding**:

\[
p_{i,2k} = \sin(...)
\]

\[
p_{i,2k+1} = \cos(...)
\]

However, the notebook actually uses a **learnable positional embedding**:

```python
self.pos = nn.Parameter(torch.zeros(max_len, d_model))
nn.init.trunc_normal_(self.pos, std=0.02)
...
f = f + self.pos[:T].unsqueeze(0)
```

This is a direct paper/code inconsistency.

## Required decision

Either:

### Option A — Recommended

Keep the notebook as-is and rewrite the paper to state:

> “A learned positional embedding is added to each frame representation.”

Then remove the sinusoidal formula.

or:

### Option B

Change the notebook to use sinusoidal encoding exactly as written in the paper.

Do not leave the two different.

The next agent must audit **every equation against the implementation**.

---

# 16. Other Math / Methodology Items to Audit

## Attention

Current scaled dot-product attention is standard:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
\right)V
\]

Define explicitly:

\[
d_h = d/H
\]

for `H` attention heads.

## Transformer block

The current paper’s residual + layer-normalization formulation is compatible with PyTorch’s default `TransformerEncoderLayer(norm_first=False)`.

Verify this remains true if the notebook is changed.

## Cross-entropy

Current loss formula is standard.

## Label smoothing

The paper currently says label smoothing is optional and includes a formula.

The notebook actually uses:

```python
nn.CrossEntropyLoss()
```

with no label smoothing.

For exact reproducibility:

- remove label smoothing from the final paper, or
- implement it and report the exact epsilon.

Do not describe unused training techniques.

## Complexity

Current paper writes roughly:

\[
O(L(T^2d + Td^2))
\]

which is a reasonable high-level Transformer complexity expression.

However, the prose currently says small `T` and “moderate d (128–256),” while the actual implementation uses:

- `T = 16`
- `d = 512`

Correct this mismatch.

## Streaming smoothing

The paper describes probability smoothing, but the attached notebook does not implement the full streaming system.

Either implement it and test it or remove the claim.

---

# 17. Preprocessing Audit

Current pipeline:

- OpenCV
- uniform frame sampling
- 16 frames
- BGR → RGB
- resize 224 × 224
- `[0,1]`
- ImageNet normalization
- sequence-consistent brightness jitter
- sequence-consistent contrast jitter
- no horizontal flip

The decision not to use horizontal flip is sensible because mirroring can change sign semantics.

## Potential improvement to evaluate

Directly resizing arbitrary video resolution to 224 × 224 may distort hand geometry.

Consider:

- aspect-ratio-preserving resize + padding
- or resize short edge + center crop

If this changes the implementation, update the paper.

## Video decoding

OpenCV random seeking can be inefficient or unreliable for `.mov`.

During actual Colab testing:

- verify every file decodes correctly;
- log failures;
- consider PyAV or Decord if OpenCV is too slow/unreliable;
- never silently replace many failed frames with zeros.

---

# 18. Recommended Experiment Protocol

All three main models must be evaluated on the **same data and identical signer split**.

Primary models:

1. **Single-frame CNN**
   - MobileNetV2
   - middle frame only
   - no temporal model

2. **CNN + LSTM/BiLSTM**
   - MobileNetV2 per-frame features
   - recurrent sequence model
   - current notebook uses **BiLSTM**
   - call it BiLSTM consistently if that remains the model

3. **CNN + Transformer**
   - proposed model

## Baseline naming consistency

The paper and notebook must not alternate between:

- CNN-LSTM
- CNN-BiLSTM

Choose one.

Current code uses **bidirectional LSTM**.

For isolated pre-recorded clips, BiLSTM is valid.

For strictly causal streaming, a unidirectional LSTM is a more directly causal baseline.

This decision should be explicit.

---

# 19. Fair Baseline Training

Do not accidentally make the Transformer look better by giving it more training or tuning.

Use:

- same input frames
- same split
- same MobileNetV2 initialization
- same augmentation
- same optimizer family
- same training budget
- same model-selection criterion

Model-specific hyperparameters can differ only if tuned using the validation set in a documented manner.

A reasonable shared starting configuration:

- AdamW
- LR = `1e-4`
- weight decay = `1e-4`
- batch size = `8` if memory-constrained
- cosine schedule
- 15–30 max epochs
- early stopping

However, **the final paper must report one exact configuration**, not ranges such as “batch size 8–32” or “15–30 epochs.”

The current draft table contains ranges and must be replaced with the actual final run configuration.

---

# 20. Training Reproducibility

Save:

- Python version
- PyTorch version
- torchvision version
- CUDA version
- GPU model
- exact random seed
- package versions
- experiment config JSON
- git commit hash

Use deterministic/reproducible data-loader seeding where feasible.

For paper-quality evidence, strongly consider:

- 3 independent training seeds per model on the same fixed split
- report mean ± standard deviation

If compute allows, repeated signer splits are even stronger, but they may be expensive.

At minimum, the test split must be fixed before comparing models.

---

# 21. Signer-Independent Split

Signer independence is one of the strongest elements of this project.

Do not randomly split clips across train/test.

Primary test set should contain **entire held-out signers**.

The paper must report exact signer IDs, e.g.:

- train: S...
- validation: S...
- test: S...

Do not leave this as “10% signer split” only.

## Coverage requirement

Before accepting a split:

- every evaluated class must exist in training;
- ideally every class exists in validation and test;
- if not possible, explain why and redesign.

Never silently delete unsupported test labels to improve metrics.

## Potential stronger strategy

If dataset construction produces enough per-signer examples, consider:

- GroupKFold / repeated signer-independent evaluation

and report mean performance over folds.

This would be scientifically stronger than a single random grouped split.

But do not force this if aggregate-video segmentation / class coverage makes it impossible.

---

# 22. Required Evaluation Metrics

Primary metrics:

## Top-1 accuracy

\[
\text{Accuracy}
=
\frac{\text{number correctly classified}}{\text{number test samples}}
\]

## Macro-F1

Compute per-class F1 then average equally across classes.

This is important because class/sample counts are not perfectly uniform.

Also save:

- macro precision
- macro recall
- weighted F1 (optional)
- per-class precision/recall/F1/support

## Confusion matrix

Generate:

- raw confusion matrix
- normalized confusion matrix

Use canonical class ordering.

## Optional but recommended

- 95% bootstrap confidence interval for test accuracy
- 95% bootstrap confidence interval for macro-F1
- 3-seed mean ± SD

These additions materially improve the credibility of the results.

---

# 23. Table Strategy

The expert recommendation discussed earlier is correct.

## Table II — Main Model Comparison

This is the paper’s primary result.

Rows:

- Single-frame CNN (MobileNetV2)
- CNN + BiLSTM
- CNN + Transformer (Proposed)

Recommended columns:

- Top-1 Accuracy (%)
- Macro-F1
- Parameters (M) — optional but useful
- T4 model latency (ms/clip, batch=1)
- T4 throughput (clips/s)

Do not label clips/s as camera FPS.

If actual streaming FPS is measured, add a separate column clearly labeled.

## Table III — Robustness / Condition Analysis

Rows:

- Prepared
- Indoor Unprepared
- RealWorld Unprepared

Recommended columns:

- N samples
- N classes
- Top-1 Accuracy (%)
- Macro-F1

If these groups are not all represented adequately in the held-out test signers, say so and revise the table.

## Optional Table IV — Ablation

Useful experiments:

- 8 frames
- 16 frames
- 24/32 frames

or:

- Transformer without positional embeddings
- Transformer with positional embeddings

A simple ablation helps establish that the temporal design choices actually matter.

---

# 24. Graphs / Figures That Must Be Generated from Real Data

The final paper currently has placeholder figures.

Replace them.

## Figure 1 — Actual model architecture

Create a clean publication-quality architecture diagram showing:

Video
→ uniform frame sampling
→ MobileNetV2 per frame
→ 1280-d feature
→ linear projection to 512
→ learned positional embeddings
→ 4-layer, 8-head Transformer
→ mean temporal pooling
→ 49-class softmax

If the final dataset has fewer than 49 classes, update the diagram.

Prefer vector output:

- PDF
- SVG

and a 300+ dpi PNG fallback.

## Figure 2 — Real NSL23 sample frames

Use actual NSL23 frames, not generic placeholders.

Show examples from:

- Prepared
- Indoor unprepared
- RealWorld
- bright/dark if helpful

Because faces may be visible, prefer:

- dataset-provided cropped views
- or hand-centered crops
- or legally/ethically appropriate licensed dataset frames

Credit NSL23.

## Figure 3 — Training curves

At minimum for proposed model:

- train loss
- validation loss
- train accuracy/F1
- validation accuracy/F1

Avoid cherry-picking.

## Figure 4 — Normalized confusion matrix

Use the final proposed model on held-out test signers.

For 49 classes, ensure labels remain legible.

Consider:

- full normalized matrix in main paper
- top confusion pairs summarized in text/table

## Figure 5 — Model comparison

Optional if Table II is sufficient.

A bar chart can show:

- accuracy
- macro-F1

for the 3 models.

## Figure 6 — Condition robustness

Optional visual companion to Table III.

Show model performance across:

- Prepared
- Indoor Unprepared
- RealWorld

Include N values in caption or table.

---

# 25. Every Figure Must Be Reproducible

The notebook should have a final section:

> **Generate Paper Artifacts**

It should write everything into a single directory such as:

`/content/paper_artifacts/`

Suggested outputs:

```text
paper_artifacts/
├── manifest_summary.csv
├── split_manifest.csv
├── classes.json
├── experiment_config.json
├── benchmark_results.csv
├── condition_results.csv
├── per_class_metrics.csv
├── confusion_matrix_counts.csv
├── confusion_matrix_normalized.csv
├── training_history_single_frame.csv
├── training_history_bilstm.csv
├── training_history_transformer.csv
├── fig_dataset_samples.png
├── fig_architecture.pdf
├── fig_training_curves.png
├── fig_confusion_matrix.png
├── fig_model_comparison.png
├── fig_condition_robustness.png
├── runtime_benchmark.csv
└── final_results.json
```

The paper should be updated from these artifacts—not from manually remembered numbers.

---

# 26. Current Paper Problems to Fix

The latest LaTeX draft has several known issues.

## Abstract mismatch

The abstract currently says:

> “The manuscript includes ... a runnable PyTorch reference implementation (model + dataset loader) in the Appendix...”

But the Appendix was changed to **Code Availability** and no longer contains source code.

Fix to something like:

> “The implementation and reproducible Colab workflow are provided through a public code repository.”

Do not say code is printed in the Appendix.

## Conclusion mismatch

Likewise, any sentence saying the paper itself contains runnable code must be changed to GitHub/repository wording.

## Placeholder figures

The dataset figure caption literally says it contains placeholders.

Replace before submission.

## Placeholder results

Tables contain `--`.

No submission until every result placeholder is replaced with real data.

## Placeholder GitHub URL

Current:

`https://github.com/<YOUR-REPO-URL>`

Replace or anonymize appropriately.

## Hyperparameter ranges

The paper currently says:

- batch size 8–32
- epochs 15–30

Final paper must state exact values used in reported results.

## Duplicate evaluation discussion

The draft has an “Evaluation Metrics” section earlier and a more complete “Experimental Setup and Evaluation” section later.

Consolidate to avoid redundancy.

## Word count

A rough static count of the current LaTeX draft is only around **2,400 words**.

The target journal currently requests approximately **3,000–5,000 words**.

The final results, richer related work, experiment details, and discussion should naturally bring the paper above the minimum.

Do not add filler merely to reach the word count.

## Keywords

Current draft has 8 keywords.

Journal guidance is approximately 5–7.

Reduce to 5–7.

Suggested:

- Nepali Sign Language
- sign language recognition
- computer vision
- Transformer
- deep learning
- video classification

---

# 27. Paper Claim Audit

Before finalizing, create a checklist of every empirical claim.

Examples:

### Claim
“The CNN–Transformer improves recognition accuracy.”

Required evidence:
Table II.

### Claim
“The model generalizes to unseen signers.”

Required evidence:
Strict signer-independent split.

### Claim
“The approach performs robustly in real-world conditions.”

Required evidence:
Table III with adequate real-world held-out data; if confounded/sparse, soften wording.

### Claim
“The system is real-time.”

Required evidence:
actual batch-1/end-to-end streaming measurements.

### Claim
“49-class NSL alphabet recognition.”

Required evidence:
manifest and test set actually contain all 49 classes.

### Claim
“14 volunteers used.”

Required evidence:
usable experiment subset actually includes 14 signers.

The dataset itself has 14 volunteers, but your experiment may not use all 14 after filtering.

Distinguish:

> “NSL23 contains 14 volunteers”

from:

> “Our experimental subset contains X volunteers.”

### Claim
“Temporal modeling is beneficial.”

Required evidence:
CNN+Transformer > single-frame CNN, ideally also > BiLSTM.

If Transformer does not outperform LSTM, report the result honestly and adjust the paper’s conclusion.

---

# 28. Related Work / References Need Expansion

Current paper has only about seven bibliography entries.

That is too thin for a polished computer vision research paper.

The next agent should conduct a focused literature review using primary sources.

Required categories:

1. NSL23 dataset paper
2. Zenodo dataset
3. previous NSL recognition work
4. word-level NSL Transformer work
5. image-based NSL character classification
6. major sign-language video recognition benchmarks
7. CNN–RNN / LSTM sign recognition
8. Transformer / video Transformer approaches
9. MobileNetV2
10. Transformer foundation paper
11. signer-independent evaluation examples in sign recognition
12. real-time sign/gesture recognition if making the real-time claim

Potential internationally relevant datasets/models:

- WLASL
- AUTSL
- isolated sign language recognition literature
- video Transformer literature such as TimeSformer / ViViT where genuinely relevant

Do not pad references with unrelated papers.

---

# 29. Reference Verification Rules

For every reference:

- verify title
- verify author list
- verify year
- verify venue
- verify DOI if one exists
- verify page range / volume / issue
- verify that the cited sentence is actually supported by the source
- prefer primary sources
- distinguish peer-reviewed publication from arXiv preprint

Known current references that should be rechecked:

- Sunuwar et al., Data in Brief, 2024
- Zenodo NSL23 v2
- Ligal and Baral, IOE Graduate Conference 2022
- Vaswani et al., 2017
- MobileNetV2, CVPR 2018
- ResNet, CVPR 2016
- Poudel et al. NSL character dataset, arXiv 2025

The Poudel paper is a **preprint** and should be described as such unless a peer-reviewed version exists by the time of final submission.

The paper must eventually be converted from numbered IEEE references to **APA** because that is what the target journal currently specifies.

---

# 30. GitHub Repository Requirements

The user specifically prefers GitHub instead of embedding code in the paper.

The final repository should contain at minimum:

```text
README.md
LICENSE
requirements.txt
notebooks/
    NSL23_CNN_Transformer_Colab.ipynb
src/                 # optional if notebook code is refactored
results/
    final_results.json
    benchmark_results.csv
    condition_results.csv
figures/
    ...
configs/
    final_config.json
docs/
    DATASET.md
```

Do **not** redistribute NSL23 unless the dataset license and repository policy explicitly permit it and doing so is necessary.

Preferred approach:

- notebook downloads NSL23 directly from Zenodo;
- README gives DOI and citation;
- derived annotations/manifest may be included if license allows.

README should include:

- what the project does
- Colab launch instructions
- dataset citation
- exact steps to reproduce
- expected compute
- final experiment configuration
- expected output files
- final paper citation after publication

Pin package versions when final results are produced.

Tag a reproducible release such as `v1.0-paper`.

Record the exact commit hash in the final manuscript/reproducibility notes.

---

# 31. Notebook Should Be Reorganized for Final Colab Use

Recommended notebook order:

## Section 0 — Reproducibility header

Print:

- Python
- PyTorch
- torchvision
- CUDA
- GPU
- seed
- git commit if applicable

## Section 1 — Install / imports

Only install packages that Colab actually needs.

## Section 2 — Download NSL23

- include all four ZIP files for final run
- verify checksums/file sizes from Zenodo metadata if possible
- report disk space
- skip files already downloaded safely

## Section 3 — Dataset audit

Build full manifest.

No training yet.

## Section 4 — Resolve aggregate videos

Segment or formally define isolated subset.

## Section 5 — Canonical labels

Build and verify official class map.

Assert expected classes.

## Section 6 — Condition metadata

Parse environment from folder structure.

## Section 7 — Split

Build signer-independent split.

Save split manifest.

Print signer/class/condition counts.

## Section 8 — Dataset loader

Decode / sample / transform.

Run sanity visualization.

## Section 9 — Models

- SingleFrameCNN
- CNNBiLSTM or CNNLSTM
- CNNTransformer

## Section 10 — Shared training engine

Avoid separate inconsistent training logic.

## Section 11 — Train / validation

Save:

- checkpoints
- histories
- config

## Section 12 — Test evaluation

Never touch test set during tuning.

## Section 13 — Runtime benchmark

T4 model-only and end-to-end.

## Section 14 — Condition robustness

Use final best model only.

## Section 15 — Paper figures and tables

Generate every CSV/PNG/PDF automatically.

## Section 16 — Single-video / streaming demo

Useful for real-time claim.

## Section 17 — Export to Drive / GitHub

Zip paper artifacts.

---

# 32. Test the Notebook Properly

The next agent must not just read the code.

It should execute as much as possible.

Required testing levels:

## Test A — Static notebook validation

- JSON opens
- every Python cell parses
- no undefined variables by inspection
- correct execution order
- all imports available

## Test B — Mini/synthetic smoke test

Before downloading 10+ GB:

- create a few short synthetic videos or use a few sample files;
- verify dataset loader returns:
  - `(B, T, 3, 224, 224)`
- run all three models forward
- run one training step
- run evaluation
- run result export

## Test C — Small real NSL23 subset

Download one part / use several actual videos:

- verify label parser
- verify signer parser
- verify condition parser
- verify `.MOV` decoding
- verify cropped/original behavior
- verify aggregate detection

## Test D — Full experiment

Only after all previous tests pass.

---

# 33. Potential Resource Problems in Colab

NSL23 is large.

Expect roughly 10+ GB compressed video data plus extracted files.

The notebook should:

- check free disk
- avoid downloading twice
- optionally save checkpoints/results to Google Drive
- not copy the entire extracted dataset unnecessarily
- allow restart from checkpoints

Video decoding may become the training bottleneck.

Possible optimizations:

- cache sampled frames to a derived local cache
- use efficient video reader (PyAV / Decord) if OpenCV seeking is slow
- persistent DataLoader workers
- pin memory
- AMP mixed precision

Do not change the scientific input content while optimizing.

---

# 34. Primary Results Table Must Come from Code

Generate a CSV such as:

`benchmark_results.csv`

with rows:

```text
Single-frame CNN
CNN + BiLSTM
CNN + Transformer
```

columns:

```text
model
seed
test_accuracy
macro_precision
macro_recall
macro_f1
params_m
latency_ms_clip_b1
throughput_clips_s
```

If using multiple seeds, produce:

`benchmark_summary.csv`

with mean/std.

The manuscript’s Table II should be populated from this file.

---

# 35. Condition Results Must Come from Code

Generate:

`condition_results.csv`

Suggested fields:

```text
condition
n_samples
n_classes
signers
accuracy
macro_f1
```

The manuscript’s Table III should be populated from this file.

If condition classes/signers differ greatly, make that visible rather than hiding it.

---

# 36. Statistical Rigor

For a stronger paper:

## Recommended

- 3 training seeds per model
- mean ± SD

## Also useful

bootstrap 95% confidence intervals on the test set.

If comparing two models on exactly the same test clips, a paired test such as McNemar can be explored for error disagreement, but it should not replace multi-seed uncertainty.

Do not add statistical tests just for appearance.

---

# 37. Discussion Must Be Written AFTER Results Exist

The final discussion should explain:

- why Transformer did or did not outperform BiLSTM
- whether temporal modeling helped dynamic vowels more than static consonants
- whether errors cluster among visually similar signs
- impact of unseen signer evaluation
- condition-related performance differences
- efficiency tradeoffs
- limitations of NSL23 scale
- limitations caused by aggregate video structure
- dataset location (Sikkim vs deployment motivation in Nepal)
- condition/signer confounding
- lack of facial/body-keypoint modeling
- isolated sign recognition is not full NSL translation
- alphabet classification is not sentence-level translation

Never describe speculative attention behavior as fact unless attention maps are actually analyzed.

---

# 38. Potentially Valuable Additional Analysis: Static vs Dynamic

Because:

- consonants are static
- some vowels are dynamic

one interesting research question is:

> Does Transformer temporal modeling provide a larger benefit on dynamic vowel gestures than on static consonants?

If the final segmented dataset supports this analysis, report:

- consonant performance
- vowel performance
- baseline gap by sign type

This could make the temporal argument much stronger.

However, do not force this analysis if the vowel data cannot support credible signer-independent evaluation.

---

# 39. “Real-Time” Claim Decision Rule

Keep “Real-Time” in the title only if:

- batch-size-1 latency is measured correctly;
- the streaming/sliding-window pipeline is implemented;
- inference is sufficiently fast relative to the input stream;
- the paper clearly defines what “real-time” means.

If not, rename to something like:

> “Nepali Sign Language Recognition Using a Hybrid CNN–Transformer Architecture”

or:

> “Efficient Video-Based Nepali Sign Language Recognition Using a Hybrid CNN–Transformer Architecture”

Scientific accuracy is more important than a stronger title.

---

# 40. Architecture Novelty Claim Decision Rule

Do not write:

> “We introduce a novel CNN–Transformer architecture”

unless there is a genuinely new architectural mechanism.

Safer:

> “We develop and evaluate a lightweight CNN–Transformer pipeline for NSL23.”

The contribution can be the dataset-specific protocol, robust evaluation, and reproducible application rather than a fundamentally new neural-network primitive.

---

# 41. Current Paper Formatting / Content Checklist

Before final submission:

- [ ] author removed from anonymized manuscript
- [ ] separate cover page created
- [ ] converted to Word
- [ ] Times New Roman 12
- [ ] double spaced
- [ ] 3,000–5,000 words
- [ ] 5–7 keywords
- [ ] APA citations
- [ ] DOIs verified
- [ ] all placeholders removed
- [ ] actual GitHub/anonymized code URL
- [ ] real dataset figures
- [ ] actual architecture figure
- [ ] actual confusion matrix
- [ ] real results in Table II
- [ ] real robustness results in Table III
- [ ] exact hyperparameters
- [ ] exact signer IDs
- [ ] exact dataset subset counts
- [ ] no unsupported SOTA claim
- [ ] no unsupported real-time claim
- [ ] no “code in Appendix” wording if code is only on GitHub
- [ ] all math matches implementation
- [ ] all references support cited statements
- [ ] current journal guidelines rechecked

---

# 42. Publication-Ready Definition of Done

The project is **not complete** when training finishes.

It is complete only when all of the following are true:

## Code

- notebook runs top-to-bottom in a clean Colab runtime
- no manual hidden state is required
- all dataset paths are reproducible
- all three models train
- final checkpoints load
- final metrics regenerate
- paper figures regenerate
- runtime benchmark works
- repository instructions work

## Data

- final usable dataset is explicitly defined
- aggregate-video policy is documented
- exact class/signer/condition counts are saved
- no accidental duplicate leakage
- split is signer-independent
- no test labels are silently discarded

## Science

- hypothesis is tested
- baselines are fair
- metrics are correct
- limitations are honest
- “real-time” is measured
- condition analysis is defensible
- results are not fabricated
- any negative result is reported

## Paper

- every numeric claim corresponds to a generated artifact
- every mathematical description matches code
- references are verified
- paper follows target journal format
- manuscript is within word limit
- figures/tables are readable and properly cited
- no draft language remains
- abstract contains actual results only after experiments finish
- conclusion summarizes actual findings
- manuscript is blinded appropriately

## Repository

- clean README
- exact Colab notebook
- requirements
- experiment config
- generated summary results
- appropriate license
- dataset citation
- no prohibited dataset redistribution
- reproducible release/tag

---

# 43. Recommended Agent Execution Order

The next AI agent should work in this order and should not skip ahead:

## Phase 1 — Verify target and sources

1. Recheck NCCSRJ author guidelines.
2. Recheck NSL23 Zenodo record and Data in Brief article.
3. Read the current LaTeX draft.
4. Read the current notebook.
5. Compile a list of every empirical claim.

## Phase 2 — Repair notebook enough for audit

6. Add vowel download.
7. fix undefined `items`.
8. remove global cropped-only policy.
9. create full manifest.
10. inspect aggregate videos and exact dataset structure.
11. verify canonical class labels.

## Phase 3 — Decide valid experimental dataset

12. determine whether aggregate videos can be reliably segmented.
13. if yes, generate verified segments.
14. if no, define and report isolated subset.
15. confirm usable signers/classes/conditions.

**Do not proceed to final training before this decision is scientifically sound.**

## Phase 4 — Finalize evaluation protocol

16. define exact signer train/val/test split.
17. freeze split file.
18. choose exact training hyperparameters.
19. ensure all models share protocol.

## Phase 5 — Run experiments

20. single-frame CNN
21. CNN + BiLSTM
22. CNN + Transformer
23. repeat seeds if feasible
24. save checkpoints + histories

## Phase 6 — Evaluation

25. accuracy
26. macro metrics
27. per-class metrics
28. confusion matrix
29. condition analysis
30. runtime T4 benchmark
31. optional ablation / static-vs-dynamic analysis

## Phase 7 — Generate paper assets

32. figures
33. tables
34. CSVs
35. final JSON summary

## Phase 8 — Rewrite paper from evidence

36. update dataset section with actual subset counts
37. update methodology to exactly match code
38. update mathematical equations
39. write Results
40. write Discussion
41. update Abstract with actual results
42. update Conclusion
43. expand Related Work
44. verify every reference
45. remove unsupported claims

## Phase 9 — Journal formatting

46. convert citations to APA
47. convert manuscript to journal Word format
48. separate title/cover page
49. anonymize body
50. check word count
51. final proofread
52. final code/repo link check

---

# 44. Instructions to the Next AI Agent

You are not being asked to preserve the existing draft at all costs.

If the experiment contradicts the draft, **the draft must change**.

Examples:

- If Transformer accuracy is lower than BiLSTM, report it.
- If the final dataset subset has 36 classes instead of 49, update the title/method/results.
- If only a small number of signers can be evaluated, remove broad generalization claims.
- If real-time latency is not achieved, remove “Real-Time.”
- If condition-wise analysis is confounded or too sparse, state the limitation instead of presenting a misleading table.
- If the current math does not match code, fix the math or code.
- If the journal format conflicts with the earlier IEEE assumption, follow the journal.

The standard is:

> **The paper must describe what was actually done, not what we hoped would happen.**

---

# 45. Important User Preferences / Decisions Already Made

Do not repeatedly re-ask these unless evidence forces a change:

- application-based machine learning / computer vision paper
- Nepal / NSL socially relevant focus
- hybrid CNN–Transformer core idea
- full reproducible Colab notebook
- NSL23 Zenodo dataset
- GitHub code instead of printing source code in manuscript
- signer-independent evaluation is important
- model comparison is the primary result
- condition robustness is secondary
- comparison models:
  - Single-frame CNN
  - CNN + LSTM/BiLSTM
  - CNN + Transformer
- report:
  - Top-1 accuracy
  - Macro-F1
  - confusion matrix
  - latency / throughput
- target runtime environment:
  - Google Colab NVIDIA T4
- actual results should populate Table II / III
- paper should ultimately be ready for real journal submission, not remain a conceptual draft

---

# 46. Important Current Paper Facts

The latest draft already has blank Table II / III structures.

## Table II

Model comparison:

- Single-frame CNN
- CNN + BiLSTM
- CNN + Transformer

with placeholders for:

- Top-1
- Macro-F1
- throughput
- latency

This concept is correct.

## Table III

Robustness of the best model across:

- Prepared
- Unprepared
- Real-world

The concept is useful, but the category definitions and dataset support must be corrected as discussed above.

---

# 47. Source Facts to Preserve Accurately

From the NSL23 source paper:

- 14 volunteers
- 5 experienced users, 9 beginners
- 36 consonants
- 13 vowels
- 630 videos
- 1,205 gestures
- 516 consonant videos / 971 gestures
- 114 vowel videos / 234 gestures
- individual-sign clips around 1–5 seconds
- aggregate videos contain multiple/all characters
- prepared and unprepared backgrounds
- indoor/outdoor real-world environments
- bright/dark/normal/natural lighting
- cropped versions exist for only part of the data
- consonants are static
- vowels include static + dynamic gestures

Do not convert “1,205 gestures” into “1,205 labeled video clips.”

---

# 48. Reference Sources / Links for the Next Agent

Primary dataset sources:

- Zenodo v2: `https://zenodo.org/records/10478554`
- Data in Brief DOI: `https://doi.org/10.1016/j.dib.2024.110080`
- PMC full text: `https://pmc.ncbi.nlm.nih.gov/articles/PMC10847474/`

Target journal:

- Journal home: `https://www.nepjol.info/index.php/nccsrj`
- Submission rules: `https://www.nepjol.info/index.php/nccsrj/about/submissions`

Example paper that originally motivated this project:

- “Stock Prediction Based on Transformer Model”
- NCCSRJ, Vol. 4 No. 1 (2025)
- DOI: `10.3126/nccsrj.v4i1.84356`

The example paper is useful for understanding the journal’s style and expected depth, but the **official Author Guidelines supersede assumptions based on the example**.

---

# 49. Final Warning About Automation / AI Writing

The next agent should not optimize for “sounding publishable” at the expense of scientific truth.

Before finalizing:

- detect unsupported numbers
- detect invented references
- detect invented comparisons
- detect methodology/code mismatch
- detect claims that exceed dataset evidence
- detect overly promotional wording
- remove language like “state-of-the-art” unless rigorously established
- use modest, precise academic writing

A paper with 88% real, reproducible accuracy is better than a paper claiming fabricated 93.5%.

---

# 50. Final Mission Statement for the Next Agent

Your job is to transform the current project from a promising draft into a **fully reproducible empirical study**.

Do not merely edit prose.

You must:

1. understand NSL23’s actual file structure;
2. repair and test the Colab notebook;
3. resolve the aggregate-video / isolated-label issue;
4. construct a valid signer-independent experiment;
5. run all baselines fairly;
6. collect real metrics;
7. measure real T4 runtime;
8. generate publication-quality figures;
9. audit the CNN–Transformer math against the code;
10. rewrite every unsupported paper claim;
11. verify and expand references;
12. convert the final manuscript to NCCSRJ’s current APA/Word format;
13. prepare the GitHub repository;
14. perform a final scientific and editorial audit.

**Do not consider the project finished until code, data protocol, results, figures, equations, citations, manuscript wording, and journal format are all mutually consistent and reproducible.  Also I have a general writing skills that is helpful for writing in general not as much in research paper probably so look at any of this that is applicable which should be most as this writing skills md file (writing_skills.md file in this project) is designed to not sound like AI so I think we can use this but we need to adapt this for resarch paper as well. Use this after everything is complete to humanize and complete the paper. Make sure not to lose important context. Also above instructions mentions github to host the code, ignore that for now. Also we can run this using mlx on the mac I have no? Let's use that instead if possibble instead of using colabbb as I want to use my own computer for this. Also the dataset information is available in the paper as reference I think which is the current state so look for that and odwnload the files from there. There should be a zip folder and inside should have a detailed dataset.docx file so make sure to read that. For reference the key paper for this is the NSL23 paper which describes the dataset. This is the link to the dataset:
https://zenodo.org/records/10478554#:~:text=This%20dataset%20is%20the%20first,in%20the%20form%20of%20videos The DetailedDataset.docx in this project describes the dataset information.
Downloaded the necessary stuff from the zeonodo.org webbsite. Make sure to think clearly how to best utilize the dataset as there are consonants, vowels etc. The current paper is nsl_cnn_transformer_ieee_UPDATED_tables.tex file in this project. Think about best to make this work. ALso if not possible to run using mlx, create a notebook I can upload to google colab to run this in colab. If we can use locally, great. If not, we can use mlx. Make sure the logic is completely sound. Keep working until the paper is ready to send for review for publication including having a sizeabble references so the background math and everything is referenced properly where needed. **

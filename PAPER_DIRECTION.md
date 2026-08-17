# Paper direction: what this project is for, and what to do next

**Status:** manuscript, landmark matrix, and webcam timing completed 17 August 2026.  
**This file is the current research plan.** It does not replace `context_nsl.md` (original handoff) or `PROTOCOL.md` (frozen experiment). It records the shift the RGB results forced.

---

## If you are confused, read this first

Three different things got mixed together. They are not the same.

1. **NSL23 (the dataset)**  
   Pre-recorded alphabet videos from Sunuwar et al. A person signs one letter (or a long aggregate tape). Files on disk. **This is not a webcam, and it is not real-time.** You cannot “see real-time usage” inside the zip files. You can only **train** on them.

2. **The experiment (what we measure on those files)**  
   Which recognizer is more accurate on held-out clips: RGB CNN / BiLSTM / Transformer vs hand landmarks. This uses the dataset as a test bench.

3. **The product goal (why you started)**  
   A cheap **live webcam** alphabet recognizer. That is a **separate program** you build after a method looks good. You point a camera at a person, run the model every few frames, and time FPS. NSL23 does not do that for you.

So: the dataset does not demonstrate real-time use. Real-time is a **deployment test** you add. The paper can be “which method is accurate enough *and* cheap enough to run live,” but only if you actually time a live loop. Until then, you only have clip accuracy.

MediaPipe is not banned. The closest NSL paper that compared it (Bhusal & Baral, sentence-level, 2024) found **lower error with MediaPipe Holistic** (WER 20.23% vs 25.62% without it). Shrestha et al. used MediaPipe for letters and reported high validation accuracy. Known weaknesses: dark video, blur, occlusion, missing hands. NSL23 has dark clips, so that is a reason to **measure** failure rate, not a reason to skip the tool.

### Other papers’ percentages are not one leaderboard

Every number below was produced by **that paper’s own model, on that paper’s own test files**. Accuracy means: correct predictions ÷ test samples in *that* experiment. They are not scores on NSL23, and they are not comparable to each other.

| Paper | Their number | What produced it |
|---|---|---|
| Poudel et al. (2026) | 90.45% (MobileNetV2), 88.78% (ResNet50) | Their 54,000 **still images**, 36 classes, 70/15/15 split. Table 1 lists 90.45% as **training** accuracy; validation is 98.99%; a later table on 8,100 images reports ~0.99. The abstract still quotes 90.45%. |
| Heuju & Rijal (2022) | 55.9% (71 words), 89% (9 words) | Their **private word videos**, 70/30 split |
| Ligal & Baral (2022) | ~88% | Different **private word** set |
| Shrestha et al. (2024) | val ~88% | Their letter images + MediaPipe |
| Bhusal & Baral (2024) | WER 20% | 40 **sentences**, not letters |
| Sunuwar et al. (2024) | none | Dataset only |

The original `context_nsl.md` goal was **not** “beat all of these.” It was a real-time, low-resource NSL alphabet system on **public NSL23 video**, with a CNN–Transformer as the candidate method.

You cannot accomplish “beat all of these” with one number. Different exams. 91% on Poudel’s photos does not beat Heuju’s 55% on words, and 8% on NSL23 unseen signers does not lose to Poudel’s 90% in any fair sense.

Using Poudel’s Kaggle images is allowed (cite them). Sensible use: **pretrain** an image CNN on Poudel, then test on NSL23 video frames. That may help RGB. It is not “beating 90.45%.” Beating Poudel only counts if you evaluate on **Poudel’s test split**. That would be a different paper (still-image classification), not the webcam-on-NSL23 paper.

---


## Dataset ownership (do not confuse this)

The videos are **not** ours.

- Source: **NSL23**, Sunuwar, Borah, and Kharga (2024), *Data in Brief*.
- Public files: Zenodo record `10478554`, DOI `10.5281/zenodo.10478554`, CC BY 4.0.
- We use them as a **secondary analysis**: cite the dataset paper, keep the license, do not pretend we collected the clips, do not redistribute the videos unless the license and venue require it.

That was always the plan in `context_nsl.md`. Using a public dataset is normal. Uniqueness has to come from **protocol + method + measurement**, not from owning NSL23.

---

## What the paper is actually for

The original point was not “invent a Transformer.” It was:

> A **real-time**, **low-resource** NSL alphabet recognizer that can run from a **webcam**, useful in a Nepal/NSL setting.

The CNN–Transformer was a **candidate method** for that product, not the product.

`context_nsl.md` already said: keep “Real-Time” in the title only if sliding-window / webcam inference is implemented and timed. The RGB study never built that pipeline. It timed offline clips on an Apple M4. End-to-end RGB inference was about **one second per clip** because decoding and preprocessing dominate. That is not a webcam system.

So the goal still stands. The RGB hybrid is the wrong vehicle for it.

### Working title

Do not use the old title (`Bridging the Silent Gap: Real-Time … CNN–Transformer`). The hybrid did not win, and real-time is not measured yet.

**Use this:**

> Low-Resource Nepali Sign Language Alphabet Recognition on NSL23: Comparing RGB Video Models and Hand Landmarks for Webcam Use

That title matches work already done (RGB comparison on NSL23) and work still required (landmarks + a webcam you can test). It does not claim the Transformer is better, and it does not say “real-time” until FPS is measured.

Shorter fallback if the journal wants fewer words:

> Comparing RGB and Landmark Models for Isolated Nepali Sign Language Alphabet Recognition on NSL23

### Poudel images: use as a helper, not as the paper

Option A (pretrain on Poudel, test on NSL23) is **worth doing if it is cheap**, because more handshape photos can help a fine-tuned CNN. It will not by itself give you a low-resource webcam. A MobileNet on full frames is still heavier than MediaPipe coordinates.

Order:

1. Landmarks + tiny classifier on the frozen NSL23 folds (this is the webcam candidate).
2. Optional: Poudel-pretrained / fine-tuned RGB CNN on the same folds (this is “did extra images help?”).
3. Put the faster accurate model in a webcam script you can run on this Mac.

Poudel does not replace NSL23 and does not replace the RGB evaluation already finished.

---

## What already exists (do not redo)

| File | What it is |
|---|---|
| `context_nsl.md` | Original CNN–Transformer handoff, journal rules, dataset pitfalls |
| `PROTOCOL.md` | Frozen RGB experiment: 375 isolated original clips, S1/S2 holdout, 18 runs |
| `CLAIM_LEDGER.md` | What the RGB manuscript is allowed to say |
| `artifacts/results/` | Real RGB numbers (~6–7% unseen-signer accuracy) |
| `manuscript/blinded_manuscript.md` | Draft of the RGB-only paper (placeholders, not a Word file) |

MediaPipe has **not** been used. No webcam demo exists. No landmark model has been trained.

---

## Why the RGB CNN–Transformer is not the path

On the honest NSL23 isolated subset:

- Single-frame CNN: **5.8% ± 0.8%**
- CNN–BiLSTM: **7.0% ± 2.1%**
- CNN–Transformer: **6.3% ± 0.8%**

The temporal heads reached **100% training accuracy** and did not transfer to unseen signers. Most NSL23 alphabet classes are **static handshapes**. Frozen ImageNet RGB features do not describe those shapes well. A Transformer on 16 full frames is also the opposite of a cheap webcam model.

Keep those runs. They are the **control**: RGB video models failed this protocol. Do not write a paper that claims the hybrid won.

---

## The shift (one experiment)

Stay on **NSL23**. Stay on the **same 375 clips** and the **same S1/S2 signer-held-out folds**.

Add a **hand-landmark** pipeline:

1. Sample the same 16 frames already cached.
2. Run **MediaPipe Hands** on those frames (CPU is enough; that is the low-resource point).
3. Train a **small** classifier on the landmark sequence (MLP or BiLSTM on 21×3 coordinates, not another large RGB Transformer).
4. Compare it with the three RGB models already finished.
5. If landmarks beat RGB, implement a **webcam sliding-window** demo and measure **frames per second on this Mac** (and later a cheap CPU/Colab if needed).

MediaPipe is allowed. It is a tool. Shrestha et al. (2024) used it on **their** letter data, not on public NSL23 video with a documented signer split. Poudel et al. fine-tuned a CNN on **still images they collected**, not on NSL23 video. This test is still open.

---

## Two numbers, because they answer different questions

The webcam product and the scientific claim are not the same split.

| Question | Split | Why it matters |
|---|---|---|
| Can this help a **new person** (future translator)? | Keep **S1 / S2 held out** | This is the hard, honest number. Do not drop it. |
| Can a **learner** practice letters on a webcam? | Signer-dependent or few-shot enrollment | This is the realistic alphabet-tutor setting. High accuracy is allowed if named correctly. |

A real-time demo without an unseen-signer number would repeat the overclaim we already rejected. An unseen-signer number without a webcam timing would not be the product paper you wanted.

**Call it real-time only after** a live or simulated stream is timed (capture → landmarks → predict → optional smoothing), in ms/frame or FPS, on the target machine. Offline clip throughput is not webcam FPS.

---

## Recommended path (do this, in order)

1. Extract MediaPipe landmarks on the existing primary clips. Log failed detections; do not silently impute junk.
2. Train the landmark model on the frozen folds (`configs/isolated_twofold_protocol.json`). Same seeds if practical: 17, 42, 73.
3. Fill a comparison table: RGB CNN / BiLSTM / Transformer vs landmark MLP / landmark BiLSTM. Accuracy, macro-F1, CPU latency.
4. Only if landmarks are clearly better than RGB: build a webcam loop (deque of recent landmark frames, predict every *k* frames). Time it on this Mac.
5. Then write the paper / TeX from those artifacts.

Do **not** start the manuscript rewrite until step 3 exists. Do **not** collect a new dataset unless landmarks still fail and the limitation is “too few signers,” in which case the paper can stop at the benchmark plus that limit.

---

## What uniqueness is allowed to mean

Allowed:

- First documented isolated-sign, signer-held-out protocol on public NSL23 video.
- First pose-vs-RGB comparison on that protocol.
- A measured low-resource webcam prototype, if it actually runs.

Not allowed:

- “NSL23 is our dataset.”
- “Our CNN–Transformer is better.”
- “We beat Poudel 90%” (different data, images, split).
- “Real-time” without a streaming measurement.

---

## Decision rule after the landmark run

- **Landmarks beat RGB and a webcam loop is fast enough:** the paper is a low-resource NSL23 alphabet recognizer, with RGB hybrids as the failed alternative.
- **Landmarks beat RGB but still fail unseen signers:** the paper is the benchmark plus a tutor-mode (same-user) webcam tool. Say so. Do not call it a translator.
- **Landmarks do not beat RGB:** stop adding architectures. The contribution is the NSL23 isolated protocol and the negative result. Real-time webcam is then out of scope for this dataset subset.

The path to take is the first block above: **landmarks on the frozen NSL23 folds, then webcam timing, then the paper.** That is the original product goal, with a method that can actually run on a camera.

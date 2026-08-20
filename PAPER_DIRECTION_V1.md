# Version 1 paper direction (historical)

This file is the Version 1 rationale: why RGB hybrids were abandoned, why MediaPipe landmarks were allowed, and how the 375-clip S1/S2 protocol was chosen. It is **not** the current plan.

Current dashboard: [`PAPER_DIRECTION.md`](PAPER_DIRECTION.md). Version 2 protocol: [`VERSION2.md`](VERSION2.md). Frozen V1 experiment: [`PROTOCOL.md`](PROTOCOL.md).

S1 and S2 were the Version 1 held-out test. They influenced landmark, angle, prototypical, crop, and fusion choices, so Version 2 treats them as **development** signers. The 59.4% ± 3.3% prototypical xyz+angle number remains a V1/development result, not a confirmatory test.

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

## What Version 1 was for

The original point was not “invent a Transformer.” It was:

> A **real-time**, **low-resource** NSL alphabet recognizer that can run from a **webcam**, useful in a Nepal/NSL setting.

The CNN–Transformer was a **candidate method** for that product, not the product.

`context_nsl.md` already said: keep “Real-Time” in the title only if sliding-window / webcam inference is implemented and timed. The RGB study never built that pipeline. It timed offline clips on an Apple M4. End-to-end RGB inference was about **one second per clip** because decoding and preprocessing dominate. That is not a webcam system.

### Working title (Version 1 manuscript)

> Low-Resource Nepali Sign Language Alphabet Recognition on NSL23: Comparing RGB Video Models and Hand Landmarks for Webcam Use

### What Version 1 completed (do not redo)

| File | What it is |
|---|---|
| `PROTOCOL.md` | Frozen RGB experiment: 375 isolated original clips, S1/S2 holdout, 18 runs |
| `CLAIM_LEDGER.md` | What the RGB manuscript is allowed to say |
| `artifacts/results/` | RGB ~6–7%; prototypical xyz+angles **59.4% ± 3.3%** |
| `manuscript/nsl23_low_resource.tex` | Version 1 exploratory technical report |

MediaPipe Hands 0.10.14 **was** used on the 375 clips. A webcam loop runs at 33–42 FPS on an Apple M4. That loop is a speed measurement, not 49-way streaming accuracy.

### Why the RGB CNN–Transformer is not the path

On the honest NSL23 isolated subset:

- Single-frame CNN: **5.8% ± 0.8%**
- CNN–BiLSTM: **7.0% ± 2.1%**
- CNN–Transformer: **6.3% ± 0.8%**

The temporal heads reached **100% training accuracy** and did not transfer to unseen signers. Most NSL23 alphabet classes are **static handshapes**. Frozen ImageNet RGB features do not describe those shapes well.

Keep those runs. They are the **control**: RGB video models failed this protocol.

### Two numbers (Version 1)

| Question | Split | Why it matters |
|---|---|---|
| Can this help a **new person**? | S1 / S2 held out | Version 1 primary number. In Version 2 those signers are development, not confirmatory testers. |
| Can a **learner** practice letters? | Signer-dependent or few-shot enrollment | Tutor protocol. Do not mix with unseen-signer accuracy. |

**Call it real-time only after** a live or simulated stream is timed. Offline clip throughput is not webcam FPS. Webcam timing stays secondary in Version 2 and is run only after Stage 4.

### What uniqueness is allowed to mean

Allowed:

- First documented isolated-sign, signer-held-out protocol on public NSL23 video.
- First pose-vs-RGB comparison on that protocol.
- A measured low-resource webcam prototype, if it actually runs.

Not allowed:

- “NSL23 is our dataset.”
- “Our CNN–Transformer is better.”
- “We beat Poudel 90%” (different data, images, split).
- “Real-time” without a streaming measurement.
- Quoting 59.4% as an untouched confirmatory test after S1/S2 guided representation choice.

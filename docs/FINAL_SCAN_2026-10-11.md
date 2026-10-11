# NSL23 final scan — 11 October 2026

## Verdict

The main results are supported by the retained experiment artifacts. The paper makes sense as an **exploratory study of isolated alphabet classification on two development signers**, not as confirmation of general accuracy for unseen NSL users. The 59.4% ± 3.3% accuracy and 57.4% ± 3.6% pooled macro-F1 remain unchanged.

This scan found substantive errors that the August audit missed. They have been corrected in the current LaTeX manuscript, its figures and tables, the claim ledger, and the reader's guide. “Everything is verified” would still be too strong: the checks below have explicit boundaries, and V2 cannot be completed without qualified human annotation.

The humanizer review replaced awkward phrasing and unsupported causal explanations with direct descriptions of the observed results. It did not invent explanations, change numerical results, or make the paper sound more conclusive than its design permits.

## Findings and corrections

| Finding | Correction and scientific consequence |
|---|---|
| The 90.6% five-class result was called unseen-signer “1-shot.” | It uses full training-class means, not one support clip per class. The label now says **five-class restriction with full training-class means**. The separate 93.3% same-signer result really uses one other-lighting support example per class. Neither is 49-way recognition accuracy. |
| The conclusion described the headline as measured on 375 evaluation clips. | 375 is the full primary sample. Each seed evaluates **196 unique pooled S1/S2 clips**, 98 per signer; each fold trains on 277. |
| The normalization equation used `max(s, ε)`, unlike the implementation. | The actual denominator is `s` when `s ≥ 10⁻⁶`, otherwise **1**. The paper and guide now describe that fallback. Saved caches and predictions were not changed. |
| The middle-frame equation mixed one-based and zero-based indexing. | With one-based mathematical notation, the code's index 8 is the **ninth of 16 frames**, written `floor(T/2)+1`. |
| Geometry was described too broadly as orientation/viewpoint invariant. | The angles are invariant to a common rigid transform and uniform scale of valid supplied points. Image-derived detector outputs need not obey that transform when viewpoint changes. The concatenated xyz component still retains orientation. Zero-length-vector handling is disclosed. |
| The angle layout sounded like an exact replication of Chamachot and Lertniphonphan. | It is an adaptation. The paper now lists the implemented finger chains and wrist triplets and states the different normalization, detection flag, temporal encoder, and unsquared Euclidean distance. |
| The hand-selection score sounded like landmark-quality confidence. | It is the **handedness-classification confidence**. The 97.8% frame detection count measures output availability, not landmark localization accuracy. |
| Prototypical episodes were described without their replacement consequence. | Small classes can reuse a training clip in an episode's support and query sets. Five support plus two query draws need not be seven distinct clips. This weakens the episode's learning interpretation, but does not leak evaluation clips into training. The existing protocol has not been silently redesigned. |
| Improvements were sometimes assigned causal mechanisms without controlled evidence. | The discussion now separates observed scores from hypotheses about resolution, background, detector behavior, or overfitting. Different training losses and schedules prevent attributing the entire gain to the loss alone. |
| The text said both the evaluation person and indoor setting were unseen. | Only the person is excluded from training. The other complete-alphabet signer supplies indoor bright/dark training clips. The manuscript no longer claims a held-out recording-condition experiment. |
| The training description obscured augmentation and learning-rate differences. | Cross-entropy uses cosine decay and Gaussian jitter on all non-flag features (including angles in follow-on runs and initially zero missed-frame coordinates). Prototypical training uses a constant learning rate and no feature jitter. The batch normalization of weighted cross-entropy is also made explicit. |
| Throughput wording could imply tested live-camera accuracy. | The reported 33–42 FPS comes from recorded-video runs through the streaming loop. It is not recognition accuracy, spotting performance, or a new live-camera test. |
| The supplementary ZIP omitted needed modules and run histories and malformed the blinded LaTeX author replacement. | The builder now includes all `nsl23` modules, scripts, configurations, 60 run histories, protocol tables, manuscript figures, reviewer instructions, and regression tests. Blinding preserves real LaTeX newlines. Aggregation uses local checkpoint paths; exported prediction-only audits require an explicit missing-checkpoint flag. |
| The old RGB-only Word template remained easy to mistake for the current paper. | It is explicitly marked historical. Its builder requires `--allow-historical-template`; it is not a current submission route. |
| The canonical reproduction notebook still expected 18 RGB-only runs; the README and results summary retained the incorrect five-class “1-shot” label. | The notebook now includes both landmark models and their extraction step, checks 30 primary runs, and routes new runs and summaries outside the reference directories. Legacy notebooks are labelled historical. The results-summary generator now regenerates the follow-on section from JSON rather than silently discarding it. |
| Re-running prototypical training could overwrite a saved V1 run. | Training now refuses any destination containing a checkpoint, metrics, history, or training receipt. A regression test checks refusal before data loading and preservation of the existing bytes. |
| The guide described Stage 3 selection as mean top-1, despite the locked macro-recall endpoint. | Its selection rule now matches the protocol and executable implementation: mean per-signer 49-class macro-recall across development LOSO folds and seeds; clip-pooled top-1 is a tie-breaker, not the selection headline. |
| Floating-point NaN/infinity and negative imported frame positions could evade some annotation checks. | Review completion, verified-segment metadata, and pass-2 import now reject nonfinite values and invalid bounds. Regression tests cover these cases. |
| Pass-2 HTTP defaults could fall back to pass-1 final boundaries if explicit bounds were omitted. | Video-window and save defaults now use the original proposal or the reviewer's own pass-2 bounds. A disposable HTTP test checks blinding and persistence without touching the master CSV. |

## What was verified

### Data and numerical results

- The strict verifier freshly decoded the **first frame of all 630 retained source videos** and checked the 375 eligible original isolated clips, 375 distinct RGB caches, exact 277/98 folds, 49-class coverage, and S1/S2 exclusion from their respective training folds. Without its explicit `--verify-source-decode` flag, the verifier only checks historical decode flags and current file presence; this distinction is now visible in its output.
- All **60 run prediction files** were checked against evaluation sample identities, ontology indices, true/predicted labels, correctness flags, signer, sign type, and lighting. Accuracy and macro-F1 were recomputed rather than trusting stored flags or summary tables. Each seed's pooled evaluation has 196 unique clips.
- The 30-run primary and 30-run follow-on summaries match those primitives, including the corrected pooled macro-F1 aggregation established in August. Macro-F1 is nonlinear: it must be calculated after pooling predictions, not averaged across fold-level F1 values.
- The six saved prototypical runs were rescored under the alternative protocols, including 500 episodes per run. Every existing run record and aggregate value matched exactly. A `support_definitions` field now makes the two five-class settings explicit.
- Enrollment summaries retain 76.2% ± 4.2% for xyz BiLSTM, 78.2% ± 2.8% for xyz+angle BiLSTM, and 73.0% ± 1.5% for the prototypical encoder. The valid closed protocol uses one other-lighting support clip for **every** class. Label-dependent `enroll_replace_class` and `enroll_adapt_class` results remain excluded from scientific enrollment claims.
- The historical V1 snapshot matches **all 15 inputs** in Git commit `0eb1ff20337b584a5a7e4b06d5e4d5a350915fc7`. This establishes the contents of that dated snapshot, not that no earlier development or smoke-test access occurred. The manuscript discloses the access chronology.
- All 19 bibliography entries are cited and all manuscript citation keys resolve. This is citation linkage, not proof that every sentence in every cited paper was independently verified.

### Saved-model replay

`scripts/verify_saved_predictions.py` loads all **60 checkpoints**, checks their ontology and available parameter-count receipts, and reconstructs training-class prototypes where applicable. Its main replay checks:

- **3,528 predictions:** all 98 evaluation clips for each of the 36 landmark/angle/prototypical runs.
- **72 predictions:** three representative clips for each of the 24 RGB/crop runs.
- **1,764 additional closed-enrollment queries:** all queries in the 18 reported enrollment runs, with accuracy and macro-F1 recomputed from newly obtained embeddings.

The checkpoint receipt is `artifacts/audit/checkpoint_replay.json`. These are repeated inference checks with existing checkpoints, not newly trained models. Exact class predictions do not imply identical probabilities or floating-point bits on all hardware.

### Executable safeguards and artifacts

All **24 regression tests pass**. They cover prediction tampering, duplicate clips, historical snapshot integrity, full bundle module inventory, valid LaTeX blinding, independent reviewer exchange, preservation of manually inserted proposals, immutable proposals, same-person double review rejection, adjudication only after completed passes, frozen-artifact hashes, test/train separation, hierarchical V2 metrics, rigid-transform palm-coordinate behavior, notebook matrix scope, and prevention of checkpoint overwrites. The HTTP check runs on disposable files; the real review master stays unmodified. A separate smoke check successfully generates a half-second reviewer MP4 window from a real retained tape using ffmpeg, with output confined to a temporary directory.

The **20-page paper** and **28-page reader's guide** are rebuilt from their sources and rendered for page inspection. The compact bundle is checked independently after extraction for numerical aggregation, import of all 18 package modules, usable core entry points, and anonymous manuscript compilation. Bundle reproducibility means repeated ZIP builds from the same inputs yield the same bytes; it does not mean a new training run reproduces every weight.

## Primary-source checks

The following sources were read for the study-specific claims:

- [NSL23 dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10847474/): alphabet inventory, 630 videos, reported 1,205 gesture instances, volunteers, collection context, and dataset limitations. The 375-clip study sample is our manifest-derived subset, not a count asserted by the source article.
- [Official Zenodo version 2 record](https://zenodo.org/records/10478554): archive names, byte sizes, MD5 values, license, and release metadata. The record's publication date is December 2023 despite later technical update timestamps; the bibliography's 2023 release year is appropriate.
- [Heuju and Rijal's word-level study](https://conference.ioe.edu.np/publications/ioegc12/IOEGC-12-115-12166.pdf): its 71-word collection, 70/30 partition, and reported 55.92% full-vocabulary and 89.14% nine-word results. These are not directly comparable NSL23 alphabet tests.
- [Shrestha et al.'s letter-image study](https://conference.ioe.edu.np/publications/ioegc15/IOEGC-15-025-C1-4-63.pdf): its own image collection and MediaPipe-guided image pipeline, rather than the original isolated NSL23 video folds.
- [Chamachot and Lertniphonphan's accepted paper](https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Chamachot_Geometry-Aware_Metric_Learning_for_Cross-Lingual_Few-Shot_Sign_Language_Recognition_on_CVPRW_2026_paper.html) and [full preprint](https://arxiv.org/html/2603.09213v1): geometric metric-learning motivation and a different static-image few-shot protocol. Our descriptor is not asserted to be their exact triplet layout or distance formulation.
- [Prototypical Networks](https://arxiv.org/abs/1703.05175) and [MediaPipe Hands](https://arxiv.org/abs/2006.10214): class-mean metric classification and the hand-landmark pipeline. The mathematical descriptions were also compared directly with this repository's implementation.

This is a targeted primary-source check, not a comprehensive literature search, an independent verification of other researchers' datasets, or a formal retraction search for every bibliography entry.

## What remains unresolved

1. **Only two complete development signers.** S1/S2 informed the follow-on choices. No prose correction or rerun of their existing checkpoints makes the result confirmatory.
2. **Seed SD is not population uncertainty.** Three seed scores describe optimization variation. Repeated predictions of the same source clips are not additional independent participants.
3. **Limited baselines and confounding.** Frozen ImageNet RGB is a controlled baseline, not a strong claim about all RGB methods. Prototypical and cross-entropy models differ in loss, episode construction, schedule, and feature jitter. A large observed gain cannot identify a single causal mechanism.
4. **Support/query reuse during training.** The existing episodic sampler permits overlap for sparse classes. Changing it would create a new experiment, not repair the saved outputs. Any prospective change must be documented before outcomes; the locked V2 training specification was not modified in this scan.
5. **V2 is still pre-annotation.** There are 691 proposals on 21 tapes, zero completed real review passes, no verified segments, and no frozen V2 sample manifests. The hierarchy hash remains `09a2b7b2cb3a580c6cf3e3cd0c952a04ec4a793863ab27212c0cdeb55c5ab009`. No V2 training or Stage 4 evaluation was run.
6. **Not every operation was independently rerun.** Full RGB inference, fresh training, landmark extraction from every retained source, fusion inference, a new runtime benchmark, and live-camera accuracy were not rerun here. Fusion/runtime statements were checked against their saved receipts and aggregation. Their limits remain explicit.
7. **Original ZIP hashes were not recomputed now.** The four duplicate archives were verified and removed in August, with a recovery receipt. The retained sources' first frames were decoded again, not every frame of every source. This scan does not pretend that missing archives were freshly hashed.

The correct next step remains the disposable S3 reviewer-calibration pilot, followed by two qualified independent annotation passes, full-tape miss sweeps, pre-adjudication agreement, adjudication, coverage, and freeze. At least two sealed signers must have all 49 verified classes before a confirmatory Stage 4 claim. If that gate fails, the annotation resource and V1 exploration can still be reported without inventing a confirmatory endpoint.

## Storage

The workspace is approximately **14 GB**, not 20–40 GB at this scan. The source collection accounts for most of it; extracted media and existing checkpoints were retained. The earlier cleanup recovered 9.626 GiB by removing checksum-verified duplicate archives. This scan did not download another copy, create new training runs, or delete research assets. New audit receipts are small; PDF, DOCX, and the compact ZIP replace existing outputs. Reviewer preview windows remain disposable cache files, not the source of annotation truth.

## Repeatable checks

Run from the project root with the reference environment and `PYTHONPATH=src`:

```bash
.venv/bin/python scripts/verify_protocol_freeze.py
.venv/bin/python scripts/verify_primary_protocol.py --verify-v1-freeze --verify-source-decode
.venv/bin/python scripts/audit_paper_numbers.py
.venv/bin/python scripts/verify_saved_predictions.py --output artifacts/audit/checkpoint_replay.json
.venv/bin/python -m unittest discover -s tests -v
bash scripts/compile_paper.sh
.venv/bin/python scripts/build_analysis_provenance.py
.venv/bin/python scripts/build_reproducibility_bundle.py
```

The explanation guide additionally needs `python-docx`; its pages must be rendered and inspected after regeneration. After real annotation starts, the pre-annotation status assertions in `audit_paper_numbers.py` must be replaced with an explicitly dated post-annotation audit, not bypassed to force a passing receipt.

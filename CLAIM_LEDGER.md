# NSL23 Manuscript Claim Ledger

This ledger is the manuscript's fail-closed evidence map. A quantitative or methodological claim may enter the final article only when the evidence column points to an existing source or generated artifact. Rows marked `pending` must be resolved after the full matrix and runtime benchmark finish.

| Claim family | Permitted wording / boundary | Evidence | Status |
|---|---|---|---|
| Dataset provenance | NSL23 version 2 is available on Zenodo under CC BY 4.0, DOI 10.5281/zenodo.10478554. | `artifacts/audit/archive_checksums.json`; Zenodo record 10478554 | verified |
| Dataset scale | The release contains 630 videos representing 1,205 claimed gesture instances, 49 alphabet classes, and 14 volunteers. “Instances” must not be called separately labelled clips. | Sunuwar et al. (2024); `Details Dataset.docx`; `artifacts/audit/nsl23_manifest_summary.json` | verified |
| File integrity | All five official files match the release byte counts and MD5 values. | `artifacts/audit/archive_checksums.json`; output of `scripts/verify_primary_protocol.py --verify-archive-hashes` | verified |
| Primary sample definition | The primary analysis uses 375 original, individually labelled clips. It excludes all 23 aggregate sources and all cropped/preprocessed derivatives. | `data/manifests/nsl23_manifest_all.csv`; `artifacts/audit/aggregate_exclusions_summary.json`; `PROTOCOL.md` | verified |
| Aggregate exclusion | Aggregate recordings lack released gesture timestamps; equal-duration candidate bins did not reliably preserve class identity on visual review. Do not claim the candidate segments are ground truth. | `annotations/aggregate_source_review.csv`; `artifacts/segment_review/`; `artifacts/audit/aggregate_exclusions.csv` | verified |
| Signer-independent protocol | Fold 1 holds out S1; fold 2 holds out S2. Each fold has 277 training and 98 test clips, and train/test both contain all 49 classes. | `configs/isolated_twofold_protocol.json`; `artifacts/audit/isolated_twofold_summary.json`; `scripts/verify_primary_protocol.py` | verified |
| Test conditions | Each held-out signer has one bright and one dark original clip for every class. Condition results are descriptive, not causal illumination effects. | cached fold manifests; `artifacts/audit/isolated_twofold_summary.json` | verified |
| External validity | The complete-alphabet unseen-signer test covers two signers only. It does not estimate population-wide signer generalization or generalization to prepared/outdoor settings. | `PROTOCOL.md`; fold manifests | verified |
| Primary-set composition | Only S1 and S2 contribute original isolated vowels; within either fold, vowel training comes from the other held-out-capable signer alone. The five additional isolated signers contribute prepared consonants only. | `artifacts/audit/primary_set_composition.json`; primary manifest | verified |
| Input sampling | Sixteen temporal-bin-center frames are aspect-preserving letterboxed to 224×224; no horizontal flip is used; photometric augmentation is sequence-consistent and training-only. | `src/nsl23/frame_cache.py`; `src/nsl23/dataset.py`; `configs/experiment_local_m4.json` | verified |
| Model definitions | All models use the same frozen ImageNet-1K V2 MobileNetV2 encoder. The temporal heads and parameter counts must match the code/configuration. | `src/nsl23/models.py`; run `metrics.json`; Figure 2 | verified |
| Optimization | Seeds 17/42/73, batch 8, 30 fixed epochs, class-weighted cross entropy, AdamW 3e-4, weight decay 1e-4, cosine annealing, deterministic algorithms. | `configs/experiment_local_m4.json`; `src/nsl23/train.py`; run artifacts | verified |
| Model selection | There is no validation set or early stopping. Only the fixed epoch-30 checkpoint is tested. Do not call `best_checkpoint.pt` validation-selected. | `PROTOCOL.md`; `selection_policy` in all run metrics/checkpoints | verified |
| Test-access chronology | A one-epoch software smoke test accessed held-out S1 after protocol/configuration files were fixed but before their exact hash snapshot; scores were not used for any design or selection decision. | `artifacts/audit/test_access_timeline.json`; file modification times | verified |
| Primary performance | Report top-1 accuracy and macro-F1 from 196 pooled out-of-fold clips per seed, then mean and sample SD over three seeds. Also expose fold-specific values. | `artifacts/results/model_summary.csv`; `pooled_out_of_fold_by_seed.csv`; `fold_summary.csv` | verified |
| Uncertainty | Clip-level class-stratified bootstrap intervals are descriptive; they do not address signer-population uncertainty with only two held-out signers. | per-run metrics; `artifacts/results/primary_results_summary.json` | verified |
| Architecture comparison | Use descriptive paired seed differences. Do not claim statistical significance or state of the art. | `artifacts/results/paired_differences_summary.csv` | verified |
| Lighting breakdown | Report bright versus dark scores from pre-specified out-of-fold subsets with mean and SD over seeds. | `artifacts/results/condition_summary.csv`; Figure 5 | verified |
| Runtime | Report synchronized batch-one model-only and end-to-end median/p95 latency on the Apple M4 MPS host. Do not call it streaming or real-time performance. | `artifacts/runtime/*/runtime.json` | verified |
| Reproducibility | The code, configurations, notebook, manifests builders, audit scripts, and compact supplementary bundle reproduce the analysis from the official archives. | `NSL23_CNN_Transformer_Reproducibility.ipynb`; `NSL23_reproducibility_bundle.zip`; protocol snapshot | verified |

## Prohibited claims

- “93.5% accuracy” or any number not present in final result artifacts.
- “State of the art,” “first,” or “novel architecture” without a directly comparable literature basis.
- “1,205 isolated clips.”
- Dataset collection “in Nepal.” The NSL23 recordings were collected in Sikkim, India.
- “Real-time system,” “streaming recognition,” or deployment readiness based only on offline clip latency.
- Generalization to 14 unseen signers, outdoor scenes, prepared backgrounds, continuous signing, or the broader Deaf/NSL-user population.
- Causal claims about lighting, architecture, or expertise from this observational, two-test-signer protocol.

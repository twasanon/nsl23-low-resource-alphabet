# An Audited Signer-Held-Out Benchmark of CNN Temporal Models for Nepali Sign Language Alphabet Videos

**Abstract**

Small public datasets and splits that mix signers make Nepali Sign Language (NSL) recognition difficult to evaluate. We audit NSL23 and compare a middle-frame MobileNetV2 classifier, a CNN–BiLSTM, and a CNN–Transformer under a fixed signer-held-out protocol. The release contains 630 decodable videos, but its 1,205 reported gesture instances are not 1,205 independently labelled clips. The primary sample therefore uses 375 original, individually labelled videos and excludes aggregate recordings without timestamps and cropped or preprocessed derivatives. Two complete-alphabet signers are held out in turn. Each fold contains 277 training clips and 98 test clips, with one bright and one dark recording for every test class. All models share a frozen ImageNet-pretrained MobileNetV2 encoder, a 30-epoch schedule, and three fixed seeds. The highest mean accuracy was {{BEST_ACCURACY_PM}} with {{BEST_ACCURACY_MODEL}}; the highest mean macro-F1 was {{BEST_MACRO_F1_PM}} with {{BEST_MACRO_F1_MODEL}}. Scores remained low, and the temporal heads did not produce a consistent descriptive advantage over the single-frame baseline despite fitting the training data. Synchronized Apple M4 measurements separate model-only latency from original-video decoding and preprocessing. On this small, compositionally uneven subset, added temporal capacity did not solve signer shift. The audit trail, predictions, figures, and executable notebook preserve the sample definition and evaluation boundary for later comparisons.

**Keywords:** Nepali Sign Language; signer-independent recognition; video classification; CNN–Transformer; low-resource sign language; NSL23

## Introduction

Automatic sign-language recognition is often presented as a classification problem, yet the experimental split can change what is being measured. When clips from the same person occur in both training and test sets, a model can exploit appearance, clothing, background, camera, or articulation patterns that recur across the split. A signer-held-out test asks a harder and more useful question: can the classifier recognize signs produced by someone it did not see during training? Large isolated-sign benchmarks such as MS-ASL, WLASL, and AUTSL treat signer variation as part of the task, although their languages, vocabularies, and collection scales differ from NSL23 (Li et al., 2020; Sincan & Keles, 2020; Vaezi Joze & Koller, 2019).

Nepali Sign Language has far fewer public video resources. NSL23 contains the 36 consonants and 13 vowels of its published alphabet inventory, recorded under several backgrounds and illumination conditions (Sunuwar et al., 2024). Its file structure, however, requires care. The dataset paper reports 1,205 gesture instances represented in 630 videos. Some videos contain a single labelled sign, whereas others contain a sequence of all consonants or vowels. Cropped and preprocessed derivatives are also distributed with original recordings. Treating every file as an independent labelled example, or treating every reported gesture instance as a separately labelled clip, would misstate the sample size. Randomly distributing related derivatives across splits would introduce another source of leakage.

This study asks whether temporal aggregation improves unseen-signer alphabet classification when the frame encoder, input sampling, optimization schedule, and label set are fixed. The comparison includes a middle-frame CNN, a bidirectional long short-term memory head, and a Transformer encoder head. The aim is not to introduce a new temporal architecture. LSTMs and Transformers are established sequence models (Hochreiter & Schmidhuber, 1997; Vaswani et al., 2017), and Transformers have already been applied to both continuous and isolated sign recognition (Camgoz et al., 2020; Heuju & Rijal, 2022). The contribution here is narrower: an audited sample definition, a reproducible complete-alphabet signer-held-out protocol, and a controlled comparison on NSL23.

Three questions organize the analysis. First, how many original, independently labelled NSL23 clips can support an isolated-sign classifier without inferred segment boundaries? Second, do either of the two temporal heads improve top-1 accuracy or macro-F1 over a single-frame model on the same held-out clips? Third, what computation does each architecture require when both network latency and complete video preparation are measured on the same local device? Because only two signers support complete-alphabet testing, the answers are descriptive. They should not be read as population estimates for NSL users.

## Related work

Modern sign-recognition systems commonly combine a spatial representation with a temporal model. CNN features can be aggregated by recurrent networks, temporal convolutions, or attention. Rastgoo et al. (2021) review these families and the substantial variation among datasets and evaluation designs. Camgoz et al. (2020) used Transformer-based sequence modeling for continuous sign recognition and translation. Those tasks require temporal alignment over sign sequences and differ from the isolated 49-class clips studied here. The architectural citation establishes precedent for attention in sign video; it does not create a directly comparable accuracy baseline.

Large isolated-sign datasets make signer-aware evaluation practical. MS-ASL introduced a broad American Sign Language vocabulary with signer-diverse benchmark partitions (Vaezi Joze & Koller, 2019). WLASL similarly provided word-level video data and benchmark methods at a much larger scale than NSL23 (Li et al., 2020). AUTSL includes 226 Turkish Sign Language classes and an explicitly user-independent split (Sincan & Keles, 2020). Scores from those datasets cannot rank an NSL23 model because the languages, class inventories, recordings, and partitions differ. Their relevance is methodological: signer identity is a grouping variable, not a clip attribute that can be ignored during splitting.

NSL-specific work remains comparatively limited. Heuju and Rijal (2022) evaluated Transformer and CNN–RNN approaches on word-level NSL data using a different dataset and partition. Shrestha et al. (2024) studied letter detection and fingerspelling with MediaPipe landmarks and a CNN. These studies show active work on NSL recognition, but neither supplies a result that is directly comparable with the audited NSL23 folds used here. The present benchmark therefore reports only internal comparisons among models trained and tested on identical clips.

MobileNetV2 was selected as a common frame encoder because its inverted residual design provides a compact visual representation (Sandler et al., 2018). ImageNet pretraining often transfers useful visual features, although transfer quality depends on the source and target domains (Kornblith et al., 2019). Hand shape and signing motion differ substantially from ImageNet object classification, so a frozen encoder is a resource-conscious control rather than an assumption that its features are optimal for NSL. Freezing it also isolates the effect of the classification and temporal heads more clearly in a small-data experiment.

## Materials and methods

### Dataset source and file audit

NSL23 version 2 was obtained from the official Zenodo record (Sunuwar et al., 2023). The source study was conducted at Sikkim Manipal Institute of Technology in Sikkim, India; this experiment is not field validation in Nepal (Sunuwar et al., 2024). The release is licensed under CC BY 4.0. Before decoding or filtering, the byte size and MD5 digest of each distributed archive and the dataset-detail document were checked against the record. All five files matched. A recursive manifest then retained the source archive, relative path, signer token, raw filename, canonical class, recording descriptors, media metadata, derivation status, and exclusion reason for every video. This preserves the distinction between a source file and a reported gesture instance.

The audit found 630 decodable videos. Of these, 607 are individually labelled files and 23 are aggregate recordings containing sequences of consonants or vowels. The distribution also includes 234 cropped or otherwise preprocessed files. Some aggregate and derived categories overlap. The primary sample is the union-complement of both exclusions: 375 original, individually labelled clips from seven signers. It contains all 49 classes. The source article describes 14 volunteers, including experienced and beginner signers, but that release-level number is not the number of participants represented by independently labelled original clips in this experiment (Sunuwar et al., 2024). Table 1 gives the audited counts and fold sizes; Figure 1 traces the exclusions from the release to the primary matrix.

{{TABLE_PROTOCOL}}

{{FIGURE_STUDY_FLOW}}

Aggregate recordings were reviewed rather than discarded by filename alone. The release does not provide per-gesture timestamps. Candidate equal-duration bins were generated in the published alphabet order and inspected as contact sheets. Timing drift accumulated across several recordings, and at least one sequence appeared incomplete or shifted. An equal-duration slice could therefore assign the wrong class while still looking like a plausible hand gesture. No candidate bin was accepted as ground truth. The review decisions and contact sheets are retained in the audit materials so future work can replace this exclusion if trustworthy temporal annotations become available.

Cropped and preprocessed derivatives were excluded to keep related renderings of a source performance out of opposite folds and to keep the input definition consistent. The study uses the original isolated clip when both an original and a derivative exist. This choice reduces the nominal sample count, but it gives every prediction a traceable source and avoids presenting derived copies as independent observations.

### Class ontology and signer-held-out folds

The fixed ontology contains 49 classes: 36 consonants followed by 13 vowels. Filename aliases were resolved with signer-aware rules because the release uses more than one transliteration convention. No unknown class was introduced, and no test clip was removed because its label was missing from training. Automated assertions require every fold to contain all 49 labels on both sides of the split.

Only S1 and S2 have original, individually labelled clips for the complete alphabet. We therefore use two pre-specified folds. Fold 1 trains on S2, S7, S8, S10, S11, and S12 and tests on S1. Fold 2 replaces S2 with S1 in training and tests on S2. Each training partition has 277 clips; each test partition has 98. The held-out signer contributes exactly two clips per class, one labelled bright and one labelled dark. There is no validation partition, early stopping, or model-specific hyperparameter search. The final epoch is evaluated once for each model, fold, and seed.

The folds are asymmetric across sign types. The five additional isolated-clip signers contribute prepared consonants only. Original isolated vowels occur only for S1 and S2, so each fold learns all vowel classes from the other complete-alphabet signer. A correct vowel prediction must therefore cross a particularly narrow signer boundary. Consonant training has greater signer diversity, although the test signers and their unprepared recording context remain unseen. We report consonant and vowel summaries to expose this composition, not to claim an intrinsic difference between sign types.

### Video preparation

Sixteen frames are sampled at the centers of equal temporal bins between the decoded start and end of each clip. This policy covers the clip without assuming a fixed frame rate or taking only its beginning. Frames are converted to RGB, resized while preserving aspect ratio, and padded with black pixels to 224 × 224. ImageNet channel means and standard deviations are then applied. The frame cache stores unsigned 8-bit arrays keyed by deterministic sample identifiers; normalization occurs when a batch is loaded.

Training augmentation is limited to brightness and contrast perturbations. One pair of factors is drawn for a clip and applied consistently to all 16 frames, so augmentation does not create artificial flicker. Horizontal flipping is disabled because laterality can affect sign meaning. Test clips receive no random transformation. The single-frame baseline takes the middle element of the same 16-frame sample, which keeps decoding and spatial preprocessing comparable while withholding temporal aggregation.

### Model comparison

All models use the ImageNet-1K V2 MobileNetV2 encoder, whose final feature vector has 1,280 dimensions (Sandler et al., 2018). Encoder weights and batch-normalization state remain frozen. The single-frame CNN applies dropout and a 49-way linear layer to the middle-frame feature. The CNN–BiLSTM first projects every frame feature to 256 dimensions, passes the sequence through a one-layer bidirectional LSTM with 128 hidden units per direction, applies layer normalization, mean-pools the 16 outputs, and uses dropout before the linear classifier. The CNN–Transformer uses the same 256-dimensional projection, adds learned positional embeddings, and applies two encoder layers with four attention heads and a 512-dimensional feed-forward block. A final layer normalization precedes temporal mean pooling, head dropout, and classification. Attention dropout is zero; head dropout is 0.2. Table 2 reports the instantiated parameter counts, and Figure 2 shows the common encoder and three heads.

{{TABLE_MODELS}}

{{FIGURE_ARCHITECTURES}}

For the Transformer, each head computes `softmax(QKᵀ/√d_h)V` from the projected sequence, where `d_h = 256/4 = 64` features per head. The implementation uses PyTorch's batch-first encoder layer and learned positions of length 16. Each encoder layer uses post-residual normalization (`norm_first=False`), GELU activation, and zero encoder dropout. No class token or causal mask is used. Mean pooling in both temporal models keeps the comparison focused on the sequence module rather than different aggregation targets. Parameter counts include the shared frozen backbone and are taken from the instantiated checkpoints, not estimated from the diagram.

### Training and evaluation

The complete matrix comprises three architectures, two folds, and seeds 17, 42, and 73, for 18 runs. Each run uses batch size 8 for exactly 30 epochs. The objective is inverse-frequency class-weighted cross entropy; fold-specific weights are normalized to mean one. AdamW uses a learning rate of 3 × 10⁻⁴ and weight decay of 10⁻⁴, followed by cosine annealing to zero over the fixed schedule (Loshchilov & Hutter, 2019). Deterministic algorithms are enabled, and no label smoothing is applied. The shared settings were fixed before the primary matrix and were not adjusted after viewing model results.

Top-1 accuracy and macro-F1 are the primary descriptive outcomes. Macro precision, macro recall, weighted F1, class-level precision/recall/F1, and fixed-order confusion matrices are also retained (Sokolova & Lapalme, 2009). Each run is first evaluated within its 98-clip fold. For a given seed, predictions from the two nonoverlapping held-out signers are then pooled into 196 out-of-fold clips. Architecture summaries are the mean and sample standard deviation of the three seed-level pooled metrics. This gives seed variability, not signer-population uncertainty.

Each run also includes 2,000 class-stratified clip bootstrap replicates for accuracy and macro-F1. Bootstrap percentiles describe prediction uncertainty conditional on the observed held-out clips (Efron & Tibshirani, 1994). With only two held-out signers, the clips are not independent draws from the population of NSL users. The intervals therefore cannot support a broad signer-generalization claim and are supplied in the machine-readable results rather than promoted as inferential evidence in the main comparison.

The bright and dark subsets were specified in advance. Each contains one clip per class from each held-out signer, or 98 pooled clips per seed. Recording condition was not randomized independently of signer, session, background, or camera. Differences between these subsets are reported as descriptions of the available test material, not causal illumination effects. Architecture differences are similarly expressed as seed-matched percentage-point differences without a significance test; three seeds do not provide a defensible sampling distribution for architectural superiority.

### Reproducibility and test-access chronology

The analysis stores the complete manifest, fold definitions, protocol configuration, frame-cache metadata, histories, checkpoints, test predictions, result tables, and figure inputs. A hash snapshot covers the frozen protocol and training implementation. A generated notebook rebuilds the audit and analysis and keeps full training behind explicit flags so its safe default path can be executed without silently launching a long experiment. The supplementary archive omits NSL23 media and author-identifying paths. Users obtain the licensed source files from Zenodo and verify them before reconstruction.

A one-epoch software smoke test accessed the S1 test partition after the protocol and configuration files had been written but before their exact cryptographic hash snapshot was recorded. Its purpose was to find runtime failures in the complete path. The score was not used to change the sample definition, model, hyperparameters, schedule, or analysis. The chronology and file modification times are preserved in a separate audit record. We disclose this ordering because a protocol claim should distinguish the time a design was fixed from the later time its bytes were hashed.

### Runtime procedure

Runtime was measured locally on a 10-core Apple M4 Mac mini with 16 GB of unified memory, using the PyTorch MPS backend and batch size one. The pre-specified checkpoint is seed 42 from the S2-held-out fold. Each model receives 20 warm-up iterations, followed by 100 synchronized model-only iterations from a normalized tensor already on the device. A second benchmark warms decoder and allocation paths with five untimed clips, then processes every one of the 98 original test videos once. Its timer includes file opening, decoding, temporal sampling, letterboxing, normalization, device transfer, forward computation, softmax, and synchronization. Median and 95th-percentile milliseconds are reported separately for the two definitions. End-to-end throughput is computed as 1,000 divided by median clip latency. This is offline clip processing, not sliding-window or webcam throughput.

## Results

### Primary signer-held-out performance

Across pooled out-of-fold predictions, the single-frame CNN reached {{SINGLE_ACC_PM}} accuracy and {{SINGLE_MACRO_F1_PM}} macro-F1. The CNN–BiLSTM reached {{BILSTM_ACC_PM}} accuracy and {{BILSTM_MACRO_F1_PM}} macro-F1, while the CNN–Transformer reached {{TRANSFORMER_ACC_PM}} accuracy and {{TRANSFORMER_MACRO_F1_PM}} macro-F1. Equal support for all 49 classes makes macro recall numerically equal to accuracy. Mean performance was low for every architecture; chance top-1 accuracy under a uniform 49-class guess is approximately 2.0%, but no chance-corrected or inferential claim is made. Table 3 reports all primary metrics, and Figure 3 shows the seed-level distributions.

{{TABLE_PRIMARY}}

{{FIGURE_PRIMARY}}

The seed-matched Transformer-minus-single-frame difference was {{TRANSFORMER_MINUS_SINGLE_ACC_PP}} percentage points for accuracy and {{TRANSFORMER_MINUS_SINGLE_F1_PP}} points for macro-F1. Relative to the BiLSTM, the Transformer difference was {{TRANSFORMER_MINUS_BILSTM_ACC_PP}} points for accuracy and {{TRANSFORMER_MINUS_BILSTM_F1_PP}} points for macro-F1. The BiLSTM-minus-single-frame differences were {{BILSTM_MINUS_SINGLE_ACC_PP}} and {{BILSTM_MINUS_SINGLE_F1_PP}} points, respectively. These small, variable contrasts do not support the directional hypothesis that temporal aggregation would consistently improve unseen-signer recognition under the fixed protocol.

### Fold and training behavior

Results were similarly weak on both unseen signers. For S1, accuracy was {{SINGLE_S1_ACC_PM}}, {{BILSTM_S1_ACC_PM}}, and {{TRANSFORMER_S1_ACC_PM}} for the single-frame, BiLSTM, and Transformer models. On S2, the corresponding values were {{SINGLE_S2_ACC_PM}}, {{BILSTM_S2_ACC_PM}}, and {{TRANSFORMER_S2_ACC_PM}}. Fold-level macro-F1 values and seed variation are shown in Table 4. Neither fold supplies evidence that one held-out participant alone caused the overall finding.

{{TABLE_FOLDS}}

At epoch 30, training accuracy was {{SINGLE_FINAL_TRAIN_ACC_PM}} for the single-frame model, {{BILSTM_FINAL_TRAIN_ACC_PM}} for the BiLSTM, and {{TRANSFORMER_FINAL_TRAIN_ACC_PM}} for the Transformer. The corresponding gaps between mean final training accuracy and pooled test accuracy were {{SINGLE_TRAIN_TEST_ACC_GAP_PP}}, {{BILSTM_TRAIN_TEST_ACC_GAP_PP}}, and {{TRANSFORMER_TRAIN_TEST_ACC_GAP_PP}} percentage points. The temporal heads' ability to fit the training set did not carry to the unseen signers. Figure 4 shows that this separation arose during the common schedule rather than from an early-stopping choice.

{{FIGURE_TRAINING}}

The most frequent off-diagonal CNN–Transformer predictions were {{TRANSFORMER_TOP_CONFUSIONS}}. Each true class contributes 12 repeated predictions to the aggregated three-seed, two-fold confusion count. These counts locate recurring errors in this matrix, but the small per-signer sample does not establish a linguistic or perceptual similarity between the label pairs. The supplementary archive contains the full count and row-normalized matrices and a per-class recall plot.

### Recording condition and sign type

Bright-condition accuracy was {{SINGLE_BRIGHT_ACC_PM}} for the single-frame model, {{BILSTM_BRIGHT_ACC_PM}} for the BiLSTM, and {{TRANSFORMER_BRIGHT_ACC_PM}} for the Transformer. Dark-condition accuracy was {{SINGLE_DARK_ACC_PM}}, {{BILSTM_DARK_ACC_PM}}, and {{TRANSFORMER_DARK_ACC_PM}}, respectively. Macro-F1 values follow the same pre-specified subsets in Table 5 and Figure 5. Because brightness labels co-occur with recording-session attributes, the contrasts should be treated as a dataset description rather than an estimate of illumination robustness.

{{TABLE_CONDITIONS}}

{{FIGURE_LIGHTING}}

We also separated consonants and vowels because their training sets differed. Consonant accuracy was {{SINGLE_CONSONANT_ACC_PM}}, {{BILSTM_CONSONANT_ACC_PM}}, and {{TRANSFORMER_CONSONANT_ACC_PM}}; vowel accuracy was {{SINGLE_VOWEL_ACC_PM}}, {{BILSTM_VOWEL_ACC_PM}}, and {{TRANSFORMER_VOWEL_ACC_PM}}. Each fold has only one vowel-training signer. This benchmark cannot separate signer count, gesture dynamics, and class difficulty, so we do not attribute the consonant–vowel contrast to a single mechanism.

### Runtime

Model-only median latency was {{SINGLE_MODEL_MEDIAN_MS}} ms for the single-frame CNN, {{BILSTM_MODEL_MEDIAN_MS}} ms for the BiLSTM, and {{TRANSFORMER_MODEL_MEDIAN_MS}} ms for the Transformer. When original-video decoding and preparation were included, median latency was {{SINGLE_E2E_MEDIAN_MS}}, {{BILSTM_E2E_MEDIAN_MS}}, and {{TRANSFORMER_E2E_MEDIAN_MS}} ms, corresponding to {{SINGLE_E2E_CPS}}, {{BILSTM_E2E_CPS}}, and {{TRANSFORMER_E2E_CPS}} clips per second. The end-to-end cost includes reading 16 frames from each full clip and is therefore not comparable with a per-frame webcam claim. Table 6 reports median and 95th-percentile latency under both timing definitions.

{{TABLE_RUNTIME}}

## Discussion

Added temporal capacity did not solve unseen-signer generalization on the audited NSL23 subset. The temporal heads had approximately 12 and 22 times as many trainable parameters as the single-frame head, and they fitted the training examples much more closely. Their pooled test metrics, however, remained in the same low range. The train–test gap is consistent with fitting the training signers and recording contexts rather than learning features invariant to them.

Vowel accuracy was higher than consonant accuracy for all three architectures, including the single-frame baseline. The pattern therefore cannot be attributed to temporal aggregation. Nor do the weak temporal-model results show that motion is irrelevant to NSL. Several vowels are described as dynamic, and a middle frame cannot represent a trajectory. This experiment did not recover a generalizable temporal representation from the available supervision and frozen backbone. For vowels, every fold observes only one training signer. A temporal network can memorize that person's trajectory, timing, framing, or background without learning how another person realizes the same sign. The descriptive consonant–vowel results cannot disentangle those possibilities.

The frozen ImageNet encoder may also have limited transfer to this task. MobileNetV2 offers a controlled and computationally modest representation, but ImageNet pretraining is not hand- or motion-specific. Fine-tuning later spatial layers, using hand or body keypoints, or pretraining on signer-diverse sign video may improve transfer. Those are new experiments, not explanations established by the present data. Any future comparison should also match trainable capacity or report it explicitly; otherwise a temporal-head gain could reflect parameter count as much as sequence modeling.

Here, sample accounting materially changed the experiment. The release-level count of 1,205 gestures describes content represented inside 630 files. An isolated-sign classifier needs independently labelled temporal boundaries. Assigning labels to equal slices of aggregate videos would create apparent scale at the cost of unknown label error. Excluding those sources leaves fewer signers and makes the complete-alphabet test depend on S1 and S2, but the resulting claim is auditable: every primary row points to one original file with one released label.

Random clip splits would answer an easier question on this release. S1 and S2 contribute repeated bright and dark takes for every class, and the archive contains derived versions of other videos. A random split could place the same signer, session, or related rendering on both sides. Such a result may still describe within-signer clip classification, but it should not be called unseen-signer performance. The present low scores quantify a stricter question and should not be compared numerically with studies that use random or unspecified splits.

Runtime results answer a limited engineering question. The temporal heads were much slower than the single-frame head in model-only timing, but all three end-to-end medians clustered near one second because video decoding and preparation dominated. Neither measurement implements a streaming buffer, overlapping decode, or temporal decision policy. A deployment study would need continuous capture, window stride, prediction smoothing, dropped-frame handling, energy use, and user-centered latency. Reporting offline clips per second as "real time" would collapse these different systems.

### Limitations

Only two signers can serve as unseen complete-alphabet tests using original individually labelled clips. The three seeds repeat optimization on the same two people; they do not increase the signer sample. Population-wide generalization, expert-versus-beginner differences, and performance across the 14 release participants are outside the evidence. The primary tests also cover unprepared indoor bright and dark recordings, not the prepared and outdoor conditions represented elsewhere in the full release.

The folds are compositionally uneven across sign types. Vowels have one training signer per fold, whereas consonants have six. Condition, signer, session, camera, and background are not independently randomized. Condition and sign-type analyses are consequently descriptive. Class-stratified clip bootstraps retain the same signer dependence and cannot repair it.

The study uses a frozen encoder, one shared hyperparameter setting, 16 uniformly sampled frames, mean temporal pooling, and a fixed 30-epoch endpoint without validation. These controls make the three heads comparable, but they are not a search for the best version of each architecture. The final checkpoint filename is retained for software compatibility; it is the fixed epoch-30 checkpoint, not a validation-selected optimum. Results may change with signer-diverse pretraining, spatial fine-tuning, pose input, boundary annotations, or a larger dataset.

Finally, the software smoke test reached one held-out fold before the exact protocol hash was recorded. Although its metric did not influence a decision, a stronger prospective study would seal and timestamp the complete analysis package before any test execution. An external preregistration and a larger signer-held-out cohort would provide a firmer basis for architectural and population claims.

### Ethical and reproducibility considerations

This is a secondary analysis of a public, cited dataset; no participants were recruited for the present study. The source videos contain identifiable people. The reproducibility bundle therefore includes manifests, hashes, code, and derived numerical outputs but does not redistribute video. Researchers should obtain the data under its CC BY 4.0 terms and retain the dataset creators' attribution. With performance in the observed range, these alphabet classifiers are not suitable for unsupervised interpretation in education, health care, legal, or public-service settings.

Technical accuracy alone would not establish usefulness. A later system should be evaluated with Deaf and NSL-using communities, with consent and control over where recordings are stored and how predictions are used. NSL should not be treated as a direct encoding of spoken Nepali, and an alphabet classifier is not a translator for natural signed discourse. The present benchmark is a research instrument for one constrained task.

## Conclusion

An audit of NSL23 identified 375 original, independently labelled clips that support leakage-resistant isolated alphabet classification without fabricated temporal boundaries. Across two signer-held-out folds and three seeds, a CNN–BiLSTM and CNN–Transformer did not show a consistent descriptive advantage over a middle-frame MobileNetV2 classifier. All three models generalized poorly despite much stronger training fit in the temporal heads. More temporal capacity alone was insufficient when complete-alphabet supervision came from very few signers. A follow-up study should begin with trustworthy aggregate-video annotations or more signer-diverse recordings, then test representations trained for hands and sign motion. The current scores do not justify a stronger claim.

## References

Camgoz, N. C., Koller, O., Hadfield, S., & Bowden, R. (2020). Sign language transformers: Joint end-to-end sign language recognition and translation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 10020–10030). https://doi.org/10.1109/CVPR42600.2020.01004

Efron, B., & Tibshirani, R. J. (1994). *An introduction to the bootstrap*. Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593

Heuju, S., & Rijal, B. (2022). Word level Nepali Sign Language recognition using Transformer. In *Proceedings of the 12th IOE Graduate Conference* (pp. 883–890). https://conference.ioe.edu.np/publications/ioegc12/IOEGC-12-115-12166.pdf

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do better ImageNet models transfer better? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2656–2666). https://doi.org/10.1109/CVPR.2019.00277

Li, D., Rodriguez Opazo, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In *Proceedings of the IEEE Winter Conference on Applications of Computer Vision* (pp. 1448–1458). https://doi.org/10.1109/WACV45572.2020.9093512

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Bkg6RiCqY7

Rastgoo, R., Kiani, K., & Escalera, S. (2021). Sign language recognition: A deep survey. *Expert Systems with Applications, 164*, 113794. https://doi.org/10.1016/j.eswa.2020.113794

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 4510–4520). https://doi.org/10.1109/CVPR.2018.00474

Shrestha, G., Thing, N., Dhakal, P., Dahal, P., Karki, P., & Guragai, M. K. (2024). Nepali Sign Language letter detection and finger spelling using MediaPipe and CNN. In *Proceedings of the 15th IOE Graduate Conference* (pp. 166–170). https://conference.ioe.edu.np/publications/ioegc15/IOEGC-15-025-C1-4-63.pdf

Sincan, O. M., & Keles, H. Y. (2020). AUTSL: A large scale multi-modal Turkish Sign Language dataset and baseline methods. *IEEE Access, 8*, 181340–181355. https://doi.org/10.1109/ACCESS.2020.3028072

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management, 45*(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Sunuwar, J., Borah, S., & Kharga, A. (2023). *Nepali Sign Language - Consonant and Vowel (NSL23)* (Version 2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10478554

Sunuwar, J., Borah, S., & Kharga, A. (2024). NSL23 dataset for alphabets of Nepali Sign Language. *Data in Brief, 53*, 110080. https://doi.org/10.1016/j.dib.2024.110080

Vaezi Joze, H. R., & Koller, O. (2019). MS-ASL: A large-scale data set and benchmark for understanding American Sign Language. In *Proceedings of the 30th British Machine Vision Conference* (Article 100). https://www.microsoft.com/applied-sciences/uploads/publications/3/ms-asl.pdf

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (Vol. 30). https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

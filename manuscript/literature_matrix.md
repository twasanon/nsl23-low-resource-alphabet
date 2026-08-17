# Literature matrix for the NSL23 manuscript

This matrix separates contextual evidence from directly comparable evidence. None of the cited large-dataset scores is a benchmark against the present NSL23 protocol.

| Source | Task / evidence | Evaluation detail relevant here | Permitted use in manuscript |
|---|---|---|---|
| Sunuwar, Borah, and Kharga (2024), *Data in Brief*, DOI 10.1016/j.dib.2024.110080 | NSL23 dataset paper | 630 videos, 1,205 represented gesture instances, 49 alphabet classes, 14 volunteers, heterogeneous recording formats and conditions | Primary source for dataset scope, collection location, participant composition, conditions, and aggregate-video structure |
| Sunuwar, Borah, and Kharga (2023), Zenodo v2, DOI 10.5281/zenodo.10478554 | Public NSL23 release | Official files, byte sizes, MD5 checksums, CC BY 4.0 | Data availability, integrity, and license |
| Shristi Heuju and Birodh Rijal (2022), “Word Level Nepali Sign Language Recognition Using Transformer,” *Proceedings of the 12th IOE Graduate Conference*, pp. 883–890 | 71 NSL word classes; Transformer versus CNN–RNN | Reports a simple 70/30 partition, not a comparable signer-held-out NSL23 protocol | Nepal/NSL-specific motivation and evidence that temporal/attention models have been explored; do not compare scores directly |
| Guptaraj Shrestha, Nishan Thing, Prajita Dhakal, Prasanga Dahal, Pukar Karki, and Manoj Kumar Guragai (2024), “Nepali Sign Language Letter Detection and Finger Spelling Using MediaPipe and CNN,” *Proceedings of the 15th IOE Graduate Conference*, pp. 166–170 | NSL letters/fingerspelling using landmarks and CNN | Different data and task formulation | Recent Nepal-specific computer-vision context; no performance ranking against this study |
| Camgoz et al. (2020), “Sign Language Transformers,” DOI 10.1109/CVPR42600.2020.01004 | End-to-end continuous sign recognition and translation | Continuous signing, large benchmarks, Transformer sequence modeling | Establishes Transformer use in sign-language sequence modeling; task is not comparable to isolated NSL23 classification |
| Li, Rodriguez Opazo, Yu, and Li (2020), WLASL, DOI 10.1109/WACV45572.2020.9093512 | Large-scale word-level isolated ASL recognition dataset/benchmark | Signer variation is a core benchmark concern | Motivates signer-aware evaluation and the gap between large-resource and NSL settings |
| Vaezi Joze and Koller (2019), MS-ASL, BMVC Article 100 | Large-scale isolated ASL benchmark | Signer-independent generalization and broad vocabulary | Context for modern isolated-sign benchmarks; no direct score comparison |
| Sincan and Keles (2020), AUTSL | 226 isolated Turkish Sign Language classes | Official user-independent split is substantially harder than a random split | Direct conceptual support for reporting signer-independent rather than random clip splits |
| Holmes et al. (2022), low-resource signer-independent recognition | Low-resource sign recognition | Explicit signer-independent emphasis | Supports evaluation design and conservative low-resource interpretation |
| Rastgoo, Kiani, and Escalera (2021), deep sign-language survey | Review of recognition methods and datasets | Reviews spatial, temporal, pose, and sequence architectures | Broad related-work synthesis |
| Vaswani et al. (2017) | Transformer/self-attention | General sequence-model foundation | Architecture citation only |
| Hochreiter and Schmidhuber (1997) | Long short-term memory | Recurrent sequence-model foundation | BiLSTM baseline foundation |
| Sandler et al. (2018) | MobileNetV2 | Lightweight frame encoder | Backbone architecture citation |
| Kornblith, Shlens, and Le (2019) | ImageNet model transfer | Transferability of ImageNet representations | Rationale for frozen pretrained features under low data, while acknowledging domain mismatch |
| Loshchilov and Hutter (2019) | AdamW | Decoupled weight decay | Optimizer citation |
| Sokolova and Lapalme (2009) | Classification performance measures | Macro versus weighted metrics | Metric-definition rationale for class imbalance |

## Comparison boundary

The final discussion may compare *methodological choices*—signer independence, temporal modeling, data scale, and split design—but must not call a score superior to WLASL, MS-ASL, AUTSL, or the Nepal-specific studies. Their languages, vocabularies, source data, and evaluation partitions differ. A claim of “state of the art” is prohibited.

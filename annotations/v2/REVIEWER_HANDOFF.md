# NSL23 Version 2 reviewer handoff

## The assignment in one sentence

Independently watch the 21 original aggregate Nepali Sign Language (NSL) alphabet videos, decide what each computer-proposed window contains, correct its temporal boundaries, and report any signs the proposal detector missed.

This is **annotation quality control**, not model training and not an evaluation of the reviewer. The original dataset labels identify whole files or their intended ordered contents; they do not provide trustworthy frame-level start/end boundaries for the 691 motion proposals. The review establishes which proposed clips are genuine, what class each clip shows, and where the complete gesture begins and ends.

## Who is qualified

The reviewer should be able to recognize all 49 NSL alphabet signs from video: 36 consonants and 13 vowels. Fluency or professional interpretation credentials are welcome but not required if the person can demonstrate reliable alphabet recognition in the calibration pilot. They should be comfortable saying “ambiguous” instead of guessing.

Pass 1 and pass 2 must be different named people. A reviewer who only annotates may be thanked in the acknowledgments; annotation alone does not automatically require authorship. Authorship should follow the target journal’s contribution policy and any contribution that develops beyond annotation.

## Exact workload

- 21 source tapes from 7 signers: S3 (6), S4 (3), S5 (2), S6 (6), S9 (1), S13 (1), and S14 (2).
- 22.13 minutes of original source footage in total.
- 691 computer-generated motion proposals: S3 204, S4 99, S5 59, S6 145, S9 73, S13 68, and S14 43.
- The source tapes occupy about 3.47 GiB uncompressed.
- The nominal tape-order total is 572 signs. This is an **upper-level expectation, not a quota**: detector duplicates raise the proposal count, detector misses require manual insertion, and a damaged or incomplete sign should not be forced into a class.
- Practical time budget: about 30–60 minutes for calibration, then roughly 5–8 focused hours for one full pass. Take breaks; accuracy matters more than speed.

| Signer | Tape | Type | Duration (s) | Nominal signs | Locked role if complete |
|---|---|---:|---:|---:|---|
| S3 | `S3_Consonant_Camera.MOV` | consonant | 74.28 | 36 | preferred validation |
| S3 | `S3_all_consonant_Phone_Camera.mov` | consonant | 69.70 | 36 | preferred validation |
| S3 | `S3_all_Consonant_RealWorld_2.mov` | consonant | 53.02 | 36 | preferred validation |
| S3 | `S3_all_Consonant_Real_World_1.MOV` | consonant | 65.88 | 36 | preferred validation |
| S3 | `S3_NSL_Vowel_Prepared_Camera_all.mov` | vowel | 19.82 | 13 | preferred validation |
| S3 | `S3_NSL_Vowel_Unprepared_Realworld_All.mov` | vowel | 19.54 | 13 | preferred validation |
| S4 | `S4_NSL_Consonant_Camera.MOV` | consonant | 64.14 | 36 | sealed test fallback |
| S4 | `S4_NSL_Consonant_Phone_camera_.mov` | consonant | 58.14 | 36 | sealed test fallback |
| S4 | `S4_NSL_Vowel_Prepared_all_Camera.mov` | vowel | 18.84 | 13 | sealed test fallback |
| S5 | `S5_NSL_Consonant_Camera.MOV` | consonant | 60.48 | 36 | preferred sealed test |
| S5 | `S5_NSL_Vowel_Prepared_all_Camera.mov` | vowel | 19.74 | 13 | preferred sealed test |
| S6 | `S6_NSL_Consonant_Camera.MOV` | consonant | 60.42 | 36 | sealed test fallback |
| S6 | `S6_NSL_Consonant_PnoneCamera.mov` | consonant | 60.04 | 36 | sealed test fallback |
| S6 | `S6_NSL_Consonant_Unprepared.mov` | consonant | 39.59 | 36 | sealed test fallback |
| S6 | `S6_NSL_Vowel_Prepared_all_Phone.mov` | vowel | 18.47 | 13 | sealed test fallback |
| S6 | `S6_NSL_Vowel_Prepared_all_camera.mov` | vowel | 25.72 | 13 | sealed test fallback |
| S6 | `S6_NSL_Vowel_All_Unprepared.mov` | vowel | 14.45 | 13 | sealed test fallback |
| S9 | `S9_NSL_ALL_Consonant.mov` | consonant | 268.77 | 36 | development augmentation |
| S13 | `S13_NSL_All_Consonant.mov` | consonant | 252.57 | 36 | development augmentation |
| S14 | `S14_NSL_Consonant.mov` | consonant | 47.57 | 36 | preferred sealed test |
| S14 | `S14_NSL_Vowel_Unprepared_all_phone.mov` | vowel | 16.73 | 13 | preferred sealed test |

The canonical file paths and full metadata are in `annotations/v2/tapes.csv`.

## Decision definitions

- **Accept:** the complete, clearly identifiable target sign is visible. Correct the start and end first, select the class, then accept.
- **Reject:** the window is a false detector proposal, transition, duplicate, idle motion, or contains no target sign.
- **Incomplete:** a real target sign appears, but the complete gesture cannot be recovered from the source tape—for example, it begins before recording or is cut off.
- **Ambiguous:** the complete gesture is visible, but you cannot confidently choose one class. Do not use this merely because the proposed bounds are poor; correct recoverable bounds.

Boundary policy: start at the first intentional movement into the sign; include the full handshape and motion; end after the final intentional hold or motion and before the next transition. Dynamic vowels must include their motion. Sweep each full tape once and use **Insert missed sign** if a real sign has no proposal.

## Calibration first

Both reviewers should separately complete the disposable 24-proposal S3 pilot. Compare the pilot agreement report and discuss only the written decision/boundary policy—not the real annotations. Do not merge pilot decisions into the real file.

```bash
python scripts/make_v2_review_pilot.py
python scripts/review_aggregates.py --mode pass1 --reviewer "REVIEWER A" --segments annotations/v2/pilot_segments.csv --port 8765
python scripts/review_aggregates.py --mode pass2 --reviewer "REVIEWER B" --segments annotations/v2/pilot_segments.csv --port 8765
python scripts/report_v2_annotation_agreement.py --segments annotations/v2/pilot_segments.csv --output annotations/v2/pilot_agreement.json
```

## Safe outside-review workflow

The review server is local-only (`127.0.0.1`) and has no internet authentication. **Do not expose port 8765 publicly.** The outside reviewer should run it on their own computer.

1. The project owner completes pass 1 locally.
2. The owner runs `python scripts/export_v2_pass2.py`. Export refuses an incomplete pass 1 and creates a metadata-only package under `artifacts/reviewer_packages/pass2/`.
3. The exported CSV contains the original computer proposals but removes every pass-1 decision, class, boundary, tape-order hint, adjudication, and final inclusion field.
4. The reviewer obtains only the 21 licensed source tapes. After downloading the four official Zenodo ZIPs, `python scripts/extract_v2_review_tapes.py --extract` selectively extracts these tapes instead of all 630 videos.
5. The reviewer runs:

```bash
python scripts/review_aggregates.py \
  --mode pass2 \
  --reviewer "FULL REVIEWER NAME" \
  --segments artifacts/reviewer_packages/pass2/pass2_segments.csv \
  --port 8765
```

6. The reviewer returns only `pass2_segments.csv` and the unchanged `pass2_package.json`; videos should not be returned or committed.
7. The owner imports with:

```bash
python scripts/import_v2_pass2.py \
  --returned PATH/TO/RETURNED/pass2_segments.csv \
  --package PATH/TO/RETURNED/pass2_package.json \
  --reviewer "FULL REVIEWER NAME"
```

The importer rejects changed proposal identities or original boundaries, leaked pass-1 fields, a changed class configuration or hierarchy lock, a same-person second pass, invalid classes/bounds, and missing rows. It copies only pass-2 fields. If pass 2 inserted a detector miss, that new row is flagged and must still receive pass 1 before freeze.

## Suggested invitation message

> Namaste. I am preparing a research study on low-resource Nepali Sign Language alphabet recognition using the public NSL23 dataset. I am looking for a person who can reliably recognize all 49 NSL alphabet signs to perform an independent annotation-quality pass. The task uses a local browser tool: watch 21 original videos (22.13 minutes total), review 691 proposed gesture windows, correct boundaries, select the visible class, and flag false, incomplete, ambiguous, or missed signs. I estimate 5–8 focused hours plus a short calibration pilot. No model training or coding is required. Your name will be acknowledged in the paper, subject to your consent; the videos remain under the dataset’s CC BY 4.0 license and must not be redistributed beyond the review work. Would you be interested, and could you first complete a short calibration session?

Before work starts, agree in writing on timing, compensation or volunteer status, confidentiality/data handling, how the acknowledgment name should appear, and permission to report anonymous inter-reviewer agreement. “Nonprofit” does not mean unpaid; avoid pressuring a volunteer and offer fair compensation if possible.

## Completion and return checklist

- Every proposal has one pass-2 decision, including rejections.
- Accepted rows have a class and corrected complete-gesture boundaries.
- Each full source tape was swept for missed signs.
- Uncertainty was recorded as incomplete or ambiguous instead of guessed.
- The reviewer used their real agreed name consistently.
- The returned package contains no source videos and no edits to `pass2_package.json`.

# Version 2 aggregate annotation

Motion proposals are navigation only. Dynamic vowels may live between holds. The accepted interval is the complete gesture, including motion. `include_in_v2` stays false until pass 2. Automatic class guesses are never labels.

1. `python scripts/list_v2_tapes.py`
2. `python scripts/propose_aggregate_segments.py --from-cache` — all detected holds kept; `class` is empty; `tape_outcomes.csv` records expected vs detected counts, including incomplete tapes.
3. `python scripts/migrate_v2_segment_schema.py` — only if an older CSV exists.
4. `python scripts/review_aggregates.py --port 8765` — one proposal at a time: full-tape timeline, ±3 s video (mandatory for vowels), previous / expected / next class, frame and time bounds, accept / reject / incomplete / ambiguous, insert missed signs. Pass 2 hides prior accept/reject. S5, S14, S4, and S6 are marked as the test hierarchy.
5. `python scripts/report_v2_coverage.py` — then **stop**. Do not train.
6. `python scripts/freeze_v2_protocol.py` — only after pass 2. Writes hashed `verified_segments_v2.0.csv`, `coverage_v2.0.csv`, and `configs/v2_protocol.json`.

Each accepted row stores at least: `segment_id`, `source_video`, `source_video_hash`, `signer`, `class`, `start_frame`, `end_frame`, `start_time_s`, `end_time_s`, `fps`, `condition`, `capture_device`, `proposal_method`, `review_pass_1`, `review_pass_2`, `include_in_v2`, `exclusion_reason`.

**S4 and S6 are held_as_fallback until freeze.** They are not development. Preferred testers are S5 and S14; fallback is S4 then S6, completeness only.

Stage 2 (after freeze) retrains the same prototypical xyz+angle model and evaluates **S1/S2**, comparable to 59.4%. Do not score frozen testers until Stage 4.

Do not commit unreviewed `segments.csv`. After freeze, commit the v2.0 verified files and protocol JSON.

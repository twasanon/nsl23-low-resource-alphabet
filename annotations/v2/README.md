# Version 2 aggregate annotation

Read [`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md) before opening the review interface.
Use [`REVIEWER_HANDOFF.md`](REVIEWER_HANDOFF.md) when preparing instructions or files for an outside reviewer.

Motion proposals are not labels. Pass 1 may show a non-authoritative tape-order hint. Pass 2 starts from the original proposal and hides pass-1 outputs and the hint. Every proposal requires two different named, NSL-knowledgeable reviewers.

```bash
python scripts/make_v2_review_pilot.py
python scripts/review_aggregates.py --mode pass1 --reviewer "REVIEWER A" --segments annotations/v2/pilot_segments.csv --port 8765
python scripts/review_aggregates.py --mode pass2 --reviewer "REVIEWER B" --segments annotations/v2/pilot_segments.csv --port 8765
python scripts/report_v2_annotation_agreement.py --segments annotations/v2/pilot_segments.csv --output annotations/v2/pilot_agreement.json

python scripts/review_aggregates.py --mode pass1 --reviewer "REVIEWER A" --port 8765
python scripts/review_aggregates.py --mode pass2 --reviewer "REVIEWER B" --port 8765
python scripts/report_v2_annotation_agreement.py --output annotations/v2/agreement_pre_adjudication.json
python scripts/review_aggregates.py --mode adjudicate --reviewer "ADJUDICATOR" --port 8765
python scripts/report_v2_coverage.py
python scripts/freeze_v2_protocol.py
```

The reviewer writes atomically and keeps `segments.csv.bak`. Never run two reviewer servers against the same CSV simultaneously. Do not use `configs/signer_split.json`, and do not train until freeze.

Current hierarchy lock: `09a2b7b2cb3a580c6cf3e3cd0c952a04ec4a793863ab27212c0cdeb55c5ab009`.

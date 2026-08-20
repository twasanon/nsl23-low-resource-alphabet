# NSL23 Version 2 manual review guide

## The short answer

Yes, every proposed gesture segment must be checked by a human before Version 2 can use it. A proposal is only a computer-generated suggestion about where a gesture might be; it is not a label.

You should personally label clips **only if you can reliably identify the 49 Nepali Sign Language alphabet classes in video**. If you cannot, your job is to organize the process and recruit qualified NSL reviewers. Do not guess from the filename, proposal number, or expected tape order. An AI agent can maintain the interface and validators, but it cannot replace the NSL ground-truth decision.

## What each reviewer does

There are three roles. Run one role at a time. Pass 1 and pass 2 must be two different people on every tape; otherwise the agreement statistic is not independent.

1. **Pass 1 reviewer:** watches each proposed window, chooses the visible NSL class, corrects the start/end, and accepts or rejects it. The interface shows a tape-order hint, but the hint may be shifted because the motion detector found too many or too few proposals. It is never ground truth.
2. **Pass 2 reviewer:** starts again from the original computer proposal. Pass-1 decisions, labels, bounds, and the tape-order class hint are hidden. This reviewer makes a fresh judgment.
3. **Adjudicator:** resolves completed-pass disagreements about accept/reject, class, or temporal bounds. The adjudicator records the final decision only after the pre-adjudication agreement report has been saved. A third qualified person is preferable.

For every proposal, pass 1 and pass 2 must be two different named, NSL-knowledgeable people. This is enforced at freeze. S5, S14, S4, and S6 have the additional restriction that their data can never enter development.

## Boundary rule

- Start at the first intentional movement into the target sign.
- Include the complete handshape or motion. Dynamic vowels must include their motion, not only a still hold.
- End after the final intentional hold or motion and before the transition into the next sign.
- Reject detector fragments, transitions, duplicate false proposals, or gestures whose identity cannot be determined.
- Use **Insert missed sign** when a real sign has no proposal. An inserted sign still needs both passes.
- Sweep the full source tape during each pass; do not inspect only the short proposal windows.
- Never force a tape to contain exactly 13 vowels or 36 consonants. Record what is genuinely visible.

A sign manually inserted during pass 1 is shown to pass 2 as a new proposal with its class hidden. That gives it independent second verification, but not independent discovery. The `manual_insert` flag is retained and must be reported; do not describe such rows as independently discovered by both reviewers.

## Calibration before the real review

Create a disposable 24-proposal S3 pilot:

```bash
python scripts/make_v2_review_pilot.py
python scripts/review_aggregates.py --mode pass1 --reviewer "REVIEWER A" --segments annotations/v2/pilot_segments.csv --port 8765
python scripts/review_aggregates.py --mode pass2 --reviewer "REVIEWER B" --segments annotations/v2/pilot_segments.csv --port 8765
python scripts/report_v2_annotation_agreement.py --segments annotations/v2/pilot_segments.csv --output annotations/v2/pilot_agreement.json
```

Reviewers should compare the pilot report, discuss the boundary rule, and resolve misunderstandings. **Do not merge pilot decisions into the real `segments.csv`.** If the written policy changes, update the protocol and hierarchy hash before beginning the real review.

## Real review commands

Pass 1:

```bash
python scripts/review_aggregates.py --mode pass1 --reviewer "REVIEWER A" --port 8765
```

Stop that server when pass 1 is complete. A different person then starts pass 2:

```bash
python scripts/review_aggregates.py --mode pass2 --reviewer "REVIEWER B" --port 8765
```

The tool records only the selected role. It writes atomically and keeps the immediately previous file as `annotations/v2/segments.csv.bak`. Do not run two review servers against the same CSV simultaneously.

Before adjudication, preserve the independent agreement results. This command refuses incomplete passes or existing adjudication:

```bash
python scripts/report_v2_annotation_agreement.py --output annotations/v2/agreement_pre_adjudication.json
```

After this report exists, do not change either pass record. Its pass-only hash is checked at freeze.

Then run adjudication if disagreements exist:

```bash
python scripts/review_aggregates.py --mode adjudicate --reviewer "ADJUDICATOR" --port 8765
```

Finally:

```bash
python scripts/report_v2_coverage.py
python scripts/freeze_v2_protocol.py
```

Freeze verifies that every proposal completed both passes, checks the pre-adjudication report, recomputes every used source-video hash, and validates identities, UTC timestamps, classes, bounds, frame/time consistency, duration, overlap, coverage, and the locked hierarchy. It creates a frozen source inventory so legacy or mismatched feature caches cannot enter training. No training is permitted before freeze.

## What “done” means

- Every proposal—including rejected proposals—has two completed human passes.
- Every proposal has two different named reviewers.
- Pre-adjudication agreement has been saved.
- Adjudications have named reviewers and timestamps.
- Coverage identifies at least two complete frozen testers for a Stage 4 evaluation.
- If fewer than two testers are complete, the annotations may still be published as a resource, but there is no confirmatory 49-way Stage 4 claim.

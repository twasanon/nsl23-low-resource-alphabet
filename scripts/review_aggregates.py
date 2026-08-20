#!/usr/bin/env python3
"""Video-first two-pass review for Version 2 aggregate segments.

One proposal at a time: full-tape timeline, ±3 s playback, editable start/end
in seconds and frames, class dropdown, accept / reject / incomplete / ambiguous.

Pass 2 hides pass-1 accept/reject. Test-hierarchy signers (S5, S14, S4, S6) get the strictest review.
include_in_v2 is true only after pass-2 accept with a class label.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs, quote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nsl23.v2_annotation import (
    SEGMENT_FIELDS,
    TEST_HIERARCHY,
    apply_include_rule,
    empty_segment,
    migrate_row,
    needs_adjudication,
    neutralize_unreviewed,
    segment_id as make_segment_id,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "annotations/v2/segments.csv"
TAPES = ROOT / "annotations/v2/tapes.csv"
WINDOWS = ROOT / "artifacts/v2_review/windows"
CLASSES = json.loads((ROOT / "configs/nsl23_classes.json").read_text(encoding="utf-8"))
FROZEN = set(TEST_HIERARCHY)
PAD = 3.0
STATE: dict[str, object] = {
    "reviewer": "",
    "mode": "pass1",
    "segments": SEGMENTS,
    "tapes": TAPES,
}
CSV_LOCK = RLock()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load() -> list[dict[str, str]]:
    with CSV_LOCK:
        hashes: dict[str, str] = {}
        path = Path(STATE["segments"])
        with path.open(newline="", encoding="utf-8") as handle:
            raw = list(csv.DictReader(handle))
        return [neutralize_unreviewed(migrate_row(row, hashes)) for row in raw]


def save(rows: list[dict[str, str]]) -> None:
    with CSV_LOCK:
        write_csv(Path(STATE["segments"]), SEGMENT_FIELDS, rows, backup=True)


def load_tapes() -> dict[str, dict[str, str]]:
    with Path(STATE["tapes"]).open(newline="", encoding="utf-8") as handle:
        return {row["video_id"]: row for row in csv.DictReader(handle)}


def alphabet(sign_type: str) -> list[str]:
    return CLASSES["vowels"] if sign_type == "vowel" else CLASSES["consonants"]


def neighbors(sign_type: str, hint: str) -> tuple[str, str, str]:
    labels = alphabet(sign_type)
    if hint in labels:
        index = labels.index(hint)
        prev_c = labels[index - 1] if index else ""
        next_c = labels[index + 1] if index + 1 < len(labels) else ""
        return prev_c, hint, next_c
    return "", hint, ""


def tape_rows(rows: list[dict[str, str]], video_id: str, pass_n: str = "1") -> list[dict[str, str]]:
    subset = [row for row in rows if row["source_video"] == video_id]
    time_field = "proposal_start_time_s" if pass_n == "2" else "start_time_s"
    return sorted(
        subset,
        key=lambda row: (
            float(row.get(time_field) or row.get("start_time_s") or 0),
            int(row["hold_index"] or 0),
        ),
    )


def first_unlabeled(rows: list[dict[str, str]], pass_n: str) -> int:
    field = "review_pass_1" if pass_n != "2" else "review_pass_2"
    for index, row in enumerate(rows):
        if not row.get(field):
            return index
    return 0


def page(title: str, body: str) -> bytes:
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: sans-serif; margin: 16px auto; max-width: 980px; }}
a {{ color: #0b57d0; }}
.card {{ border: 1px solid #ccc; padding: 12px 14px; margin: 12px 0; }}
.banner {{ background: #fff3cd; border: 1px solid #e0c36a; padding: 10px 12px; margin: 12px 0; }}
.strict {{ background: #f8d7da; border-color: #c97a80; }}
.error {{ background: #f8d7da; border: 1px solid #c97a80; padding: 10px 12px; }}
video {{ width: 100%; background: #000; max-height: 420px; }}
label {{ display: inline-block; margin: 4px 8px 4px 0; }}
input[type=number] {{ width: 7.5em; }}
select {{ min-width: 8em; }}
button {{ margin: 4px 6px 4px 0; padding: 6px 10px; }}
.muted {{ color: #555; font-size: 14px; }}
.classes {{ font-size: 22px; margin: 8px 0 14px; letter-spacing: 0.04em; }}
.classes .cur {{ font-weight: 700; padding: 2px 8px; background: #e8f0fe; border-radius: 4px; }}
.timeline {{ position: relative; height: 28px; background: #ececec; border: 1px solid #bbb; margin: 8px 0 4px; }}
.timeline .span {{ position: absolute; top: 0; bottom: 0; background: #8ab4f8; }}
.timeline .win {{ position: absolute; top: 0; bottom: 0; background: #000; opacity: 0.08; }}
.nav {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
</style>
<h1>{html.escape(title)}</h1>
{body}
"""
    return doc.encode("utf-8")


def window_mp4(row: dict[str, str], start: float | None = None, end: float | None = None, pad: float = PAD) -> Path:
    WINDOWS.mkdir(parents=True, exist_ok=True)
    start_t = float(row["start_time_s"] if start is None else start)
    end_t = float(row["end_time_s"] if end is None else end)
    win0 = max(0.0, start_t - pad)
    win1 = end_t + pad
    duration = max(win1 - win0, 0.4)
    out = WINDOWS / f"{row['segment_id']}_{win0:.2f}_{win1:.2f}.mp4"
    if out.exists() and out.stat().st_size > 1000:
        return out
    src = Path(row["source_path"])
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{win0:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        "-y",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def send_bytes(handler: BaseHTTPRequestHandler, data: bytes, content_type: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def send_file_range(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    size = path.stat().st_size
    rng = handler.headers.get("Range")
    if not rng:
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(size))
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1 << 16)
                if not chunk:
                    break
                handler.wfile.write(chunk)
        return
    _, _, spec = rng.partition("=")
    start_s, _, end_s = spec.partition("-")
    start = int(start_s or 0)
    end = int(end_s) if end_s else size - 1
    end = min(end, size - 1)
    length = end - start + 1
    handler.send_response(206)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.send_header("Content-Length", str(length))
    handler.send_header("Accept-Ranges", "bytes")
    handler.end_headers()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(1 << 16, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def class_select(sign_type: str, current: str, hint: str) -> str:
    del hint  # shown separately; do not pre-select a guess
    chosen = current
    options = ['<option value="">(unset)</option>']
    for label in alphabet(sign_type):
        selected = " selected" if label == chosen else ""
        options.append(f'<option value="{html.escape(label)}"{selected}>{html.escape(label)}</option>')
    return f'<select name="klass">{"".join(options)}</select>'


def index_body(rows: list[dict[str, str]]) -> str:
    tapes = load_tapes()
    mode = str(STATE["mode"])
    pass_n = "1" if mode == "pass1" else "2"
    items = [
        f"<p>Mode: <strong>{html.escape(mode)}</strong>. Reviewer: "
        f"<strong>{html.escape(str(STATE['reviewer']) or '(set --reviewer)')}</strong>. "
        "This server records only the displayed role. Pass 2 hides pass-1 outputs and the "
        "tape-order class hint. Nothing enters Version 2 until both passes agree or a later "
        "adjudication is recorded.</p>"
    ]
    video_ids = sorted({row["source_video"] for row in rows}, key=lambda vid: (tapes.get(vid, {}).get("signer", ""), vid))
    for video_id in video_ids:
        subset = tape_rows(rows, video_id, pass_n)
        signer = subset[0]["signer"]
        n1 = sum(row["review_pass_1"] == "accepted" for row in subset)
        n2 = sum(row["review_pass_2"] == "accepted" for row in subset)
        u1 = first_unlabeled(subset, "1")
        u2 = first_unlabeled(subset, "2")
        tag = " · TEST HIERARCHY" if signer in FROZEN else ""
        cls = " card strict" if signer in FROZEN else " card"
        target = u1 if pass_n == "1" else u2
        items.append(
            f'<div class="{cls.strip()}">'
            f'<a href="/tape?video={html.escape(video_id)}&i={target}">Open {html.escape(mode)}</a>{tag}<br>'
            f"{html.escape(video_id)}"
            f'<div class="muted">{html.escape(signer)} · {subset[0]["sign_type"]} · '
            f"{len(subset)} proposals · this pass accepted {n1 if pass_n == '1' else n2}</div></div>"
        )
    return "".join(items)


def review_body(
    video_id: str,
    pass_n: str,
    index: int,
    rows: list[dict[str, str]],
    error: str = "",
) -> str:
    tapes = load_tapes()
    row = rows[index]
    signer = row["signer"]
    duration = float(tapes.get(video_id, {}).get("duration_seconds") or 0.0)
    if duration <= 0:
        duration = max(float(row["end_time_s"]), 1.0) + 1.0
    fps = float(row["fps"] or 30)
    independent = pass_n == "2"
    if independent:
        if row.get("review_pass_2"):
            start = float(row.get("pass2_start_time_s") or row["proposal_start_time_s"])
            end = float(row.get("pass2_end_time_s") or row["proposal_end_time_s"])
            start_f = int(row.get("pass2_start_frame") or round(start * fps))
            end_f = int(row.get("pass2_end_frame") or round(end * fps))
            chosen_class = row.get("pass2_class") or ""
        else:
            start = float(row.get("proposal_start_time_s") or row["start_time_s"])
            end = float(row.get("proposal_end_time_s") or row["end_time_s"])
            start_f = int(row.get("proposal_start_frame") or round(start * fps))
            end_f = int(row.get("proposal_end_frame") or round(end * fps))
            chosen_class = ""
    else:
        start = float(row["start_time_s"])
        end = float(row["end_time_s"])
        start_f = int(row["start_frame"] or round(start * fps))
        end_f = int(row["end_frame"] or round(end * fps))
        chosen_class = row["class"]
    pad = PAD
    win0 = max(0.0, start - pad)
    win1 = min(duration, end + pad)
    if independent:
        prev_c, hint, next_c = "", "", ""
    else:
        prev_c, hint, next_c = neighbors(row["sign_type"], row["expected_class_hint"])
    left = (start / duration) * 100 if duration else 0
    width = max(((end - start) / duration) * 100, 0.4) if duration else 0
    wleft = (win0 / duration) * 100 if duration else 0
    wwidth = ((win1 - win0) / duration) * 100 if duration else 0
    banner = []
    banner.append(
        '<div class="banner">Pass 1 and pass 2 must be completed by two different named, '
        "NSL-knowledgeable reviewers. Freeze rejects same-person double review.</div>"
    )
    if signer in FROZEN:
        banner.append(
            '<div class="banner strict">Test-hierarchy signer. Pass 1 and pass 2 must be different '
            "NSL-knowledgeable reviewers, and this signer remains sealed from development.</div>"
        )
    if row["sign_type"] == "vowel":
        banner.append(
            '<div class="banner">Dynamic vowel: play the window. The sign may sit between holds. '
            "Accept the complete gesture, including motion, not only the longest still section.</div>"
        )
    if pass_n == "2":
        banner.append(
            '<div class="banner">Independent pass 2: pass-1 class and bounds are hidden. '
            "You are labeling from the original proposal window. The tape-order class hint is also hidden.</div>"
        )
        if row.get("proposal_method") == "manual_insert":
            banner.append(
                '<div class="banner">This proposal was a sign missed by the motion detector and '
                "manually inserted during pass 1. Its existence and initial window are therefore "
                "second-verified here, not independently discovered. This status is retained for reporting.</div>"
            )
    elif row["sign_type"] != "vowel":
        banner.append(
            '<div class="banner">Consonant contact sheets are optional. Video playback is still available. '
            "Adjust start/end to the full sign.</div>"
        )
    if error:
        banner.append(f'<div class="error">{html.escape(error)}</div>')
    sid = html.escape(row["segment_id"])
    prev_i = max(0, index - 1)
    next_i = min(len(rows) - 1, index + 1)
    unlabeled = first_unlabeled(rows, pass_n)
    class_line = ""
    if not independent:
        class_line = (
            f'<div class="classes">'
            f'<span class="muted">order hint — prev</span> {html.escape(prev_c or "—")} '
            f'<span class="cur">{html.escape(hint or "(no hint)")}</span> '
            f'<span class="muted">next</span> {html.escape(next_c or "—")}'
            f"</div>"
        )
    return f"""
<p><a href="/">All tapes</a> · {html.escape(signer)} · {html.escape(row["sign_type"])} ·
proposal {index + 1} / {len(rows)} · hold {html.escape(row["hold_index"])} of {html.escape(row["n_holds_detected"])}</p>
{"".join(banner)}
{class_line}
<div class="muted">Full-tape timeline. Blue = proposed bounds. Grey band = ±{pad:.0f}s playback window.</div>
<div class="timeline" id="timeline" data-duration="{duration:.4f}" title="Tape duration {duration:.2f}s">
  <div class="win" style="left:{wleft:.3f}%;width:{wwidth:.3f}%;"></div>
  <div class="span" style="left:{left:.3f}%;width:{width:.3f}%;"></div>
</div>
<div class="muted">tape 0s — {duration:.2f}s · proposed {start:.3f}s–{end:.3f}s · frames {start_f}–{end_f} @ {fps:g} fps</div>
<details class="card">
<summary><strong>Open the full source tape to check for detector misses</strong></summary>
<p class="muted">Sweep the entire tape at least once during each pass. If the browser cannot decode the MOV, use the proposal windows and open the source path in a local video player.</p>
<video controls preload="metadata" src="/source?video={html.escape(quote(video_id, safe=''))}"></video>
</details>
<video id="clip" controls autoplay preload="metadata" src="/window?sid={sid}&amp;start={start:.4f}&amp;end={end:.4f}&amp;pad={pad}"></video>
<form method="post" action="/decide" id="form" onsubmit="if(lastEdited==='frame')syncTimes();else syncFrames();">
<input type="hidden" name="sid" value="{sid}">
<input type="hidden" name="video" value="{html.escape(video_id)}">
<input type="hidden" name="pass" value="{html.escape(pass_n)}">
<input type="hidden" name="index" value="{index}">
<p>
<label>class {class_select(row["sign_type"], chosen_class, hint)}</label>
<label>start s <input type="number" step="0.001" name="start" id="start" value="{start:.4f}"></label>
<label>end s <input type="number" step="0.001" name="end" id="end" value="{end:.4f}"></label>
<label>start frame <input type="number" step="1" name="start_frame" id="start_frame" value="{start_f}"></label>
<label>end frame <input type="number" step="1" name="end_frame" id="end_frame" value="{end_f}"></label>
</p>
<p>
<button type="button" onclick="playWin()">Play window</button>
<button type="button" onclick="mark('start')">Set start to playhead</button>
<button type="button" onclick="mark('end')">Set end to playhead</button>
<button type="button" onclick="reloadWin()">Reload window around bounds</button>
</p>
<p>
<button name="action" value="accepted">Accept</button>
<button name="action" value="rejected">Reject</button>
<button name="action" value="incomplete">Incomplete</button>
<button name="action" value="ambiguous">Ambiguous</button>
</p>
</form>
<div class="nav">
  <a href="/tape?video={html.escape(video_id)}&amp;pass={html.escape(pass_n)}&amp;i={prev_i}">Previous</a>
  <a href="/tape?video={html.escape(video_id)}&amp;pass={html.escape(pass_n)}&amp;i={unlabeled}">Next unlabeled</a>
  <a href="/tape?video={html.escape(video_id)}&amp;pass={html.escape(pass_n)}&amp;i={next_i}">Next</a>
</div>
<div class="card">
<strong>Insert a missed sign</strong>
<div class="muted">Use this when a detector hold is missing (for example S14 consonants 27/36). Bounds are the complete gesture, including motion.</div>
<form method="post" action="/insert">
<input type="hidden" name="video" value="{html.escape(video_id)}">
<input type="hidden" name="pass" value="{html.escape(pass_n)}">
<label>class {class_select(row["sign_type"], "", "")}</label>
<label>start s <input type="number" step="0.001" name="start" required></label>
<label>end s <input type="number" step="0.001" name="end" required></label>
<button name="action" value="insert">Insert unlabeled segment</button>
</form>
</div>
<script>
const fps = {fps};
const pad = {pad};
const win0 = {win0};
let lastEdited = 'time';
function playWin(){{
  const v = document.getElementById('clip');
  const s = parseFloat(document.getElementById('start').value);
  const e = parseFloat(document.getElementById('end').value);
  v.currentTime = Math.max(0, s - win0);
  v.play();
  const stop = () => {{
    if (v.currentTime >= (e - win0)) {{
      v.pause();
      v.removeEventListener('timeupdate', stop);
    }}
  }};
  v.addEventListener('timeupdate', stop);
}}
function mark(which){{
  const v = document.getElementById('clip');
  const t = (win0 + v.currentTime).toFixed(3);
  document.getElementById(which).value = t;
  lastEdited = 'time';
  syncFrames();
}}
function syncFrames(){{
  const s = parseFloat(document.getElementById('start').value);
  const e = parseFloat(document.getElementById('end').value);
  document.getElementById('start_frame').value = Math.round(s * fps);
  document.getElementById('end_frame').value = Math.max(parseInt(document.getElementById('start_frame').value, 10) + 1, Math.round(e * fps));
}}
function syncTimes(){{
  document.getElementById('start').value = (parseInt(document.getElementById('start_frame').value, 10) / fps).toFixed(3);
  document.getElementById('end').value = (parseInt(document.getElementById('end_frame').value, 10) / fps).toFixed(3);
}}
function reloadWin(){{
  const s = document.getElementById('start').value;
  const e = document.getElementById('end').value;
  const v = document.getElementById('clip');
  v.src = '/window?sid={sid}&start=' + s + '&end=' + e + '&pad=' + pad + '&t=' + Date.now();
  v.play();
}}
document.getElementById('start').addEventListener('change', () => {{ lastEdited = 'time'; syncFrames(); }});
document.getElementById('end').addEventListener('change', () => {{ lastEdited = 'time'; syncFrames(); }});
document.getElementById('start_frame').addEventListener('change', () => {{ lastEdited = 'frame'; syncTimes(); }});
document.getElementById('end_frame').addEventListener('change', () => {{ lastEdited = 'frame'; syncTimes(); }});
playWin();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print(self.address_string(), format % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/":
            if STATE["mode"] == "adjudicate":
                self.send_response(303)
                self.send_header("Location", "/adjudicate")
                self.end_headers()
                return
            send_bytes(self, page("NSL23 V2 segment review", index_body(load())), "text/html; charset=utf-8")
            return
        if parsed.path == "/tape":
            if STATE["mode"] == "adjudicate":
                self.send_error(403, "This server is in adjudication mode")
                return
            video_id = qs.get("video", [""])[0]
            pass_n = "1" if STATE["mode"] == "pass1" else "2"
            rows = tape_rows(load(), video_id, pass_n)
            if not rows:
                self.send_error(404)
                return
            index = int(qs.get("i", [str(first_unlabeled(rows, pass_n))])[0] or 0)
            index = min(max(index, 0), len(rows) - 1)
            error = qs.get("error", [""])[0]
            body = review_body(video_id, pass_n, index, rows, error=error)
            send_bytes(self, page(f"{video_id} pass {pass_n}", body), "text/html; charset=utf-8")
            return
        if parsed.path == "/adjudicate":
            if STATE["mode"] != "adjudicate":
                self.send_error(403, "Restart with --mode adjudicate")
                return
            rows = [
                row
                for row in load()
                if needs_adjudication(row) and not row.get("adjudication_decision")
            ]
            blocks = [
                "<p>Both passes are complete, but their decisions, classes, or bounds disagree. "
                "Adjudication cannot replace a missing second pass. Record the final class and bounds "
                "for an acceptance, or reject the segment.</p>"
            ]
            if not rows:
                blocks.append("<p>No disagreements right now.</p>")
            for row in rows:
                sid = html.escape(row["segment_id"])
                p1 = (
                    f"{html.escape(row.get('pass1_reviewer', ''))} "
                    f"{html.escape(row.get('review_pass_1', ''))} "
                    f"{html.escape(row.get('pass1_class', ''))} "
                    f"{html.escape(row.get('pass1_start_time_s', ''))}–"
                    f"{html.escape(row.get('pass1_end_time_s', ''))}"
                )
                p2 = (
                    f"{html.escape(row.get('pass2_reviewer', ''))} "
                    f"{html.escape(row.get('review_pass_2', ''))} "
                    f"{html.escape(row.get('pass2_class', ''))} "
                    f"{html.escape(row.get('pass2_start_time_s', ''))}–"
                    f"{html.escape(row.get('pass2_end_time_s', ''))}"
                )
                blocks.append(
                    f'<div class="card"><strong>{html.escape(row["source_video"])}</strong> {sid}'
                    f'<div class="muted">pass1 {p1}</div>'
                    f'<div class="muted">pass2 {p2}</div>'
                    f'<div class="muted">{html.escape(row.get("exclusion_reason", ""))}</div>'
                    f'<form method="post" action="/adjudicate">'
                    f'<input type="hidden" name="sid" value="{sid}">'
                    f'<label>class {class_select(row["sign_type"], row.get("pass2_class") or "", "")}</label>'
                    f'<label>start s <input type="number" step="0.001" name="start" value="{html.escape(row.get("pass2_start_time_s") or "")}"></label>'
                    f'<label>end s <input type="number" step="0.001" name="end" value="{html.escape(row.get("pass2_end_time_s") or "")}"></label>'
                    f'<button name="action" value="accepted">Adjudicate accept</button>'
                    f'<button name="action" value="rejected">Adjudicate reject</button>'
                    f"</form></div>"
                )
            send_bytes(self, page("Adjudicate disagreements", "".join(blocks)), "text/html; charset=utf-8")
            return
        if parsed.path == "/window":
            sid = qs.get("sid", [""])[0]
            rows = [row for row in load() if row["segment_id"] == sid]
            if not rows:
                self.send_error(404)
                return
            start = float(qs["start"][0]) if qs.get("start") else None
            end = float(qs["end"][0]) if qs.get("end") else None
            pad = float(qs.get("pad", [str(PAD)])[0])
            try:
                path = window_mp4(rows[0], start=start, end=end, pad=pad)
            except subprocess.CalledProcessError:
                self.send_error(500, "ffmpeg failed")
                return
            send_file_range(self, path, "video/mp4")
            return
        if parsed.path == "/source":
            video_id = qs.get("video", [""])[0]
            tape = load_tapes().get(video_id)
            if not tape:
                self.send_error(404)
                return
            path = Path(tape["source_path"])
            if not path.is_file():
                self.send_error(404, "Source tape is missing")
                return
            content_type = "video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4"
            send_file_range(self, path, content_type)
            return
        self.send_error(404)

    def _insert(self, form: dict[str, list[str]]) -> None:
        video_id = form.get("video", [""])[0]
        pass_n = "1" if STATE["mode"] == "pass1" else "2"
        start = form.get("start", [""])[0].strip()
        end = form.get("end", [""])[0].strip()
        klass = form.get("klass", [""])[0].strip()
        rows = load()
        subset = [row for row in rows if row["source_video"] == video_id]
        if not subset or not start or not end:
            self.send_response(303)
            self.send_header("Location", f"/tape?video={video_id}&pass={pass_n}&i=0&error=Insert+needs+start+and+end")
            self.end_headers()
            return
        template = subset[0]
        fps = float(template["fps"] or 30)
        hold_index = max(int(row["hold_index"] or 0) for row in subset) + 1
        start_f = int(round(float(start) * fps))
        end_f = max(start_f + 1, int(round(float(end) * fps)))
        rec = empty_segment()
        rec.update(
            {
                "segment_id": make_segment_id(video_id, hold_index),
                "source_video": video_id,
                "source_video_hash": template["source_video_hash"],
                "signer": template["signer"],
                "class": "",
                "start_frame": str(start_f),
                "end_frame": str(end_f),
                "start_time_s": f"{float(start):.4f}",
                "end_time_s": f"{float(end):.4f}",
                "fps": template["fps"],
                "duration_seconds": template.get("duration_seconds", ""),
                "condition": template["condition"],
                "lighting": template["lighting"],
                "capture_device": template["capture_device"],
                "sign_type": template["sign_type"],
                "proposal_method": "manual_insert",
                "hold_index": str(hold_index),
                "n_holds_detected": template["n_holds_detected"],
                "expected_class_hint": klass,
                "proposal_start_frame": str(start_f),
                "proposal_end_frame": str(end_f),
                "proposal_start_time_s": f"{float(start):.4f}",
                "proposal_end_time_s": f"{float(end):.4f}",
                "include_in_v2": "false",
                "notes": "manually inserted; detector did not propose a hold",
                "source_path": template["source_path"],
            }
        )
        rows.append(rec)
        save(rows)
        ordered = tape_rows(rows, video_id, pass_n)
        new_id = make_segment_id(video_id, hold_index)
        index = next((i for i, row in enumerate(ordered) if row["segment_id"] == new_id), len(ordered) - 1)
        self.send_response(303)
        self.send_header("Location", f"/tape?video={video_id}&pass={pass_n}&i={index}")
        self.end_headers()

    def _adjudicate(self, form: dict[str, list[str]]) -> None:
        if STATE["mode"] != "adjudicate":
            self.send_response(303)
            self.send_header("Location", "/adjudicate")
            self.end_headers()
            return
        sid = form.get("sid", [""])[0]
        action = form.get("action", [""])[0]
        klass = form.get("klass", [""])[0].strip()
        start = form.get("start", [""])[0].strip()
        end = form.get("end", [""])[0].strip()
        if action not in {"accepted", "rejected"}:
            self.send_error(400, "Adjudication decision must be accepted or rejected")
            return
        if action == "accepted" and (not klass or not start or not end):
            self.send_error(400, "Adjudication acceptance requires class, start, and end")
            return
        if start and end and float(end) <= float(start):
            self.send_error(400, "Adjudication end must be after start")
            return
        rows = load()
        for row in rows:
            if row["segment_id"] != sid:
                continue
            fps = float(row["fps"] or 30)
            row["adjudication_reviewer"] = str(STATE["reviewer"])
            row["adjudication_timestamp"] = now_utc()
            row["adjudication_decision"] = action
            row["adjudication_class"] = klass if action == "accepted" else ""
            if start:
                row["adjudication_start_time_s"] = start
                row["adjudication_start_frame"] = str(int(round(float(start) * fps)))
            if end:
                row["adjudication_end_time_s"] = end
                row["adjudication_end_frame"] = str(max(int(row["adjudication_start_frame"] or 0) + 1, int(round(float(end) * fps))))
            if action == "rejected":
                for field in (
                    "adjudication_start_time_s",
                    "adjudication_end_time_s",
                    "adjudication_start_frame",
                    "adjudication_end_frame",
                ):
                    row[field] = ""
            apply_include_rule(row)
            break
        save(rows)
        self.send_response(303)
        self.send_header("Location", "/adjudicate")
        self.end_headers()

    def do_POST(self) -> None:
        with CSV_LOCK:
            self._do_POST_locked()

    def _do_POST_locked(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        if parsed.path == "/insert":
            if STATE["mode"] == "adjudicate":
                self.send_error(403)
                return
            self._insert(form)
            return
        if parsed.path == "/adjudicate":
            self._adjudicate(form)
            return
        sid = form.get("sid", [""])[0]
        video_id = form.get("video", [""])[0]
        if STATE["mode"] == "adjudicate":
            self.send_error(403)
            return
        pass_n = "1" if STATE["mode"] == "pass1" else "2"
        index = int(form.get("index", ["0"])[0] or 0)
        action = form.get("action", [""])[0]
        klass = form.get("klass", [""])[0].strip()
        start = form.get("start", [""])[0].strip()
        end = form.get("end", [""])[0].strip()
        start_frame = form.get("start_frame", [""])[0].strip()
        end_frame = form.get("end_frame", [""])[0].strip()
        if action == "accepted" and not klass:
            self.send_response(303)
            self.send_header(
                "Location",
                f"/tape?video={video_id}&pass={pass_n}&i={index}&error=Accept+requires+a+class+label",
            )
            self.end_headers()
            return
        rows = load()
        for row in rows:
            if row["segment_id"] != sid:
                continue
            fps = float(row["fps"] or 30)
            if start_frame:
                start_s = f"{int(start_frame) / fps:.4f}"
                start_f = start_frame
            elif start:
                start_s = start
                start_f = str(int(round(float(start) * fps)))
            else:
                start_s, start_f = row["start_time_s"], row["start_frame"]
            if end_frame:
                end_s = f"{int(end_frame) / fps:.4f}"
                end_f = end_frame
            elif end:
                end_s = end
                end_f = str(max(int(start_f) + 1, int(round(float(end) * fps))))
            else:
                end_s, end_f = row["end_time_s"], row["end_frame"]
            prefix = "pass1" if pass_n != "2" else "pass2"
            row[f"review_pass_{'1' if prefix == 'pass1' else '2'}"] = action
            row[f"{prefix}_reviewer"] = str(STATE["reviewer"])
            row[f"{prefix}_timestamp"] = now_utc()
            row[f"{prefix}_class"] = klass
            row[f"{prefix}_start_frame"] = start_f
            row[f"{prefix}_end_frame"] = end_f
            row[f"{prefix}_start_time_s"] = start_s
            row[f"{prefix}_end_time_s"] = end_s
            if prefix == "pass1":
                row["start_frame"] = start_f
                row["end_frame"] = end_f
                row["start_time_s"] = start_s
                row["end_time_s"] = end_s
                if klass:
                    row["class"] = klass
            if action != "accepted":
                row["exclusion_reason"] = action
            apply_include_rule(row)
            break
        save(rows)
        next_i = min(index + 1, max(0, len(tape_rows(load(), video_id, pass_n)) - 1))
        self.send_response(303)
        self.send_header("Location", f"/tape?video={video_id}&pass={pass_n}&i={next_i}")
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=["pass1", "pass2", "adjudicate"], required=True)
    parser.add_argument("--reviewer", required=True, help="Name recorded for this one role.")
    parser.add_argument("--segments", type=Path, default=SEGMENTS)
    parser.add_argument("--tapes", type=Path, default=TAPES)
    args = parser.parse_args()
    STATE["reviewer"] = args.reviewer.strip()
    STATE["mode"] = args.mode
    STATE["segments"] = args.segments
    STATE["tapes"] = args.tapes
    if not STATE["reviewer"]:
        raise SystemExit("--reviewer is required")
    if not args.segments.exists():
        raise SystemExit(f"Missing {args.segments}")
    if not args.tapes.exists():
        raise SystemExit(f"Missing {args.tapes}")
    hashes: dict[str, str] = {}
    raw = list(csv.DictReader(args.segments.open(newline="", encoding="utf-8")))
    save([neutralize_unreviewed(migrate_row(row, hashes)) for row in raw])
    print(f"Review UI: http://127.0.0.1:{args.port}", flush=True)
    print(
        f"Mode {STATE['mode']}; reviewer {STATE['reviewer']}. "
        "This server cannot record another role. Do not train V2 yet.",
        flush=True,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped cleanly.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

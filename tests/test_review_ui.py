"""Reviewer HTTP checks use disposable CSVs, never the annotation master."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from nsl23.v2_annotation import write_csv, SEGMENT_FIELDS
from test_v2_exchange import master_row

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("review_ui", ROOT / "scripts/review_aggregates.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class ReviewUITests(unittest.TestCase):
    def test_pass2_http_hides_pass1_and_saves_original_proposal_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            row = master_row()
            row["pass1_reviewer"] = "Hidden Reviewer"
            segments, tapes = base / "segments.csv", base / "tapes.csv"
            write_csv(segments, SEGMENT_FIELDS, [row])
            write_csv(tapes, ["video_id", "duration_seconds"],
                      [{"video_id": row["source_video"], "duration_seconds": "10"}])
            original = dict(ui.STATE)
            ui.STATE.update(mode="pass2", reviewer="Independent Reviewer", segments=segments, tapes=tapes)
            server = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            address = f"http://127.0.0.1:{server.server_port}"
            try:
                html = urlopen(address + "/tape?" + urlencode({"video": row["source_video"]})).read().decode()
                self.assertNotIn("Hidden Reviewer", html)
                self.assertNotIn('<option value="KA" selected', html)
                self.assertIn('value="0.8000"', html)
                self.assertIn('value="2.2000"', html)
                media = base / "window.mp4"
                media.write_bytes(b"test-window")
                with patch.object(ui, "window_mp4", return_value=media) as build:
                    self.assertEqual(urlopen(address + "/window?" + urlencode({"sid": row["segment_id"]})).read(), b"test-window")
                    self.assertEqual(build.call_args.kwargs["start"], 0.8)
                    self.assertEqual(build.call_args.kwargs["end"], 2.2)
                data = urlencode({"video": row["source_video"], "pass": "2", "i": "0",
                                  "sid": row["segment_id"], "action": "accepted", "klass": "KA"}).encode()
                urlopen(Request(address + "/review", data=data)).read()
                with segments.open(newline="", encoding="utf-8") as handle:
                    saved = next(csv.DictReader(handle))
                self.assertEqual(saved["pass2_reviewer"], "Independent Reviewer")
                self.assertEqual(saved["pass2_start_time_s"], "0.8")
                self.assertEqual(saved["pass2_end_time_s"], "2.2")
                self.assertEqual(saved["pass1_reviewer"], "Hidden Reviewer")
                self.assertTrue(segments.with_suffix(".csv.bak").is_file())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                ui.STATE.clear()
                ui.STATE.update(original)

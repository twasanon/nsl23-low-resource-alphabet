"""Prevent missing model modules and malformed blinded LaTeX in supplements."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bundle", ROOT / "scripts/build_reproducibility_bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class BundleTests(unittest.TestCase):
    def test_complete_model_and_v2_source_inventory_is_packaged(self):
        paths = {path.relative_to(ROOT).as_posix() for path in bundle.collect(ROOT, require_results=True)}
        for path in (ROOT / "src/nsl23").glob("*.py"):
            self.assertIn(path.relative_to(ROOT).as_posix(), paths)
        self.assertIn("manuscript/generated_protocol_tables.tex", paths)
        self.assertIn("manuscript/figures/figure_12_protocol_ladder.pdf", paths)
        self.assertIn("configs/v2_protocol.json", paths)
        self.assertEqual(sum(path.endswith("/training_history.csv") for path in paths), 60)

    def test_blinding_preserves_real_latex_line_breaks(self):
        tex = b"\\author{A Name}\n\n\\date{}\n"
        public = bundle.public_member_data(ROOT, ROOT / "manuscript/nsl23_low_resource.tex", tex)
        self.assertIn(b"\\author{Anonymous}\n\n\\date", public)
        self.assertNotIn(b"\\n\\n", public)


if __name__ == "__main__":
    unittest.main()

"""Safe reproduction entry points must preserve the reference experiment."""

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionEntrypointTests(unittest.TestCase):
    def test_prototypical_training_refuses_existing_artifacts_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "best_checkpoint.pt"
            checkpoint.write_bytes(b"preserve this checkpoint")
            result = subprocess.run(
                [sys.executable, "scripts/train_protonet.py", "--samples", "absent.csv",
                 "--output-dir", str(run_dir), "--seed", "17"],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Run artifact already exists", result.stderr)
            self.assertEqual(checkpoint.read_bytes(), b"preserve this checkpoint")

    def test_generated_notebook_covers_the_thirty_run_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            notebook = Path(directory) / "reproduction.ipynb"
            for command in (
                [sys.executable, "scripts/generate_reproducibility_notebook.py", "--output", str(notebook)],
                [sys.executable, "scripts/audit_reproducibility_notebook.py", str(notebook),
                 "--output", str(Path(directory) / "audit.json")],
            ):
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

import json
import importlib.metadata
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch


class ColabInstallTests(unittest.TestCase):
    def install_arguments(self, dependencies):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               "colab/Ettin32M_evaluation.ipynb").read_text())
        source = "".join(notebook["cells"][1]["source"])
        with patch("importlib.metadata.requires", side_effect=dependencies), \
                patch("subprocess.check_call") as install:
            exec(compile(source, "colab-install", "exec"), {})
        return install.call_args.args[0]

    def test_install_preserves_gcsfs_requirement_and_gpu_torch(self):
        arguments = self.install_arguments(lambda name: ["fsspec==2025.12.0", "aiohttp"])
        self.assertIn("fsspec==2025.12.0", arguments)
        self.assertIn("huggingface-hub>=1.32.0,<3.0", arguments)
        self.assertFalse(any(argument.startswith(("torch=", "torch>", "torch<")) for argument in arguments))
        self.assertNotIn("transformers==4.51.3", arguments)

    def test_install_also_works_without_gcsfs(self):
        def missing(name):
            raise importlib.metadata.PackageNotFoundError(name)
        arguments = self.install_arguments(missing)
        self.assertFalse(any(argument.startswith("fsspec") for argument in arguments))


class ColabScoreTests(unittest.TestCase):
    def calculate(self, flops, runtime=100):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               "colab/Ettin32M_evaluation.ipynb").read_text())
        source = "".join(next(cell for cell in notebook["cells"] if cell.get("id") == "ettin-score")["source"])
        metrics = dict(mean_f1=0.7, forward_seconds_per_pass=runtime,
                       gflops_per_pass=flops, full_public_test=True,
                       Vulnerability=dict(precision=0.9, recall=0.8), MAT=dict(precision=0.6, recall=0.7))
        with tempfile.TemporaryDirectory() as temporary:
            scope = dict(metrics=metrics, LABELS=["Vulnerability", "MAT"],
                         RESULT_DIR=Path(temporary), json=json)
            exec(compile(source, "colab-score", "exec"), scope)
            self.assertEqual(json.loads((Path(temporary) / "metrics.json").read_text())
                             ["competition_score_local"], scope["total_score"])
        return scope["total_score"]

    def test_score_matches_original_formula_and_clamps_penalties(self):
        self.assertEqual(self.calculate(1_500_000), 0.7)
        self.assertEqual(self.calculate(4_000_000, runtime=2000), 0.42)

    def test_missing_flops_produces_no_fabricated_score(self):
        self.assertIsNone(self.calculate(None))


if __name__ == "__main__":
    unittest.main()

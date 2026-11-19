import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nlbse", ROOT / "scripts/nlbse.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class MetricTests(unittest.TestCase):
    def test_independent_tasks_and_mean_f1(self):
        labels = [[1, 0], [1, 1], [0, 0], [0, 1]]
        predictions = [[1, 1], [0, 1], [0, 0], [1, 1]]
        metrics = pipeline.classification_metrics(labels, predictions)
        self.assertEqual(metrics["Vulnerability"]["tp"], 1)
        self.assertEqual(metrics["Vulnerability"]["fp"], 1)
        self.assertEqual(metrics["Vulnerability"]["tn"], 1)
        self.assertEqual(metrics["Vulnerability"]["fn"], 1)
        self.assertEqual(metrics["Vulnerability"]["f1"], 0.5)
        self.assertAlmostEqual(metrics["MAT"]["f1"], 0.8)
        self.assertAlmostEqual(metrics["mean_f1"], 0.65)

    def test_no_positive_predictions_or_labels(self):
        metrics = pipeline.classification_metrics([[0, 0]], [[0, 0]])
        self.assertEqual(metrics["mean_f1"], 0)
        self.assertEqual(metrics["MAT"]["accuracy"], 1)

    def test_bad_labels_and_sizes_rejected(self):
        for labels, predictions in [([], []), ([[0, 1]], []), ([[1]], [[0, 0]]),
                                    ([[1, 2]], [[0, 0]])]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                pipeline.classification_metrics(labels, predictions)

    def test_prepared_input_mismatch_rejected(self):
        config = pipeline.load_config(ROOT / "configs/codebert-baseline.json")
        manifest = {key: config[key] for key in ("model_name", "dataset_name", "max_length",
                                                "model_revision", "dataset_revision")}
        manifest["max_length"] = 128
        with tempfile.TemporaryDirectory() as directory:
            pipeline.write_json(Path(directory) / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "max_length"):
                pipeline.check_prepared(config, Path(directory))


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nlbse tests ")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        prepared = self.path / "prepared"
        prepared.mkdir()
        (prepared / "manifest.json").write_text("{}")
        venv = self.path / "venv/bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("#!/bin/sh\nexit 0\n")
        (venv / "python").chmod(0o755)
        self.config = self.path / "config.sh"
        self.config.write_text("\n".join([
            f"HPC_WORKDIR={shlex.quote(str(self.path))}",
            'HPC_VENV="$HPC_WORKDIR/venv"', 'HPC_PREPARED="$HPC_WORKDIR/prepared"',
            'HPC_TRAIN_CONFIG="$REPO_ROOT/configs/codebert-baseline.json"',
            "HPC_MODULES=()", "HPC_PARTITION=verified-test-partition", "HPC_ACCOUNT=project123",
            "HPC_QOS=normal", "HPC_GRES=gpu:1", "HPC_CPUS=4", "HPC_MEM=32G",
            "HPC_TIME=24:00:00", "HPC_SMOKE_TIME=00:20:00",
        ]) + "\n")
        self.env = dict(os.environ, HPC_CONFIG=str(self.config), DRY_RUN="1")
        self.env.pop("RESUME", None)

    def submit(self, *arguments):
        return subprocess.run(["bash", str(ROOT / "hpc/submit.sh"), *arguments],
                              env=self.env, capture_output=True, text=True)

    def test_smoke_uses_short_time_and_paths_with_spaces(self):
        result = self.submit("smoke", "first-smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout.splitlines()[-1])
        self.assertIn("--time=00:20:00", command)
        self.assertIn("--partition=verified-test-partition", command)
        self.assertIn("--account=project123", command)
        self.assertIn(f"--output={self.path}/logs/%x-%j.out", command)

    def test_unverified_partition_rejected(self):
        self.config.write_text(self.config.read_text().replace("verified-test-partition", "CHANGE_ME"))
        self.assertNotEqual(self.submit("train").returncode, 0)

    def test_run_path_traversal_rejected(self):
        self.assertNotEqual(self.submit("train", "../elsewhere").returncode, 0)

    def test_existing_run_requires_resume(self):
        (self.path / "runs/existing").mkdir(parents=True)
        self.assertNotEqual(self.submit("train", "existing").returncode, 0)
        self.env["RESUME"] = "auto"
        self.assertEqual(self.submit("train", "existing").returncode, 0)

    def test_evaluation_requires_saved_model(self):
        self.assertNotEqual(self.submit("evaluate", "absent").returncode, 0)
        (self.path / "runs/finished/final_model").mkdir(parents=True)
        self.assertEqual(self.submit("evaluate", "finished").returncode, 0)

    def test_submission_passes_environment_to_sbatch(self):
        binary = self.path / "bin"
        binary.mkdir()
        mock = binary / "sbatch"
        mock.write_text('#!/bin/bash\nprintf "%s\\n" "$NLBSE_MODE" "$NLBSE_RUN_DIR" "$HPC_CONFIG" "$@"\n')
        mock.chmod(0o755)
        self.env.update(DRY_RUN="0", PATH=f"{binary}:{self.env['PATH']}")
        result = self.submit("train", "new-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[:3], ["train", str(self.path / "runs/new-run"), str(self.config)])
        self.assertIn("--export=ALL", lines)


if __name__ == "__main__":
    unittest.main()

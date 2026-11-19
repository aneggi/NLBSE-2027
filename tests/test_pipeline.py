import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nlbse", ROOT / "scripts/nlbse.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class MetricTests(unittest.TestCase):
    def test_default_model_is_pinned_ettin_32m(self):
        config = pipeline.load_config(pipeline.DEFAULT_CONFIG)
        self.assertEqual(config["model_name"], "jhu-clsp/ettin-encoder-32m")
        self.assertEqual(len(config["model_revision"]), 40)
        self.assertTrue(config["use_fast_tokenizer"])

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

    def test_tokenizer_changes_require_new_preparation(self):
        config = pipeline.load_config(pipeline.DEFAULT_CONFIG)
        manifest = {key: config[key] for key in ("model_name", "dataset_name", "max_length",
                                                "model_revision", "dataset_revision")}
        manifest["use_fast_tokenizer"] = False
        with tempfile.TemporaryDirectory() as directory:
            pipeline.write_json(Path(directory) / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                pipeline.check_prepared(config, Path(directory))


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nlbse tests ")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        prepared = self.path / "prepared/ettin-32m-512"
        prepared.mkdir(parents=True)
        (prepared / "manifest.json").write_text("{}")
        venv = self.path / "venv/bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("#!/bin/sh\nexit 0\n")
        (venv / "python").chmod(0o755)
        self.config = self.path / "config.sh"
        self.config.write_text("\n".join([
            f"HPC_WORKDIR={shlex.quote(str(self.path))}",
            "HPC_PARTITION=verified-test-partition", "HPC_ACCOUNT=project123", "HPC_QOS=normal",
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
        self.assertIn(f"--output={self.path}/logs/ettin-%j.out", command)

    def test_only_partition_and_storage_override_are_needed(self):
        self.config.write_text(f"HPC_WORKDIR={shlex.quote(str(self.path))}\nHPC_PARTITION=verified-test-partition\n")
        result = self.submit("train", "minimal-config")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout.splitlines()[-1])
        for option in ("--gres=gpu:1", "--cpus-per-task=4", "--mem=16G", "--time=02:00:00"):
            self.assertIn(option, command)
        self.assertFalse(any(option.startswith(("--account=", "--qos=")) for option in command))

    def test_cluster_profile_passes_both_partitions_and_account(self):
        self.config.write_text((ROOT / "hpc/config.example.sh").read_text() +
                               f"\nHPC_WORKDIR={shlex.quote(str(self.path))}\n")
        result = self.submit("train", "two-partition-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout.splitlines()[-1])
        for option in ("--partition=gpu-low,gpu-pre", "--account=greenml", "--nodes=1",
                       "--ntasks=1", "--gres=gpu:1", "--cpus-per-task=4", "--mem=16G",
                       "--time=02:00:00", f"--output={self.path}/logs/ettin-%j.out",
                       f"--error={self.path}/logs/ettin-%j.err"):
            self.assertIn(option, command)
        self.assertEqual(sum(option.startswith("--partition=") for option in command), 1)
        self.assertFalse(any(option.startswith(("--constraint=", "--nodelist=")) for option in command))

    def test_optional_node_feature_is_passed_without_fixing_a_hostname(self):
        with self.config.open("a") as handle:
            handle.write("HPC_CONSTRAINT=gpu80g\n")
        result = self.submit("train", "feature-filtered-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout.splitlines()[-1])
        self.assertIn("--constraint=gpu80g", command)
        self.assertFalse(any(option.startswith("--nodelist=") for option in command))

    def test_advanced_overrides_still_work(self):
        with self.config.open("a") as handle:
            handle.write("HPC_MEM=64G\nHPC_TIME=06:00:00\n")
        result = self.submit("train", "custom-limits")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout.splitlines()[-1])
        self.assertIn("--mem=64G", command)
        self.assertIn("--time=06:00:00", command)

    def test_python_detection_skips_unsupported_interpreter(self):
        binary = self.path / "python-bin"
        binary.mkdir()
        for name, status in (("python3.11", 1), ("python3.12", 0)):
            interpreter = binary / name
            interpreter.write_text(f"#!/bin/sh\nexit {status}\n")
            interpreter.chmod(0o755)
        self.env["PATH"] = f"{binary}:{self.env['PATH']}"
        result = subprocess.run(["bash", "-c", 'source "$1"; load_hpc_modules; find_hpc_python; echo "$HPC_PYTHON"',
                                 "_", str(ROOT / "hpc/common.sh")], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "python3.12")

    def test_explicit_unsupported_python_fails_with_guidance(self):
        unsupported = self.path / "unsupported-python"
        unsupported.write_text("#!/bin/sh\nexit 1\n")
        unsupported.chmod(0o755)
        with self.config.open("a") as handle:
            handle.write(f"HPC_PYTHON={shlex.quote(str(unsupported))}\n")
        result = subprocess.run(["bash", "-c", 'source "$1"; find_hpc_python',
                                 "_", str(ROOT / "hpc/common.sh")], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HPC_MODULES", result.stderr)

    def test_unverified_partition_rejected(self):
        self.config.write_text(self.config.read_text().replace("verified-test-partition", "CHANGE_ME"))
        self.assertNotEqual(self.submit("train").returncode, 0)

    def test_run_path_traversal_rejected(self):
        self.assertNotEqual(self.submit("train", "../elsewhere").returncode, 0)

    def test_existing_run_requires_resume(self):
        (self.path / "runs/existing").mkdir(parents=True)
        self.assertNotEqual(self.submit("train", "existing").returncode, 0)
        self.env["RESUME"] = "auto"
        self.assertNotEqual(self.submit("train", "existing").returncode, 0)
        (self.path / "runs/existing/run.json").write_text("{}")
        checkpoint = self.path / "runs/existing/checkpoint-500"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}")
        self.assertEqual(self.submit("train", "existing").returncode, 0)

    def test_resume_missing_run_is_rejected_before_submission(self):
        self.env["RESUME"] = "auto"
        result = self.submit("train", "never-started")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without RESUME", result.stderr)
        self.assertNotIn("sbatch", result.stdout)
        self.assertFalse((self.path / "runs/never-started").exists())

    def test_resume_rejects_run_without_saved_checkpoints(self):
        run = self.path / "runs/interrupted-early"
        run.mkdir(parents=True)
        (run / "run.json").write_text("{}")
        self.env["RESUME"] = "auto"
        result = self.submit("train", "interrupted-early")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No saved checkpoint", result.stderr)
        self.assertNotIn("sbatch", result.stdout)

    def test_evaluation_requires_saved_model(self):
        self.assertNotEqual(self.submit("evaluate", "absent").returncode, 0)
        (self.path / "runs/finished/final_model").mkdir(parents=True)
        self.assertEqual(self.submit("evaluate", "finished").returncode, 0)

    def test_submission_passes_environment_to_sbatch(self):
        binary = self.path / "bin"
        binary.mkdir()
        mock = binary / "sbatch"
        mock.write_text('#!/bin/bash\nprintf "%s\\n" "$NLBSE_MODE" "$NLBSE_RUN_DIR" "$HPC_CONFIG" "$REPO_ROOT" "$@"\n')
        mock.chmod(0o755)
        self.env.update(DRY_RUN="0", PATH=f"{binary}:{self.env['PATH']}")
        result = self.submit("train", "new-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[:4], ["train", str(self.path / "runs/new-run"), str(self.config), str(ROOT)])
        self.assertIn("--export=ALL", lines)

    def test_smoke_job_runs_from_slurm_spool_directory(self):
        spool = self.path / "slurm spool/job276325"
        spool.mkdir(parents=True)
        script = spool / "slurm_script"
        shutil.copyfile(ROOT / "hpc/job.sh", script)
        calls_path = self.path / "python-calls.jsonl"
        venv_bin = self.path / "venv/bin"
        (venv_bin / "activate").write_text(f"export PATH={shlex.quote(str(venv_bin))}:\"$PATH\"\n")
        (venv_bin / "python").write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['NLBSE_TEST_CALLS'], 'a') as handle:\n"
            "    handle.write(json.dumps({'args': sys.argv[1:], "
            "'offline': os.environ.get('HF_HUB_OFFLINE'), "
            "'threads': os.environ.get('OMP_NUM_THREADS')}) + '\\n')\n"
        )
        env = dict(self.env, REPO_ROOT=str(ROOT), SLURM_JOB_ID="276325", SLURM_CPUS_PER_TASK="4",
                   NLBSE_MODE="smoke", NLBSE_RUN_DIR=str(self.path / "runs/spooled-smoke"),
                   NLBSE_TEST_CALLS=str(calls_path))
        result = subprocess.run(["bash", str(script)], cwd=spool, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
        training = next(call for call in calls if "train" in call["args"])
        evaluation = next(call for call in calls if "evaluate" in call["args"])
        self.assertEqual(training["args"][0], str(ROOT / "scripts/nlbse.py"))
        self.assertIn("--max-steps", training["args"])
        self.assertEqual(training["args"][-4:], ["--max-steps", "2", "--max-train-samples", "32"])
        self.assertEqual(evaluation["args"][-2:], ["--max-test-samples", "16"])
        self.assertIn(str(self.path / "runs/spooled-smoke/evaluation-276325"), evaluation["args"])
        for call in (training, evaluation):
            self.assertEqual(call["offline"], "1")
            self.assertEqual(call["threads"], "4")


if __name__ == "__main__":
    unittest.main()

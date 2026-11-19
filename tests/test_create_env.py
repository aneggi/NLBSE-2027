import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

MODULE_PATH = Path(__file__).resolve().parents[1] / "hpc/create_env.py"
spec = importlib.util.spec_from_file_location("create_env", MODULE_PATH)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nlbse env tests ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.venv = self.root / "venv"
        self.cache = self.root / "cache"

    def test_python_without_activation_is_incomplete(self):
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin/python").touch()
        self.assertFalse(bootstrap.environment_ready(self.venv))

    def test_missing_pip_is_incomplete(self):
        (self.venv / "bin").mkdir(parents=True)
        for name in ("python", "activate"):
            (self.venv / "bin" / name).touch()
        with patch.object(bootstrap.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
            self.assertFalse(bootstrap.environment_ready(self.venv))

    def test_complete_environment_is_reused(self):
        with patch.object(bootstrap, "environment_ready", return_value=True), \
                patch.object(bootstrap.subprocess, "run") as run:
            bootstrap.create_environment(self.venv, self.cache)
        run.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_standard_venv_succeeds_without_download(self):
        with patch.object(bootstrap, "environment_ready", side_effect=[False, True]), \
                patch.object(bootstrap.importlib.util, "find_spec", return_value=object()), \
                patch.object(bootstrap.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(bootstrap, "download_zipapp") as download:
            bootstrap.create_environment(self.venv, self.cache)
        download.assert_not_called()

    def test_missing_ensurepip_uses_cached_zipapp_and_preserves_existing_files(self):
        self.venv.mkdir()
        marker = self.venv / "keep.txt"
        marker.write_text("keep")
        self.cache.mkdir()
        version = f"{bootstrap.sys.version_info.major}.{bootstrap.sys.version_info.minor}"
        (self.cache / f"virtualenv-{version}.pyz").touch()
        with patch.object(bootstrap, "environment_ready", side_effect=[False, True]), \
                patch.object(bootstrap.importlib.util, "find_spec", return_value=None), \
                patch.object(bootstrap.subprocess, "run") as run, \
                patch.object(bootstrap, "download_zipapp") as download:
            bootstrap.create_environment(self.venv, self.cache)
        self.assertEqual(run.call_count, 1)
        self.assertIn("--no-download", run.call_args.args[0])
        download.assert_not_called()
        self.assertEqual(marker.read_text(), "keep")

    def test_failed_standard_venv_falls_back(self):
        with patch.object(bootstrap, "environment_ready", side_effect=[False, True]), \
                patch.object(bootstrap.importlib.util, "find_spec", return_value=object()), \
                patch.object(bootstrap.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)) as run, \
                patch.object(bootstrap, "download_zipapp") as download:
            bootstrap.create_environment(self.venv, self.cache)
        self.assertEqual(run.call_count, 2)
        download.assert_called_once()

    def test_failed_download_does_not_leave_cached_bootstrap(self):
        self.cache.mkdir()
        destination = self.cache / "virtualenv.pyz"
        with patch.object(bootstrap, "urlopen", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                bootstrap.download_zipapp(destination)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_download_is_complete_before_cache_becomes_visible(self):
        self.cache.mkdir()
        source = self.root / "source.pyz"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("__main__.py", "print('fixture')")
        destination = self.cache / "virtualenv.pyz"
        with source.open("rb") as response, patch.object(bootstrap, "urlopen", return_value=response):
            bootstrap.download_zipapp(destination)
        self.assertTrue(zipfile.is_zipfile(destination))
        self.assertEqual(list(self.cache.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()

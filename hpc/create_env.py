#!/usr/bin/env python3
"""Create or repair the HPC virtual environment without system package changes."""

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile


def environment_ready(path):
    python = path / "bin/python"
    if not python.is_file() or not (path / "bin/activate").is_file():
        return False
    result = subprocess.run(
        [str(python), "-c", "import sys, pip; assert sys.prefix != sys.base_prefix"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def download_zipapp(destination):
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    url = f"https://bootstrap.pypa.io/virtualenv/{version}/virtualenv.pyz"
    print(f"Downloading virtualenv from {url}", flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, handle)
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError("Downloaded virtualenv is not a valid zipapp")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_environment(path, cache):
    if environment_ready(path):
        print(f"Reusing environment: {path}", flush=True)
        return
    # A failed Debian/Ubuntu venv creation can leave bin/python without pip.
    # Re-run creation in place; never delete the existing directory or its files.
    if importlib.util.find_spec("ensurepip") is not None:
        result = subprocess.run([sys.executable, "-m", "venv", str(path)])
        if result.returncode == 0 and environment_ready(path):
            return
    print("Standard venv is unavailable or incomplete; using virtualenv without sudo.", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    zipapp = cache / f"virtualenv-{version}.pyz"
    if not zipapp.is_file():
        download_zipapp(zipapp)
    subprocess.run(
        [sys.executable, str(zipapp), "--python", sys.executable,
         "--app-data", str(cache / "app-data"), "--no-periodic-update",
         "--no-download", str(path)], check=True,
    )
    if not environment_ready(path):
        raise RuntimeError(f"Environment is still incomplete: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    try:
        create_environment(args.venv, args.cache)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Environment setup failed: {error}\n")


if __name__ == "__main__":
    main()

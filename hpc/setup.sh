#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_hpc_modules
"$HPC_PYTHON" -c 'import sys; assert sys.version_info[:2] in ((3, 11), (3, 12)), "Use Python 3.11 or 3.12 for this environment"'
mkdir -p "$HPC_WORKDIR" "$HF_HOME"
if [[ ! -x "$HPC_VENV/bin/python" ]]; then
    "$HPC_PYTHON" -m venv "$HPC_VENV"
fi
"$HPC_VENV/bin/python" -m pip install --upgrade pip
"$HPC_VENV/bin/python" -m pip install torch==2.6.0 --index-url "$TORCH_INDEX_URL"
"$HPC_VENV/bin/python" -m pip install -r "$REPO_ROOT/requirements.txt"
"$HPC_VENV/bin/python" -m pip check
"$HPC_VENV/bin/python" -m pip freeze > "$HPC_WORKDIR/environment-freeze.txt"
echo "Environment ready: $HPC_VENV"

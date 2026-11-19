#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_hpc_modules
find_hpc_python
echo "Using $HPC_PYTHON; workspace: $HPC_WORKDIR"
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

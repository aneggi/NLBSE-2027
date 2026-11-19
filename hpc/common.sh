#!/usr/bin/env bash
# Shared by setup, preparation, submission, and batch execution.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HPC_CONFIG="${HPC_CONFIG:-$REPO_ROOT/hpc/config.sh}"
if [[ ! -f "$HPC_CONFIG" ]]; then
    echo "Missing $HPC_CONFIG. Copy hpc/config.example.sh to hpc/config.sh and edit it." >&2
    exit 1
fi
HPC_CONFIG="$(cd -- "$(dirname -- "$HPC_CONFIG")" && pwd)/$(basename -- "$HPC_CONFIG")"
# shellcheck source=/dev/null
source "$HPC_CONFIG"
: "${HPC_WORKDIR:?Set HPC_WORKDIR in hpc/config.sh}"
: "${HPC_VENV:?Set HPC_VENV in hpc/config.sh}"
: "${HPC_PREPARED:?Set HPC_PREPARED in hpc/config.sh}"
: "${HPC_TRAIN_CONFIG:?Set HPC_TRAIN_CONFIG in hpc/config.sh}"
for path in "$HPC_WORKDIR" "$HPC_VENV" "$HPC_PREPARED" "$HPC_TRAIN_CONFIG"; do
    if [[ "$path" != /* ]]; then
        echo "HPC storage and configuration paths must be absolute: $path" >&2
        exit 1
    fi
done
export REPO_ROOT HPC_CONFIG HPC_WORKDIR HPC_VENV HPC_PREPARED HPC_TRAIN_CONFIG
export HF_HOME="$HPC_WORKDIR/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

load_hpc_modules() {
    if (( ${#HPC_MODULES[@]} )); then
        if ! type module >/dev/null 2>&1; then
            echo "The module command is unavailable; initialize it as the cluster help instructs." >&2
            exit 1
        fi
        module load "${HPC_MODULES[@]}"
    fi
}

activate_hpc_env() {
    load_hpc_modules
    if [[ ! -x "$HPC_VENV/bin/python" ]]; then
        echo "Missing environment; run bash hpc/setup.sh first." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$HPC_VENV/bin/activate"
}

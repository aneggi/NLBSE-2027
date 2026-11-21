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
HPC_MODULES=()
# shellcheck source=/dev/null
source "$HPC_CONFIG"
# Normal runs only need a GPU partition. These can still be overridden in config.sh.
HPC_WORKDIR="${HPC_WORKDIR:-$HOME/nlbse27}"
HPC_VENV="${HPC_VENV:-$HPC_WORKDIR/venv}"
HPC_MODEL="${HPC_MODEL:-ettin-32m}"
case "$HPC_MODEL" in
    ettin-17m|ettin-32m|codebert-baseline) ;;
    *) echo "Unknown HPC_MODEL: $HPC_MODEL. Choose ettin-17m, ettin-32m, or codebert-baseline." >&2; exit 1 ;;
esac
HPC_PREPARED="${HPC_PREPARED:-$HPC_WORKDIR/prepared/$HPC_MODEL-512}"
HPC_TRAIN_CONFIG="${HPC_TRAIN_CONFIG:-$REPO_ROOT/configs/$HPC_MODEL.json}"
HPC_PYTHON="${HPC_PYTHON:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
HPC_ACCOUNT="${HPC_ACCOUNT:-}"
HPC_QOS="${HPC_QOS:-}"
HPC_GRES="${HPC_GRES:-gpu:1}"
HPC_CONSTRAINT="${HPC_CONSTRAINT:-}"
HPC_NODELIST="${HPC_NODELIST:-}"
HPC_CPUS="${HPC_CPUS:-4}"
HPC_MEM="${HPC_MEM:-16G}"
HPC_TIME="${HPC_TIME:-02:00:00}"
HPC_SMOKE_TIME="${HPC_SMOKE_TIME:-00:20:00}"
for path in "$HPC_WORKDIR" "$HPC_VENV" "$HPC_PREPARED" "$HPC_TRAIN_CONFIG"; do
    if [[ "$path" != /* ]]; then
        echo "HPC storage and configuration paths must be absolute: $path" >&2
        exit 1
    fi
done
export REPO_ROOT HPC_CONFIG HPC_WORKDIR HPC_VENV HPC_PREPARED HPC_TRAIN_CONFIG HPC_MODEL
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

find_hpc_python() {
    local candidate
    local candidates=(python3.11 python3.12 python3)
    if [[ -n "$HPC_PYTHON" ]]; then candidates=("$HPC_PYTHON"); fi
    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1 && \
            "$candidate" -c 'import sys; assert sys.version_info[:2] in ((3, 11), (3, 12))' 2>/dev/null; then
            HPC_PYTHON="$candidate"
            return
        fi
    done
    echo "Python 3.11/3.12 not found. Load your cluster's Python module, or set HPC_MODULES in $HPC_CONFIG." >&2
    echo "You can also set HPC_PYTHON to a specific Python 3.11/3.12 executable." >&2
    exit 1
}

activate_hpc_env() {
    load_hpc_modules
    if [[ ! -x "$HPC_VENV/bin/python" || ! -f "$HPC_VENV/bin/activate" ]] || \
        ! "$HPC_VENV/bin/python" -c 'import sys, pip; assert sys.prefix != sys.base_prefix' >/dev/null 2>&1; then
        echo "Missing or incomplete environment; run bash hpc/setup.sh first." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$HPC_VENV/bin/activate"
}

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
activate_hpc_env
python "$REPO_ROOT/scripts/nlbse.py" prepare \
    --config "$HPC_TRAIN_CONFIG" --prepared "$HPC_PREPARED"

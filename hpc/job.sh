#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script through hpc/submit.sh}"
# Slurm executes a spool copy of this file. submit.sh exports the original
# repository path via common.sh and --export=ALL; BASH_SOURCE points at the copy.
: "${REPO_ROOT:?Missing repository path; submit through hpc/submit.sh}"
source "$REPO_ROOT/hpc/common.sh"
: "${NLBSE_MODE:?Missing submission mode}"
: "${NLBSE_RUN_DIR:?Missing run directory}"
activate_hpc_env
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "Job $SLURM_JOB_ID on $(hostname); run $NLBSE_RUN_DIR"
python -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU visible"; print(torch.__version__, torch.cuda.get_device_name(0))'
common=(--config "$HPC_TRAIN_CONFIG" --prepared "$HPC_PREPARED" --require-cuda)
train_options=()
eval_options=()
if [[ "$NLBSE_MODE" == smoke ]]; then
    train_options+=(--max-steps 2 --max-train-samples 32)
    eval_options+=(--max-test-samples 16)
fi
if [[ -n "${NLBSE_RESUME:-}" ]]; then train_options+=(--resume "$NLBSE_RESUME"); fi
if [[ "$NLBSE_MODE" != evaluate ]]; then
    python "$REPO_ROOT/scripts/nlbse.py" train "${common[@]}" \
        --output "$NLBSE_RUN_DIR" "${train_options[@]}"
fi
python "$REPO_ROOT/scripts/nlbse.py" evaluate "${common[@]}" \
    --model "$NLBSE_RUN_DIR/final_model" \
    --output "$NLBSE_RUN_DIR/evaluation-$SLURM_JOB_ID" "${eval_options[@]}"

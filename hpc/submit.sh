#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mode="${1:-}"
if [[ "$mode" != smoke && "$mode" != train && "$mode" != evaluate ]]; then
    echo "Usage: bash hpc/submit.sh {smoke|train|evaluate} [run-name]" >&2
    exit 1
fi
if [[ "$mode" == evaluate && -z "${2:-}" ]]; then
    echo "Evaluation requires the name of an existing run." >&2
    exit 1
fi
run_name="${2:-$mode-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "Run name must start with a letter/digit and contain only letters, digits, ., _, -." >&2
    exit 1
fi
if [[ -z "${HPC_PARTITION:-}" || "$HPC_PARTITION" == CHANGE_ME ]]; then
    echo "Set a verified GPU HPC_PARTITION in $HPC_CONFIG before submitting." >&2
    exit 1
fi
if [[ ! -f "$HPC_PREPARED/manifest.json" || ! -x "$HPC_VENV/bin/python" ]]; then
    echo "Run setup.sh and prepare.sh before submitting." >&2
    exit 1
fi
export NLBSE_MODE="$mode" NLBSE_RUN_DIR="$HPC_WORKDIR/runs/$run_name"
export NLBSE_RESUME="${RESUME:-}"
if [[ "$mode" == evaluate && ! -d "$NLBSE_RUN_DIR/final_model" ]]; then
    echo "No final_model in $NLBSE_RUN_DIR" >&2
    exit 1
fi
if [[ "$mode" != evaluate && -z "$NLBSE_RESUME" && -d "$NLBSE_RUN_DIR" ]]; then
    echo "Run already exists; choose a new name or set RESUME=auto." >&2
    exit 1
fi
time_limit="$HPC_TIME"
if [[ "$mode" == smoke ]]; then time_limit="$HPC_SMOKE_TIME"; fi
mkdir -p "$HPC_WORKDIR/logs"
options=(--job-name="nlbse-$mode" --nodes=1 --ntasks=1
    --partition="$HPC_PARTITION" --gres="$HPC_GRES"
    --cpus-per-task="$HPC_CPUS" --mem="$HPC_MEM" --time="$time_limit"
    --chdir="$REPO_ROOT" --export=ALL
    --output="$HPC_WORKDIR/logs/%x-%j.out" --error="$HPC_WORKDIR/logs/%x-%j.err")
if [[ -n "${HPC_ACCOUNT:-}" ]]; then options+=(--account="$HPC_ACCOUNT"); fi
if [[ -n "${HPC_QOS:-}" ]]; then options+=(--qos="$HPC_QOS"); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'NLBSE_MODE=%q NLBSE_RUN_DIR=%q NLBSE_RESUME=%q\n' "$NLBSE_MODE" "$NLBSE_RUN_DIR" "$NLBSE_RESUME"
    printf '%q ' sbatch "${options[@]}" "$REPO_ROOT/hpc/job.sh"
    printf '\n'
else
    command -v sbatch >/dev/null || { echo "sbatch unavailable; verify the cluster scheduler." >&2; exit 1; }
    sbatch "${options[@]}" "$REPO_ROOT/hpc/job.sh"
fi

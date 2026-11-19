# Copy to hpc/config.sh on the cluster. This is sourced as trusted shell code.
# Consult https://hpc.scientificnet.org/help/ while on ScientificNet/VPN.
# This Slurm template is conditional: first confirm the cluster uses sbatch.

# Use persistent shared project/scratch storage visible on login AND GPU nodes.
# The home-directory default is a fallback, not a verified university storage path.
HPC_WORKDIR="${HPC_WORKDIR:-$HOME/nlbse27}"
HPC_VENV="$HPC_WORKDIR/venv"
HPC_PREPARED="$HPC_WORKDIR/prepared/codebert-512"
HPC_TRAIN_CONFIG="$REPO_ROOT/configs/codebert-baseline.json"

# Set the actual module names from `module avail`, or leave empty if Python 3.11/3.12
# is already available in your login and batch environments.
HPC_MODULES=()
HPC_PYTHON=python3.11

# Select a wheel compatible with the GPU driver on the allocated compute node.
# Alternative official indexes include .../cu118 and .../cpu (CPU for local testing).
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

# REQUIRED before submitting: replace this with a verified GPU partition.
HPC_PARTITION=CHANGE_ME
# Accounting project and QoS are separate from the SSH username aaneggi.
HPC_ACCOUNT=""
HPC_QOS=""
HPC_GRES=gpu:1
HPC_CPUS=4
HPC_MEM=32G
HPC_TIME=24:00:00
HPC_SMOKE_TIME=00:20:00

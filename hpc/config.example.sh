# Copy to hpc/config.sh. Set the GPU partition shown by `sinfo` on your Slurm HPC.
HPC_PARTITION=gpu-pre

# Optional: uncomment only if required by your cluster.
# HPC_WORKDIR=/absolute/path/to/shared/project/nlbse27
# HPC_ACCOUNT=your_allocation
# HPC_MODULES=(your_python_module)

# Everything else has defaults: Python 3.11/3.12 is detected automatically;
# files go in $HOME/nlbse27; jobs request 1 GPU, 4 CPUs, 32 GB, and 24 hours.
# Storage must be visible on compute nodes. Advanced options: ../hpc-unibz.md

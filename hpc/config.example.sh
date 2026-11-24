# Copy to hpc/config.sh. Let Slurm consider both listed GPU partitions.
HPC_PARTITION=gpu-low,gpu-pre
# greenml was the account used by your previously accepted job.
HPC_ACCOUNT=greenml

# Optional: uncomment only if required by your cluster.
# HPC_WORKDIR=/absolute/path/to/shared/project/nlbse27
# HPC_MODULES=(your_python_module)
# HPC_CONSTRAINT=gpu80g  # Optional: restrict to nodes with this feature; may wait longer.
# HPC_GRES=gpu:v100:1   # Optional: match a previous run's GPU type for timing.
# HPC_NODELIST=hpcsgn07 # Optional: match a verified previous node; may wait longer.

# Everything else has defaults: Python 3.11/3.12 is detected automatically;
# model: Ettin encoder 32M, with a separate prepared/ettin-32m-512 data cache;
# select 17M per command with HPC_MODEL=ettin-17m (see README.md);
# select the Crivo decision encoder with HPC_MODEL=crivo-17m;
# select the original backbone with HPC_MODEL=codebert-baseline;
# files go in $HOME/nlbse27; jobs request 1 GPU, 4 CPUs, 16 GB RAM, and 2 hours.
# Storage must be visible on compute nodes. Advanced options: ../hpc-unibz.md

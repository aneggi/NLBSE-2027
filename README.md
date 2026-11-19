# NLBSE 2027: HPC CodeBERT baseline

Fine-tune one CodeBERT model to predict `[Vulnerability, MAT/SATD]` from the
competition dataset's `Function` field. Start with the existing competition
baseline architecture before trying other models or techniques.

**[HPC setup and execution guide](hpc-unibz.md)** covers cloning this repository
as `aaneggi`, configuring the cluster, preparing offline inputs, running a smoke
job, training, resuming, evaluating, and predicting on new functions.

On the Slurm HPC, copy the small configuration file:

```bash
cp hpc/config.example.sh hpc/config.sh
```

Set **`HPC_PARTITION`** in `hpc/config.sh` to your GPU partition (check `sinfo`).
Python 3.11/3.12 is detected automatically and files default to `$HOME/nlbse27`.
If the cluster requires a shared storage path, accounting project, or Python
module, uncomment the matching optional line. Then run:

```bash
bash hpc/setup.sh
bash hpc/prepare.sh
bash hpc/submit.sh smoke smoke-001
# Check the smoke job's logs and metrics before starting the complete run:
bash hpc/submit.sh train codebert-001
```

The supplied notebooks evaluate an already trained model; they do not contain
the original training procedure. This implementation retains their task,
input, output order, truncation, and decision threshold. Training hyperparameters
are explicit starting choices in [the configuration](configs/codebert-baseline.json),
not a claim to reproduce the published weights or F1 exactly.

The ScientificNet help site requires its network/VPN. The provided batch wrapper
targets **Slurm, subject to confirmation on the actual cluster**. No login host,
GPU partition, account/project, QoS, or software module names have been invented.
See the guide before submitting.

Repository layout:

- `scripts/nlbse.py`: preparation, training, evaluation, and JSONL inference.
- `configs/codebert-baseline.json`: model, data, and training settings.
- `hpc/`: environment setup and configurable single-GPU job submission.
- `code-comment-classification-main/`: supplied current competition resources.
- `old_colab/`: older reference material; its split terminology differs.
- `tests/`: metric/submission checks and a tiny offline integration check.

Run lightweight checks with `python3 -m unittest discover -s tests -v`.
With the training dependencies installed, run
`python tests/smoke_pipeline.py` to exercise training, checkpoint resume,
evaluation, FLOP profiling, and inference on a tiny random model without downloads.

# NLBSE 2027: Ettin-32M on the HPC

Fine-tune **[`jhu-clsp/ettin-encoder-32m`](https://huggingface.co/jhu-clsp/ettin-encoder-32m)**
to predict `[Vulnerability, MAT/SATD]` from the competition dataset's `Function`
field. This is our smaller-model experiment following the CodeBERT baseline.

The default is now [Ettin-32M](configs/ettin-32m.json). The original
[CodeBERT configuration](configs/codebert-baseline.json) remains available for comparison.
Ettin uses its own tokenizer and a separate prepared-data directory. We keep the
512-token limit and the initial training hyperparameters for the first comparison;
accuracy, speed, and energy savings still need measurement.

For a first installation, copy `hpc/config.example.sh` to `hpc/config.sh`.
On the HPC, update these settings in your existing `hpc/config.sh` (this file is
ignored by Git, so `git pull` does not update it):

```bash
HPC_PARTITION=gpu-low,gpu-pre
HPC_ACCOUNT=greenml
HPC_MEM=16G
HPC_TIME=02:00:00
```

`greenml` is the account used by your earlier accepted job. Both partitions
must be available to that account. One job requests one GPU and four CPUs;
Slurm can select either partition. The smoke limit remains 20 minutes.
Python 3.11/3.12 is detected automatically; setup also handles missing `ensurepip`
without `sudo`. The 16 GB request is host RAM; GPU memory depends on the assigned card.

The default allows any GPU type in the eligible pool. To restrict the job to the
nodes advertising `gpu80g` in your listing (A100/H200 nodes), optionally add
`HPC_CONSTRAINT=gpu80g`; this reduces the pool and can increase waiting time.
The guide explains the scheduling tradeoffs and checkpoint resume.

From the repository root, after these changes are committed, pushed, and pulled
on the HPC:

```bash
# Reuses an existing environment; dependencies are unchanged for Ettin.
bash hpc/setup.sh

# Required even if CodeBERT data was prepared previously: Ettin has a different tokenizer.
bash hpc/prepare.sh

DRY_RUN=1 bash hpc/submit.sh smoke ettin-32m-smoke-001  # Preview the request.
bash hpc/submit.sh smoke ettin-32m-smoke-001
# Once the smoke job succeeds:
bash hpc/submit.sh train ettin-32m-001
```

If you are already inside `hpc/`, use `./prepare.sh` and
`./submit.sh smoke ettin-32m-smoke-001` instead.

The new defaults are:

- Configuration: `configs/ettin-32m.json`, with a pinned pretrained model revision.
- Prepared data: `~/nlbse27/prepared/ettin-32m-512`.
- Full-run output: `~/nlbse27/runs/ettin-32m-001`.
- Logs: `~/nlbse27/logs/ettin-JOBID.out` and `.err`.

Start the first full run without `RESUME`; smoke checkpoints belong to their
separate run. Invalid resume requests are rejected before a GPU job is submitted.
Two hours is the training job's time limit, not a runtime estimate. If it reaches
the limit after saving a checkpoint, continue the same run with:

```bash
RESUME=auto bash hpc/submit.sh train ettin-32m-001
```

After training, export `final_model/` and `run.json` and open
[the Colab evaluation notebook](colab/Ettin32M_evaluation.ipynb). It loads the
fine-tuned classifier, runs an example, and evaluates the frozen public test
split. The [HPC guide](hpc-unibz.md) contains the export and transfer commands.
The notebook is prepared for later use; training must produce the artifact first.
Its installation cell uses Colab-compatible evaluation dependencies and preserves
the installed `gcsfs` filesystem requirement. Restart the Colab session after
packages change. The HPC requirements remain unchanged.

ModernBERT uses native PyTorch SDPA attention with reference compilation disabled,
so no separate Flash Attention installation or compiler is required. Exported
models must be loaded with automatic model/tokenizer classes; the supplied
CodeBERT notebook's hard-coded RoBERTa classes do not support Ettin.

The supplied competition notebooks contain evaluation, not the original training
procedure. Our settings are explicit starting choices and do not claim to reproduce
the published CodeBERT weights or F1. For tuning, draw validation data from `train`
and reserve the public `test` results for reporting.

To train Ettin-17M alongside the 32M experiment, use a per-command model selection
after syncing these changes to the HPC. From `hpc/`:

```bash
HPC_MODEL=ettin-17m ./prepare.sh
HPC_MODEL=ettin-17m ./submit.sh smoke ettin-17m-smoke-001
# Once that smoke job completes:
HPC_MODEL=ettin-17m ./submit.sh train ettin-17m-001
```

This uses `configs/ettin-17m.json` and `~/nlbse27/prepared/ettin-17m-512`.
The output is `~/nlbse27/runs/ettin-17m-001`. The environment is shared, so no new
installation is needed. Keep `hpc/config.sh` as your cluster configuration; the
model prefix applies only to that command, leaving the default at 32M. Job names
include the selected model. Slurm decides whether the allocation permits the two
jobs to run concurrently or queues the second one.

If resuming the 17M run after an interruption, include both selections:

```bash
HPC_MODEL=ettin-17m RESUME=auto ./submit.sh train ettin-17m-001
```

To fine-tune the original CodeBERT backbone with the same three-epoch, FP32
settings, select its existing configuration. From `hpc/`:

```bash
HPC_MODEL=codebert-baseline ./prepare.sh
HPC_MODEL=codebert-baseline HPC_GRES=gpu:v100:1 ./submit.sh smoke codebert-smoke-001
# After the smoke job succeeds, using the GPU type of your completed Ettin run:
HPC_MODEL=codebert-baseline HPC_GRES=gpu:v100:1 ./submit.sh train codebert-001
```

Use `gpu:v100:1` only if the earlier run used a V100; check its `run.json` and
Slurm allocation. This creates separate `prepared/codebert-baseline-512` inputs,
`runs/codebert-001` output, and `logs/codebert-JOBID.{out,err}` logs. Existing
explicit `HPC_PREPARED`/`HPC_TRAIN_CONFIG` settings override model profiles;
remove stale overrides before switching. No environment reinstall is needed.
This is a controlled CodeBERT fine-tuning comparison with our training settings,
not an exact reconstruction of the organizers' trained checkpoint.

For past jobs and training-only durations, see
[recovering HPC timings](hpc-unibz.md#12-recover-completed-job-times-and-benchmark-codebert).

The same Colab notebook can evaluate either Ettin model. Export the 17M run using the
guide's commands with `ettin-17m-001`, then change `ARCHIVE`, `WORK_DIR`, and
`RESULT_DIR` in Colab to the 17M paths. It reads the model type and preprocessing
from the export rather than assuming a particular Ettin size.

To fine-tune [`tardellirs/crivo-17m`](https://huggingface.co/tardellirs/crivo-17m),
select the new profile. From the repository root after syncing to the HPC:

```bash
HPC_MODEL=crivo-17m bash hpc/prepare.sh
HPC_MODEL=crivo-17m bash hpc/submit.sh smoke crivo-17m-smoke-001
# After the smoke job succeeds:
HPC_MODEL=crivo-17m bash hpc/submit.sh train crivo-17m-001
# If interrupted after a checkpoint was saved:
HPC_MODEL=crivo-17m RESUME=auto bash hpc/submit.sh train crivo-17m-001
```

This uses the pinned checkpoint in `configs/crivo-17m.json`, separate inputs in
`~/nlbse27/prepared/crivo-17m-512`, output in `~/nlbse27/runs/crivo-17m-001`,
and `logs/crivo-JOBID.{out,err}`. Dependencies and training hyperparameters are
the same as Ettin-17M; an existing HPC environment needs no reinstall.
Remove any explicit `HPC_PREPARED`/`HPC_TRAIN_CONFIG` overrides before switching.

Crivo ships a custom per-option decision scorer. We transfer its entire trained
ModernBERT encoder, preserve its rotary-position settings, and train a fresh
two-output `[Vulnerability, MAT]` classifier on `Function` with the existing
multi-label loss. The exported classifier uses standard Transformers classes
and the same Colab evaluation notebook. See the
[Crivo HPC instructions](hpc-unibz.md#13-fine-tune-crivo-17m) for details.

Run checks:

```bash
python3 -m unittest discover -s tests -v
# With requirements.txt installed:
python tests/smoke_pipeline.py                         # tiny ModernBERT, offline
python tests/smoke_pipeline.py --architecture roberta  # original backbone compatibility
python tests/smoke_pipeline.py --architecture crivo    # custom decision encoder transfer
```

The integration checks cover preparation, training, deterministic resume,
evaluation, inference, and the portable cells of the Colab notebook.
Original competition resources remain in `code-comment-classification-main/`
and `old_colab/`. Detailed operation, profiling limits, and research references
are in [hpc-unibz.md](hpc-unibz.md).

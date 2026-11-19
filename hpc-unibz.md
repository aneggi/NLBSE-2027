# Fine-tune Ettin-32M on the Bolzano HPC

This workflow follows the task and model described in the supplied
[competition README](code-comment-classification-main/README.md) and
[evaluation notebook](code-comment-classification-main/CodeBERT_baseline.ipynb).
The default experiment trains an Ettin-32M encoder with a two-output classification head,
then evaluates the final model on the public `test` split.

## 1. Confirm the cluster settings and clone the repository

The [ScientificNet access page](https://hpc.scientificnet.org/index.html) says
the site requires ScientificNet Wi-Fi or VPN. Open the
[HPC help](https://hpc.scientificnet.org/help/) from that network to obtain the
login hostname, scheduler, GPU queue, software modules, storage policy, and
accounting requirements. Those details could not be verified from this workspace.
The scripts below assume **Slurm**; check that `sbatch` exists before using them.
If the cluster uses a different scheduler, use the Python commands in section 8
inside that scheduler's GPU allocation instead.

Connect using the actual login hostname supplied by the university:

```bash
ssh aaneggi@YOUR_HPC_LOGIN_HOST
```

Once these local files have been committed and pushed to your GitHub repository,
clone it into a persistent shared directory on the cluster:

```bash
git clone https://github.com/aneggi/NLBSE-2027.git
cd NLBSE-2027
```

For a private repository, use your existing GitHub credentials or SSH key and
`git clone git@github.com:aneggi/NLBSE-2027.git`. Later updates use `git pull --ff-only`
from this directory. Changes in your local working copy are available on the HPC
only after you commit and push them. This setup does not push or submit jobs for you.

For a Slurm cluster, inspect the configuration available to your account:

```bash
command -v sbatch
sinfo -o '%P %a %l %D %G'
module avail
```

Get any required accounting project/QoS from your allocation or the help page.
The SSH username `aaneggi` is not necessarily a Slurm account/project name.

## 2. Configure storage, Python, and GPU resources

```bash
cp hpc/config.example.sh hpc/config.sh
```

The current example uses the two partitions from your node listing and the
account shown in your earlier accepted job:

```bash
HPC_PARTITION=gpu-low,gpu-pre
HPC_ACCOUNT=greenml
```

Both partitions must allow that account; `sinfo` alone does not establish account
access. Change these values for another allocation. If you already have
`hpc/config.sh` on the HPC, edit it there: Git deliberately preserves this file.
Remove any old memory/time overrides or set `HPC_MEM=16G` and `HPC_TIME=02:00:00`.

Everything else has defaults: setup finds Python 3.11/3.12, stores files in
`$HOME/nlbse27`, and requests one GPU, four CPUs, 16 GB host RAM, and two hours
(20 minutes for a smoke job). Python modules already loaded in your shell work
for setup. If a module must also be loaded in batch jobs, add it to `HPC_MODULES`.

The storage and Python module lines remain optional. Uncomment them if your site
requires a different shared path (`HPC_WORKDIR`) or module (`HPC_MODULES`). The home-directory default
must be visible on compute nodes and have enough quota; use the shared storage
recommended by the university if needed. Allow tens of GB for data and checkpoints.

<details>
<summary>Advanced overrides (usually leave these alone)</summary>

Add any of these settings to `hpc/config.sh` when needed. Existing full
configuration files continue to work.

| Setting | Default / purpose |
| --- | --- |
| `HPC_MODEL` | `ettin-32m`; prefix a command with `HPC_MODEL=ettin-17m` for the other model. |
| `HPC_PYTHON` | Automatically finds `python3.11`, `python3.12`, or a compatible `python3`; set an executable to override. |
| `HPC_VENV` | `$HPC_WORKDIR/venv` |
| `HPC_PREPARED` | `$HPC_WORKDIR/prepared/$HPC_MODEL-512`; follows the selected model. |
| `HPC_TRAIN_CONFIG` | Repository's `configs/$HPC_MODEL.json` |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu124`; change if the allocated GPU driver needs another wheel. |
| `HPC_QOS` | Empty; set if required by your allocation. |
| `HPC_GRES` | `gpu:1`; change if the site requires a GPU type in the resource request. |
| `HPC_CONSTRAINT` | Empty, allowing any eligible GPU node; set `gpu80g` to filter to that node feature. |
| `HPC_CPUS`, `HPC_MEM` | `4`, `16G` |
| `HPC_TIME`, `HPC_SMOKE_TIME` | `02:00:00`, `00:20:00`; adjust to queue limits. |

</details>

`submit.sh` passes the resource options directly to `sbatch`, equivalent to your
`#SBATCH` example. It records output under `$HPC_WORKDIR/logs/ettin-JOBID.out`
and `ettin-JOBID.err`. There is no need to paste a second set of directives into
the job script. The 16 GB limit is system RAM, not GPU VRAM. Inspect the smoke
job's `MaxRSS` with `sacct` to check host-memory needs before a full run.

With a comma-separated partition list, Slurm considers the earliest eligible
start, subject to partition priorities; the list order does not rank GPU speed.
See the official [partition option](https://slurm.schedmd.com/sbatch.html#OPT_partition).
One job uses one GPU in one partition. The default is not tied to a hostname or
GPU model, so it can use the V100, A40, A100, or H200 pool shown in your snapshot.
Availability changes while a job is queued; let Slurm place the allocation.

If you prefer the A100/H200 nodes in that listing over the broader pool, add
`HPC_CONSTRAINT=gpu80g`. Based on the features you supplied, this filters to
nodes carrying that feature and excludes the listed V100/A40 nodes. It is a
hard node filter, not a GPU memory reservation or a guarantee of earlier completion;
it can increase queue time. Leave it unset for the broader pool. The meaning of
`MIXED` concerns allocated resources and does not prove a GPU is free; see
[Slurm node states](https://slurm.schedmd.com/sinfo.html#SECTION_NODE-STATE-CODES).
Check actual allocation limits and preemption policy in the university help or
`scontrol show partition gpu-low` and `scontrol show partition gpu-pre`.

The repository must also be visible from compute nodes. `hpc/config.sh` is
ignored by Git so pulling updates preserves your machine-specific settings.
For an alternate configuration file, set an absolute path:

```bash
export HPC_CONFIG=/absolute/path/to/my-hpc-config.sh
```

When switching from CodeBERT, keep the existing cluster configuration. If an
older `hpc/config.sh` explicitly sets `HPC_TRAIN_CONFIG` or `HPC_PREPARED` to
CodeBERT paths, remove those two overrides to use the new Ettin defaults.
Run preparation again for Ettin; the old CodeBERT prepared directory and runs
remain available. Use the new run names below to avoid a name collision.

## 3. Install the environment and prepare inputs once

Run on a node where installing packages and downloading data are permitted by
the site. This does not fine-tune a model or require a GPU. If login-node policy
forbids tokenization, run preparation through an approved CPU allocation instead.

```bash
bash hpc/setup.sh
bash hpc/prepare.sh
```

If the system Python lacks `ensurepip` (the `python3-venv` package), setup
automatically uses the official [virtualenv zipapp](https://virtualenv.pypa.io/en/stable/how-to/install.html).
It downloads the bootstrap into `$HPC_WORKDIR/cache/virtualenv` and creates the
environment in your own workspace. No administrator access or system package
installation is needed. An incomplete environment from an earlier failed attempt
is repaired in place; rerun setup without deleting the environment directory.
This fallback requires HTTPS access to `bootstrap.pypa.io` during setup.

Setup supports Python 3.11/3.12 and creates an isolated virtual environment with the pinned versions in
`requirements.txt`; it records all installed versions in
`$HPC_WORKDIR/environment-freeze.txt`. Local integration was checked with
Python 3.12 and CPU PyTorch on ModernBERT and RoBERTa, including deterministic
checkpoint resume and exported-model evaluation. The CodeBERT smoke job has run
on the HPC; full Ettin training still needs to run there.
PyTorch's
[previous-version installation instructions](https://pytorch.org/get-started/previous-versions/)
document the available CUDA and CPU wheel indexes. Verify driver compatibility
on the actual allocated GPU node, not just the login node.

Preparation downloads the pinned `jhu-clsp/ettin-encoder-32m` checkpoint and the competition dataset,
resolves their Hub revisions to commit hashes, and writes:

```text
$HPC_PREPARED/
  base-model/     # pretrained backbone and tokenizer
  train/          # tokenized Arrow dataset
  test/           # tokenized public test dataset
  manifest.json   # resolved revisions, label order, split sizes, environment
```

The input is exactly `Function`, preserving its comments. The existing dataset
is already formatted; this workflow does not rebuild MADE-WIC, remove comments,
rebalance classes, or create a new split. It tokenizes with Ettin's fast tokenizer,
truncates to 512 tokens, and pads to that limit. Preparing inputs uses
public test text and labels for tokenization only; training never reads that split.
An existing prepared directory is protected against replacement. Change
`HPC_PREPARED` to a new path when changing the model, dataset revision, or token limit.
Reusing it preserves the original resolved inputs even if Hub `main` changes.

GPU jobs set Hugging Face offline flags. They need the prepared directory and
environment on shared storage, but no internet access or Hub login.

## 4. Run a small GPU smoke job

Preview the exact submission without enqueueing it:

```bash
DRY_RUN=1 bash hpc/submit.sh smoke ettin-32m-smoke-001
```

Submit:

```bash
bash hpc/submit.sh smoke ettin-32m-smoke-001
squeue -u aaneggi
```

This trains for two optimizer steps on at most 32 training examples and evaluates
at most 16 public test examples. Inspect both `.out` and `.err` logs under
`$HPC_WORKDIR/logs`, plus the run's `evaluation-JOBID/metrics.json`.
Smoke results are pipeline checks and must not be reported as competition results.

If a job exits immediately, inspect its `.err` file and check
`sacct -j JOBID --format=JobID,State,Elapsed,ExitCode`. An error looking for
`common.sh` inside `/var/lib/slurm/...` means the job script is an older copy:
update `hpc/job.sh` before resubmitting. The current script loads helpers through
the repository path exported by `submit.sh`, because Slurm runs a spool copy of
the script. If the job failed before creating a run directory, the same run name
can be submitted again; setup and data preparation do not need to be repeated.

On CUDA out-of-memory errors, reduce `batch_size` in the JSON configuration and
raise `gradient_accumulation_steps` to preserve the effective training batch size.
Reduce `eval_batch_size` separately for evaluation. Use a new run name after
changing the training configuration. Optional `fp16` or `bf16` training is
available through `precision`; bf16 requires hardware support. The starting
configuration uses fp32. Changes in precision can change results.

## 5. Train and evaluate Ettin-32M

After the smoke job succeeds:

```bash
bash hpc/submit.sh train ettin-32m-001
squeue -u aaneggi
```

Each submitted job requests one GPU and refuses to train if CUDA is unavailable.
It trains using only the public training split, saves the final model, and evaluates
it once on the full public `test` split. There is no test-based early stopping,
threshold tuning, or checkpoint selection. Keep this public test split out of
future hyperparameter searches; create validation data from `train` for that work.

The current competition repository identifies public `train` and `test` splits;
the older text in `old_colab` calls the public evaluation set validation.
The current [organiser repository](https://github.com/nlbse2027/code-comment-classification)
is the reference for this implementation. Actual downloaded sizes are recorded
in the prepared manifest and reported with every evaluation.

The supplied notebooks contain inference on the published checkpoint, not its
training code or optimizer settings. The following are explicit initial choices
for this implementation, rather than recovered original settings:

| Parameter | Starting value |
| --- | --- |
| Initialization | `jhu-clsp/ettin-encoder-32m`, new two-output classifier |
| Pooling | Mean pooling, inherited from the Ettin checkpoint |
| Attention | PyTorch SDPA; reference compilation disabled |
| Targets | `[Vulnerability, MAT]`, float labels |
| Loss | Unweighted binary cross entropy with logits |
| Optimizer | AdamW, learning rate `2e-5`, weight decay `0.01` |
| Schedule | Linear decay, warmup ratio `0.1` |
| Epochs | `3` |
| Batch | `8` per device, accumulation `2` → effective `16` on one GPU |
| Gradient clipping | Norm `1.0` |
| Precision and seed | fp32, `42` |
| Checkpoints | Every `500` optimizer steps, retain the last `2` |
| Evaluation | Batch `16`, sigmoid probability strictly greater than `0.5` |

The task, label order, token limit, and threshold match the competition baseline.
Ettin changes the encoder, tokenizer, and pooling. The model revision is pinned
in `configs/ettin-32m.json`; resolved dataset revisions remain in the prepared
manifest. Other starting hyperparameters match our CodeBERT configuration for
the first comparison. Neither predictive accuracy nor resource savings are
assumed; compare full runs on the same hardware. Full training time depends on
the assigned GPU. Two hours is a wall-time cap for each training allocation,
not a promise that three epochs finish in that time. If needed, use successive
manual checkpoint resumes without changing the training configuration.

## 6. Find results, resume, or evaluate again

```text
$HPC_WORKDIR/
  environment-freeze.txt
  cache/huggingface/
  logs/ettin-JOBID.{out,err}
  runs/ettin-32m-001/
    run.json                       # config, inputs, environment, Git state, GPU
    checkpoint-*/                  # model, optimizer, scheduler, RNG, trainer state
    final_model/                   # model, tokenizer, preprocessing.json
    train_results.json
    trainer_state.json
    evaluation-JOBID/
      metrics.json
      predictions.jsonl            # public-test order, zero-based indices
```

`metrics.json` reports each task's accuracy, precision, recall, F1, confusion
counts, and their arithmetic mean F1. `full_public_test` states whether the
entire public test split was evaluated. Training sample/step limits are recorded
in `run.json`. Check these before reporting results.

After a time limit or interruption, resume the latest saved checkpoint. This
is also useful if a selected partition's policy permits preemption; the actual
policy must be checked, not inferred from partition names:

Use `RESUME=auto` only after the same full run has created checkpoints. For the
first full training launch, use `bash hpc/submit.sh train ettin-32m-001` without
`RESUME`. A successful smoke job creates a separate run and does not initialize
the full training run. The submit wrapper rejects missing runs/checkpoints before
requesting GPU resources.

```bash
RESUME=auto bash hpc/submit.sh train ettin-32m-001
```

Use the same configuration and prepared inputs. Resume restores optimizer,
scheduler, and RNG state; work since the latest completed checkpoint is repeated.
If interrupted before the first checkpoint, start with a new run name.
A run with a final model is protected against another training launch. If only
evaluation failed, run it separately:

```bash
bash hpc/submit.sh evaluate ettin-32m-001
```

For Slurm status and cancellation, substitute the real job ID:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel JOBID
```

Repeated full-test evaluations are useful for checking artifacts; do not use
their results to choose training hyperparameters.

## 7. Predict on new functions

The saved `final_model` is self-contained. Activate the environment, then supply
UTF-8 JSONL with a `Function` string and optional `id` on each line:

```json
{"id": "example-1", "Function": "int f() { /* TODO: fix */ return 0; }"}
```

```bash
# Use your configured environment path; inference belongs in an allocated
# GPU session or a permitted CPU session, not a large login-node workload.
source /absolute/path/to/nlbse27/venv/bin/activate
python scripts/nlbse.py predict \
  --model /absolute/path/to/nlbse27/runs/ettin-32m-001/final_model \
  --input functions.jsonl --output predictions.jsonl --batch-size 16
```

Output preserves the optional `id` and contains `predictions` and `probabilities`,
both in `[Vulnerability, MAT]` order. Input and output are streamed in batches;
labels are not required. Token length comes from the saved preprocessing settings.
This JSONL command is a local inference interface; adapt it to any final organiser
submission interface rather than assuming the competition accepts JSONL.

## 8. Direct Python commands and optional performance profiling

The Python entry point also works inside other scheduler allocations. With the
environment active, replace the paths below with absolute shared-storage paths:

```bash
python scripts/nlbse.py prepare --prepared /shared/nlbse/prepared
python scripts/nlbse.py train --prepared /shared/nlbse/prepared \
  --output /shared/nlbse/runs/ettin-32m-001 --require-cuda
python scripts/nlbse.py evaluate --prepared /shared/nlbse/prepared \
  --model /shared/nlbse/runs/ettin-32m-001/final_model \
  --output /shared/nlbse/runs/ettin-32m-001/evaluation --require-cuda
```

For a separate repeated inference benchmark, run in a GPU allocation:

```bash
python scripts/nlbse.py evaluate --prepared /shared/nlbse/prepared \
  --model /shared/nlbse/runs/ettin-32m-001/final_model \
  --output /shared/nlbse/runs/ettin-32m-001/benchmark \
  --repetitions 10 --profile-flops --require-cuda
```

Reported runtime and optional GFLOPs are totals per full dataset pass, averaged
over repetitions, not per-function values. CUDA is synchronized around timing.
Forward timing excludes tokenization and host-to-device transfer; profiler overhead
is included when profiling is enabled. PyTorch FLOP counts cover supported operators
and can omit fused attention operations. Evaluation uses fp32; the supplied notebook
uses CUDA autocast and a different batch size. These measurements are for local
comparison and are not an exact reproduction of the notebook's timing or the
official ranking. Run final competition benchmarking in the organiser's common
environment using their evaluator. This script does not calculate an official score.

## 9. Research references and future experiments

- [Competition resources](https://github.com/nlbse2027/code-comment-classification):
  authoritative task, public split semantics, model links, and evaluation notebook.
- [MADE-WIC](https://doi.org/10.1145/3691620.3695348): origin of the multi-annotated data.
- [VulSATD](https://arxiv.org/html/2501.15934v1): a related shared-CodeBERT approach.
  Its experiments use different input preparation and dataset splits. Section IV-C
  selects learning rate `2e-5`, 10 epochs, batch 16, dropout 0.1, and zero L2 penalty;
  those are not evidence for the undisclosed competition checkpoint settings.
- [CodeBERT](https://aclanthology.org/2020.findings-emnlp.139/): encoder pretraining.
- [Ettin model](https://huggingface.co/jhu-clsp/ettin-encoder-32m) and
  [paper](https://arxiv.org/abs/2507.11412): the smaller ModernBERT-based encoder
  used for this experiment.
- [Transformers Trainer 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer):
  training/checkpoint API used here.

First save a complete baseline run and its manifests. For subsequent experiments,
copy the JSON configuration, point `HPC_TRAIN_CONFIG` to it, and use a fresh run name.
Changing models or tokenization also requires a new prepared directory. Other
encoder models need compatible tokenizers and sequence-classification heads;
this workflow is currently limited to inputs of at most 512 tokens. New techniques
such as different comment handling or weighted losses should be separate changes,
evaluated using a validation partition drawn only from the training split.

## 10. Export the trained model and evaluate in Colab

Once `ettin-32m-001` finishes successfully, run on the HPC:

```bash
tar -czf "$HOME/nlbse27/ettin-32m-001.tar.gz" \
  -C "$HOME/nlbse27/runs/ettin-32m-001" final_model run.json
```

Adjust the paths if you configured a different `HPC_WORKDIR`. Include both the
model and tokenizer files in `final_model/`, plus `run.json`. The training data,
prepared cache, and optimizer checkpoints are not needed for Colab evaluation.

From your own computer, download the archive through the university's actual
SSH hostname (with VPN connected if required):

```bash
scp aaneggi@YOUR_HPC_LOGIN_HOST:~/nlbse27/ettin-32m-001.tar.gz .
```

Upload the archive to Google Drive, for example under `MyDrive/NLBSE-2027/`.
Open [colab/Ettin32M_evaluation.ipynb](colab/Ettin32M_evaluation.ipynb) in Colab,
choose a GPU runtime, and set `ARCHIVE` to the uploaded file. Run the cells in
order. The notebook loads the exported classifier and its saved preprocessing,
tries one example, then evaluates the public test split at the exact revision
recorded during HPC preparation. Set `MAX_TEST_SAMPLES=16` for a quick check;
set it back to `None` for the complete public test.

The notebook keeps Colab's GPU-enabled PyTorch and installs the same Transformers
and Datasets versions as training. It explicitly disables ModernBERT reference
compilation on load, because Transformers omits that setting from `config.json`.
Do not replace the checkpoint with the original pretrained Ettin Hub model:
that would discard the fine-tuned classifier. The old competition notebook's
`RobertaForSequenceClassification` and `RobertaTokenizer` must be replaced by
`AutoModelForSequenceClassification` and `AutoTokenizer` for Ettin.

The Colab notebook exports task metrics, predictions, and optional local runtime
and FLOP estimates. Its portable evaluation cells are exercised locally; Google
Drive authorization and an actual hosted Colab GPU session are later steps.
For the official ranking, use the organisers' required evaluation environment.

The same notebook also supports Ettin-17M: export its `final_model/` and `run.json`
from `runs/ettin-17m-001` and use a different archive/work/results path in Colab.

## 11. Run Ettin-17M alongside Ettin-32M

From `hpc/`, select the smaller model on each command:

```bash
HPC_MODEL=ettin-17m ./prepare.sh
HPC_MODEL=ettin-17m ./submit.sh smoke ettin-17m-smoke-001
# After the smoke job succeeds:
HPC_MODEL=ettin-17m ./submit.sh train ettin-17m-001
```

The 17M configuration pins `jhu-clsp/ettin-encoder-17m` and otherwise uses the
same initial hyperparameters as 32M: 512 tokens, three epochs, effective batch
16, and fp32. Model loading, attention, tokenizer handling, and evaluation use
the existing ModernBERT path. The new prepared directory contains its own
pretrained backbone, and the separate run name keeps checkpoints apart.

No setup reinstall or global model switch is needed. The temporary assignment
is passed to the submitted job and does not alter an already running 32M job.
Concurrent execution depends on the allocation's GPU/job limits; the second job
may wait in the queue. Node choice is still controlled by the existing cluster
settings. Compare model performance and runtime on equivalent hardware.

If your cluster configuration has explicit `HPC_TRAIN_CONFIG` or `HPC_PREPARED`
overrides, remove those overrides before using model profiles; otherwise they
take precedence over the automatic model-specific paths. The minimal cluster
configuration used in this session does not contain them.

To resume the same 17M training run after it has saved checkpoints:

```bash
HPC_MODEL=ettin-17m RESUME=auto ./submit.sh train ettin-17m-001
```

To run CodeBERT again for comparison, add these explicit overrides to your
cluster configuration and use a CodeBERT run name:

```bash
HPC_TRAIN_CONFIG="$REPO_ROOT/configs/codebert-baseline.json"
HPC_PREPARED="$HPC_WORKDIR/prepared/codebert-512"
```

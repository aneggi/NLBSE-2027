# Run the NLBSE 2027 baseline on the Bolzano HPC

This workflow follows the task and model described in the supplied
[competition README](code-comment-classification-main/README.md) and
[evaluation notebook](code-comment-classification-main/CodeBERT_baseline.ipynb).
It trains one shared CodeBERT encoder with a two-output classification head,
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

**Set `HPC_PARTITION` to your GPU partition.** This is the only required setting
in the configuration file. Use the output of `sinfo` or the university help to
find the correct name.

Everything else has defaults: setup finds Python 3.11/3.12, stores files in
`$HOME/nlbse27`, and requests one GPU, four CPUs, 32 GB host RAM, and 24 hours
(20 minutes for a smoke job). Python modules already loaded in your shell work
for setup. If a module must also be loaded in batch jobs, add it to `HPC_MODULES`.

The example has three optional lines. Uncomment them only if your site requires
a different shared storage path (`HPC_WORKDIR`), an accounting project
(`HPC_ACCOUNT`), or a Python module (`HPC_MODULES`). The home-directory default
must be visible on compute nodes and have enough quota; use the shared storage
recommended by the university if needed. Allow tens of GB for data and checkpoints.

<details>
<summary>Advanced overrides (usually leave these alone)</summary>

Add any of these settings to `hpc/config.sh` when needed. Existing full
configuration files continue to work.

| Setting | Default / purpose |
| --- | --- |
| `HPC_PYTHON` | Automatically finds `python3.11`, `python3.12`, or a compatible `python3`; set an executable to override. |
| `HPC_VENV` | `$HPC_WORKDIR/venv` |
| `HPC_PREPARED` | `$HPC_WORKDIR/prepared/codebert-512`; change for another model or tokenization. |
| `HPC_TRAIN_CONFIG` | Repository's `configs/codebert-baseline.json` |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu124`; change if the allocated GPU driver needs another wheel. |
| `HPC_QOS` | Empty; set if required by your allocation. |
| `HPC_GRES` | `gpu:1`; change if the site requires a GPU type in the resource request. |
| `HPC_CPUS`, `HPC_MEM` | `4`, `32G` |
| `HPC_TIME`, `HPC_SMOKE_TIME` | `24:00:00`, `00:20:00`; adjust to queue limits. |

</details>

The repository must also be visible from compute nodes. `hpc/config.sh` is
ignored by Git so pulling updates preserves your machine-specific settings.
For an alternate configuration file, set an absolute path:

```bash
export HPC_CONFIG=/absolute/path/to/my-hpc-config.sh
```

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
Python 3.12 and CPU PyTorch on a tiny random RoBERTa model, including deterministic
checkpoint resume. Full CodeBERT training and the actual cluster remain untested.
PyTorch's
[previous-version installation instructions](https://pytorch.org/get-started/previous-versions/)
document the available CUDA and CPU wheel indexes. Verify driver compatibility
on the actual allocated GPU node, not just the login node.

Preparation downloads `microsoft/codebert-base` and the competition dataset,
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
rebalance classes, or create a new split. It tokenizes with the slow RoBERTa
tokenizer, truncates to 512 tokens, and pads to that limit. Preparing inputs uses
public test text and labels for tokenization only; training never reads that split.
An existing prepared directory is protected against replacement. Change
`HPC_PREPARED` to a new path when changing the model, dataset revision, or token limit.
Reusing it preserves the original resolved inputs even if Hub `main` changes.

GPU jobs set Hugging Face offline flags. They need the prepared directory and
environment on shared storage, but no internet access or Hub login.

## 4. Run a small GPU smoke job

Preview the exact submission without enqueueing it:

```bash
DRY_RUN=1 bash hpc/submit.sh smoke smoke-001
```

Submit:

```bash
bash hpc/submit.sh smoke smoke-001
squeue -u aaneggi
```

This trains for two optimizer steps on at most 32 training examples and evaluates
at most 16 public test examples. Inspect both `.out` and `.err` logs under
`$HPC_WORKDIR/logs`, plus the run's `evaluation-JOBID/metrics.json`.
Smoke results are pipeline checks and must not be reported as competition results.

On CUDA out-of-memory errors, reduce `batch_size` in the JSON configuration and
raise `gradient_accumulation_steps` to preserve the effective training batch size.
Reduce `eval_batch_size` separately for evaluation. Use a new run name after
changing the training configuration. Optional `fp16` or `bf16` training is
available through `precision`; bf16 requires hardware support. The starting
configuration uses fp32. Changes in precision can change results.

## 5. Train and evaluate the full baseline

After the smoke job succeeds:

```bash
bash hpc/submit.sh train codebert-001
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
| Initialization | `microsoft/codebert-base`, new two-output head |
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

The shared encoder, output head, label order, truncation, and threshold follow
the provided competition baseline. Scores need not match the published model:
its original fine-tuning settings are missing, and hardware/library differences
can affect results. Full training time depends on the assigned GPU. The 24-hour
template limit is a starting request, not a measured estimate.

## 6. Find results, resume, or evaluate again

```text
$HPC_WORKDIR/
  environment-freeze.txt
  cache/huggingface/
  logs/nlbse-train-JOBID.{out,err}
  runs/codebert-001/
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

After a time limit or interruption, resume the latest saved checkpoint:

```bash
RESUME=auto bash hpc/submit.sh train codebert-001
```

Use the same configuration and prepared inputs. Resume restores optimizer,
scheduler, and RNG state; work since the latest completed checkpoint is repeated.
If interrupted before the first checkpoint, start with a new run name.
A run with a final model is protected against another training launch. If only
evaluation failed, run it separately:

```bash
bash hpc/submit.sh evaluate codebert-001
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
  --model /absolute/path/to/nlbse27/runs/codebert-001/final_model \
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
  --output /shared/nlbse/runs/codebert-001 --require-cuda
python scripts/nlbse.py evaluate --prepared /shared/nlbse/prepared \
  --model /shared/nlbse/runs/codebert-001/final_model \
  --output /shared/nlbse/runs/codebert-001/evaluation --require-cuda
```

For a separate repeated inference benchmark, run in a GPU allocation:

```bash
python scripts/nlbse.py evaluate --prepared /shared/nlbse/prepared \
  --model /shared/nlbse/runs/codebert-001/final_model \
  --output /shared/nlbse/runs/codebert-001/benchmark \
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
- [Transformers Trainer 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer):
  training/checkpoint API used here.

First save a complete baseline run and its manifests. For subsequent experiments,
copy the JSON configuration, point `HPC_TRAIN_CONFIG` to it, and use a fresh run name.
Changing models or tokenization also requires a new prepared directory. Other
encoder models need compatible tokenizers and sequence-classification heads;
this workflow is currently limited to inputs of at most 512 tokens. New techniques
such as different comment handling or weighted losses should be separate changes,
evaluated using a validation partition drawn only from the training split.

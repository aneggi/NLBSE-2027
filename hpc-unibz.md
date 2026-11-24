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
| `HPC_MODEL` | `ettin-32m`; select `ettin-17m`, `crivo-17m`, or `codebert-baseline` per command. |
| `HPC_PYTHON` | Automatically finds `python3.11`, `python3.12`, or a compatible `python3`; set an executable to override. |
| `HPC_VENV` | `$HPC_WORKDIR/venv` |
| `HPC_PREPARED` | `$HPC_WORKDIR/prepared/$HPC_MODEL-512`; follows the selected model. |
| `HPC_TRAIN_CONFIG` | Repository's `configs/$HPC_MODEL.json` |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu124`; change if the allocated GPU driver needs another wheel. |
| `HPC_QOS` | Empty; set if required by your allocation. |
| `HPC_GRES` | `gpu:1`; change if the site requires a GPU type in the resource request. |
| `HPC_CONSTRAINT` | Empty, allowing any eligible GPU node; set `gpu80g` to filter to that node feature. |
| `HPC_NODELIST` | Empty; optionally select a verified previous node for hardware comparisons. |
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

The notebook keeps Colab's GPU-enabled PyTorch and uses evaluation versions of
Transformers/Datasets compatible with current Colab packages. HPC training stays
on its original pinned requirements. The install cell preserves the `fsspec`
requirement of installed `gcsfs` and uses a Hub version compatible with the
reported Gradio/Diffusers requirements. Restart the Colab session after changing
packages, before loading the model. Evaluation library versions are recorded in
the metrics; old-library exports were checked against the updated inference path.
The notebook explicitly disables ModernBERT reference
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

## 12. Recover completed job times and benchmark CodeBERT

Run these commands in your SSH session. `squeue` shows the current queue;
`sacct` queries completed jobs if the cluster retains their accounting records.
List the past 90 days, using only parent allocations to avoid counting `.batch`
and `.extern` steps again:

```bash
sacct -u aaneggi -S now-90days -X \
  --format=JobID,JobName%30,State,Elapsed,NodeList%20,AllocTRES%100
```

The earlier screenshots showed custom site formatting. An explicit `--format`
selects the fields needed here. For one job, replace `JOBID` with its numeric ID:

```bash
sacct -j JOBID -X \
  --format=JobID,JobName%30,State,Submit,Start,End,Elapsed,ElapsedRaw,NodeList%20,AllocTRES%100,ExitCode
```

`Elapsed` is job elapsed wall time; `ElapsedRaw` expresses it in seconds.
Queue waiting is separate: inspect `Submit` and `Start`. `CPUTime` is elapsed
time multiplied by allocated CPUs and is not training wall time. `AllocTRES`
can identify the allocated GPU type, while `NodeList` identifies the host. The
run's `run.json` also stores PyTorch's actual GPU name.
See the [official Slurm field definitions](https://slurm.schedmd.com/sacct.html).
An empty result does not prove the job never ran: extend the date range or ask
the HPC team about accounting retention.

Export the allocation history as a pipe-separated file:

```bash
sacct -u aaneggi -S now-90days -X --parsable2 \
  --format=JobID,JobName,State,ElapsedRaw,NodeList,AllocTRES,Start,End \
  > "$HOME/nlbse27/job-history.psv"
```

Our `train` job trains and then evaluates the complete public test set.
Consequently, job elapsed time includes startup, model loading, training,
checkpoint saving, and evaluation. To inspect the Trainer's training-loop
duration, run this lightweight command on the login node. It uses the default
workspace; adjust the path if `HPC_WORKDIR` differs:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for path in sorted((Path.home() / "nlbse27/runs").glob("*/train_results.json")):
    run_dir = path.parent
    metrics = json.loads(path.read_text())
    run = json.loads((run_dir / "run.json").read_text())
    seconds = metrics.get("train_runtime")
    print(f"{run_dir.name}: GPU={run.get('device', 'unknown')}; "
          f"train_seconds={seconds}; "
          f"train_hours={seconds / 3600 if seconds is not None else None}; "
          f"epochs={metrics.get('epoch')}; "
          f"max_steps={run.get('max_steps')}; "
          f"max_train_samples={run.get('max_train_samples')}")
PY
```

These are CPU-only JSON reads, not model execution. Ignore smoke runs when
comparing full training. The Trainer timer is scoped to the training loop;
loading, final export, and test inference are outside it. After checkpoint
resume, `train_results.json` describes the final invocation and does not recover
all earlier attempts. For an interrupted run, identify every job from its
`.out` header (`Job ...; run ...`) and sum parent-allocation `ElapsedRaw` values
to obtain cumulative allocated job time. Include timeout/preempted attempts;
do not sum `.batch`/`.extern` rows or label the sum as pure training time.
Without historical stage timers, exact cumulative training-only time may not
be recoverable. Preserve the files before rerunning anything.

For A100 runs, the helper below matches retained `.out` headers to run names,
queries parent-job accounting once, and sums training-job allocations including
timeouts. It prints the final Trainer invocation separately and excludes
evaluation-only submissions. Run from the repository root:

```bash
python3 scripts/training_times.py
```

If a particular attempt's output log was deleted, include its verified job ID:

```bash
python3 scripts/training_times.py --extra-job codebert-a100-002=277034
```

Use that example only if job 277034 belongs to that run. Totals cover retained
logs and explicit extra IDs, not unknown/deleted attempts. Missing accounting
or still-active attempts are marked incomplete. All three models' allocation
totals share the same timing scope; they are not pure training-loop durations.

For existing HPC evaluations, `runs/RUN/evaluation-JOBID/metrics.json` contains
`wall_seconds` for the evaluation loop and `forward_seconds_per_pass` for the
average forward-only pass. Neither is the training duration. These HPC results
also do not replace the competition's T4 benchmark.

The competition's original backbone is
[`microsoft/codebert-base`](https://github.com/nlbse2027/code-comment-classification#codebert-baseline).
The organizer checkpoint is already fine-tuned. To compare the cost of adapting
the original backbone against Ettin, start from pretrained CodeBERT and use
the existing `configs/codebert-baseline.json`: three epochs, FP32, 512 tokens,
batch 8 and accumulation 2, with the same optimizer settings as our Ettin runs.
This controls our training procedure; it does not claim to reconstruct an
unpublished organizer training recipe.

After syncing these repository changes to the cluster, from `hpc/`:

```bash
HPC_MODEL=codebert-baseline ./prepare.sh

# Example only if the completed Ettin run used a V100 allocation:
HPC_MODEL=codebert-baseline HPC_GRES=gpu:v100:1 DRY_RUN=1 \
  ./submit.sh smoke codebert-smoke-001
HPC_MODEL=codebert-baseline HPC_GRES=gpu:v100:1 \
  ./submit.sh smoke codebert-smoke-001

# Once smoke succeeds:
HPC_MODEL=codebert-baseline HPC_GRES=gpu:v100:1 \
  ./submit.sh train codebert-001
```

No setup reinstall is needed. Remove stale explicit `HPC_TRAIN_CONFIG` and
`HPC_PREPARED` overrides in `config.sh` before selecting a model profile. New
inputs go to `prepared/codebert-baseline-512`; the output is `runs/codebert-001`;
logs are `logs/codebert-JOBID.out` and `.err`. Existing Ettin paths are preserved.
Preparation protects an existing cache against replacement; reuse a verified
matching cache rather than repeatedly preparing into the same directory.

`HPC_GRES=gpu:v100:1` uses the typed GRES name in your supplied node listing.
It restricts the GPU family, but V100 variants can still differ. Inspect the
completed Ettin run's actual GPU and node first. If the previous node was, for
example, `hpcsgn07`, optionally add `HPC_NODELIST=hpcsgn07` to the submission
commands; substitute your real node. This limits placement and may increase
waiting time; it does not reserve a particular physical GPU on a multi-GPU node.
Check the dry-run resource request, especially if `config.sh` sets GPU options
that override command-prefixed values. Remove a conflicting `gpu80g` constraint
before requesting a V100. See Slurm's
[typed GPU requests](https://slurm.schedmd.com/gres.html) and
[node-list option](https://slurm.schedmd.com/sbatch.html#OPT_nodelist).

Match the prepared manifests' dataset revision and split sizes, GPU variant,
FP32 precision, effective batch, token limit, and epoch count across runs.
Host load and software versions can still affect timing. A lower parameter
count alone does not establish a measured speedup or power saving.

The default two-hour allocation cap remains in force. If training times out
after creating a checkpoint, resume with the same hardware selection:

```bash
HPC_MODEL=codebert-baseline HPC_GRES=gpu:v100:1 RESUME=auto \
  ./submit.sh train codebert-001
```

Use `RESUME=auto` only for a run with saved checkpoints. If the final model exists
and only evaluation failed, use the same profile with `./submit.sh evaluate
codebert-001` instead of retraining. Preserve all job IDs for total-time reporting.

## 13. Fine-tune Crivo-17M

After pulling the updated repository on the HPC, run from the repository root:

```bash
HPC_MODEL=crivo-17m bash hpc/prepare.sh
HPC_MODEL=crivo-17m DRY_RUN=1 bash hpc/submit.sh smoke crivo-17m-smoke-001
HPC_MODEL=crivo-17m bash hpc/submit.sh smoke crivo-17m-smoke-001
# After the smoke job succeeds:
HPC_MODEL=crivo-17m bash hpc/submit.sh train crivo-17m-001
```

From inside `hpc/`, use `./prepare.sh` and `./submit.sh` with the same prefixes.
An existing environment needs no reinstall. Remove explicit `HPC_TRAIN_CONFIG`
or `HPC_PREPARED` overrides before selecting the new profile.

`configs/crivo-17m.json` pins
[`tardellirs/crivo-17m`](https://huggingface.co/tardellirs/crivo-17m/tree/8e42a0dbc309ca3360bfa94ebfd30061d6994ba7)
at revision `8e42a0dbc309ca3360bfa94ebfd30061d6994ba7`. It keeps the Ettin-17M
training settings: three epochs, 512 tokens, FP32, learning rate 2e-5, batch 8
and accumulation 2. New inputs go to `~/nlbse27/prepared/crivo-17m-512`, output
to `~/nlbse27/runs/crivo-17m-001`, and logs to `~/nlbse27/logs/crivo-JOBID.{out,err}`.
The default profile remains Ettin-32M. Preparation must run once for Crivo even
if Ettin inputs already exist; the checkpoint and tokenizer belong to this profile.

The [upstream model source](https://huggingface.co/tardellirs/crivo-17m/blob/8e42a0dbc309ca3360bfa94ebfd30061d6994ba7/model.py)
saves a ModernBERT encoder under `encoder.*` plus a custom per-option decision
head. Our loader transfers every encoder tensor with strict key and shape
checks, preserves both rotary-position bases from its newer configuration, and
initializes a fresh standard two-output classifier. The entire encoder is
fine-tuned on `Function` for independent `[Vulnerability, MAT]` labels with the
existing multi-label loss. This experiment uses Crivo's trained encoder weights;
it does not use its typed-question format or original decision scorer. No
upstream Python code is executed, and the HPC dependencies stay unchanged.

If training stops after saving a checkpoint, resume the same run:

```bash
HPC_MODEL=crivo-17m RESUME=auto bash hpc/submit.sh train crivo-17m-001
```

If `final_model/` already exists and only evaluation failed:

```bash
HPC_MODEL=crivo-17m bash hpc/submit.sh evaluate crivo-17m-001
```

Export `final_model/` and `run.json` as in section 10, substituting
`crivo-17m-001` for the run and archive name. The exported model uses the standard
ModernBERT sequence-classification format, so the existing Colab evaluation
notebook supports it after changing `ARCHIVE`, `WORK_DIR`, and `RESULT_DIR`.
The upstream model is licensed CC BY-NC 4.0; retain its attribution and license
when sharing adapted weights.

Local offline integration checks (with `requirements.txt` installed):

```bash
python tests/smoke_pipeline.py --architecture crivo
# Optional: exercise a previously downloaded pinned checkpoint:
python tests/smoke_pipeline.py --architecture crivo --backbone /absolute/path/to/crivo-17m
```

The checks verify exact encoder transfer, rejection of missing encoder tensors,
training, deterministic checkpoint resume, evaluation, prediction, and export
loading through the Colab notebook on a small fixture dataset.

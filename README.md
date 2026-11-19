# NLBSE 2027: Ettin-32M on the HPC

Fine-tune **[`jhu-clsp/ettin-encoder-32m`](https://huggingface.co/jhu-clsp/ettin-encoder-32m)**
to predict `[Vulnerability, MAT/SATD]` from the competition dataset's `Function`
field. This is our smaller-model experiment following the CodeBERT baseline.

The default is now [Ettin-32M](configs/ettin-32m.json). The original
[CodeBERT configuration](configs/codebert-baseline.json) remains available for comparison.
Ettin uses its own tokenizer and a separate prepared-data directory. We keep the
512-token limit and the initial training hyperparameters for the first comparison;
accuracy, speed, and energy savings still need measurement.

On the HPC, keep your existing `hpc/config.sh`, including `HPC_PARTITION=gpu-pre`.
For a first installation, copy `hpc/config.example.sh` to `hpc/config.sh`.
Python 3.11/3.12 is detected automatically; setup also handles missing `ensurepip`
without `sudo`. Defaults request one GPU, four CPUs, 32 GB RAM, and 24 hours.

From the repository root, after these changes are committed, pushed, and pulled
on the HPC:

```bash
# Reuses an existing environment; dependencies are unchanged for Ettin.
bash hpc/setup.sh

# Required even if CodeBERT data was prepared previously: Ettin has a different tokenizer.
bash hpc/prepare.sh

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
- Logs: `~/nlbse27/logs/`.

After training, export `final_model/` and `run.json` and open
[the Colab evaluation notebook](colab/Ettin32M_evaluation.ipynb). It loads the
fine-tuned classifier, runs an example, and evaluates the frozen public test
split. The [HPC guide](hpc-unibz.md) contains the export and transfer commands.
The notebook is prepared for later use; training must produce the artifact first.

ModernBERT uses native PyTorch SDPA attention with reference compilation disabled,
so no separate Flash Attention installation or compiler is required. Exported
models must be loaded with automatic model/tokenizer classes; the supplied
CodeBERT notebook's hard-coded RoBERTa classes do not support Ettin.

The supplied competition notebooks contain evaluation, not the original training
procedure. Our settings are explicit starting choices and do not claim to reproduce
the published CodeBERT weights or F1. For tuning, draw validation data from `train`
and reserve the public `test` results for reporting.

Run checks:

```bash
python3 -m unittest discover -s tests -v
# With requirements.txt installed:
python tests/smoke_pipeline.py                         # tiny ModernBERT, offline
python tests/smoke_pipeline.py --architecture roberta  # original backbone compatibility
```

The integration checks cover preparation, training, deterministic resume,
evaluation, inference, and the portable cells of the Colab notebook.
Original competition resources remain in `code-comment-classification-main/`
and `old_colab/`. Detailed operation, profiling limits, and research references
are in [hpc-unibz.md](hpc-unibz.md).

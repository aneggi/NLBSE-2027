#!/usr/bin/env python3
"""Prepare, fine-tune, evaluate, and run a two-label CodeBERT model."""

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

LABELS = ["Vulnerability", "MAT"]
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path):
    config = read_json(path)
    for key in ("batch_size", "gradient_accumulation_steps", "eval_batch_size",
                "save_steps", "save_total_limit", "max_length"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_length"] > 512:
        raise ValueError("This CodeBERT pipeline supports at most 512 tokens")
    if config["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16")
    for key in ("epochs", "learning_rate"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 <= config["warmup_ratio"] <= 1 or config["weight_decay"] < 0:
        raise ValueError("Invalid warmup_ratio or weight_decay")
    return config


def validate_labels(labels):
    if len(labels) != 2 or any(value not in (0, 1) for value in labels):
        raise ValueError(f"Expected binary [weakness, MAT] labels, got {labels!r}")
    return [float(value) for value in labels]


def classification_metrics(labels, predictions):
    if not labels or len(labels) != len(predictions):
        raise ValueError("Need equally sized, nonempty labels and predictions")
    for row in labels + predictions:
        validate_labels(row)
    result = {}
    for column, name in enumerate(LABELS):
        tp = sum(y[column] == 1 and p[column] == 1 for y, p in zip(labels, predictions))
        fp = sum(y[column] == 0 and p[column] == 1 for y, p in zip(labels, predictions))
        tn = sum(y[column] == 0 and p[column] == 0 for y, p in zip(labels, predictions))
        fn = sum(y[column] == 1 and p[column] == 0 for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[name] = dict(accuracy=(tp + tn) / len(labels), precision=precision,
                            recall=recall, f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
                            tp=tp, fp=fp, tn=tn, fn=fn)
    result["mean_f1"] = sum(result[name]["f1"] for name in LABELS) / 2
    return result


def provenance():
    def git(*args):
        process = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return process.stdout.strip() if process.returncode == 0 else None
    return {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "datasets", "accelerate", "numpy")},
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
    }


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    config = load_config(args.config)
    destination = args.prepared.resolve()
    if destination.exists():
        raise ValueError(f"Prepared directory already exists: {destination}; choose a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
    try:
        api = HfApi()
        model_sha = api.model_info(config["model_name"], revision=config["model_revision"]).sha
        data_sha = api.dataset_info(config["dataset_name"], revision=config["dataset_revision"]).sha
        snapshot_download(config["model_name"], revision=model_sha, local_dir=staging / "base-model",
                          allow_patterns=["*.json", "vocab.*", "merges.txt", "tokenizer.*",
                                          "*.safetensors", "pytorch_model*.bin"])
        tokenizer = AutoTokenizer.from_pretrained(staging / "base-model", use_fast=False,
                                                  local_files_only=True)
        ds = load_dataset(config["dataset_name"], revision=data_sha)
        if "train" not in ds or "test" not in ds:
            raise ValueError("The competition dataset must contain train and test splits")
        def encode(batch):
            if any(not isinstance(code, str) for code in batch["Function"]):
                raise ValueError("Function must contain strings")
            encoded = tokenizer(batch["Function"], truncation=True, padding="max_length",
                                max_length=config["max_length"])
            encoded["labels"] = [validate_labels(row) for row in batch["Label"]]
            return encoded
        # Cache Arrow arrays, not one giant in-memory padded tensor.
        for split in ("train", "test"):
            encoded = ds[split].map(encode, batched=True, remove_columns=ds[split].column_names)
            encoded.save_to_disk(staging / split)
        write_json(staging / "manifest.json", {
            "model_name": config["model_name"], "model_revision": model_sha,
            "dataset_name": config["dataset_name"], "dataset_revision": data_sha,
            "max_length": config["max_length"], "labels": LABELS,
            "split_sizes": {split: len(ds[split]) for split in ("train", "test")},
            "environment": provenance(),
        })
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f"Prepared offline inputs: {destination}")


def collate(features):
    from transformers import default_data_collator
    batch = default_data_collator(features)
    if "labels" in batch:
        batch["labels"] = batch["labels"].float()
    return batch


def select_samples(dataset, limit):
    return dataset.select(range(min(limit, len(dataset)))) if limit is not None else dataset


def check_prepared(config, prepared):
    manifest = read_json(prepared / "manifest.json")
    for key in ("model_name", "dataset_name", "max_length"):
        if manifest[key] != config[key]:
            raise ValueError(f"Prepared {key} differs from config; prepare a new directory")
    for key in ("model_revision", "dataset_revision"):
        if config[key] != "main" and config[key] != manifest[key]:
            raise ValueError(f"Prepared {key} differs from requested revision")
    return manifest


def train(args):
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    config = load_config(args.config)
    manifest = check_prepared(config, args.prepared)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the GPU allocation and PyTorch wheel")
    if config["precision"] != "fp32" and not torch.cuda.is_available():
        raise ValueError("Use fp32 for CPU runs")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16")
    previous = None
    if args.resume:
        previous = get_last_checkpoint(str(args.output)) if args.resume == "auto" else args.resume
        if not previous or not Path(previous).is_dir():
            raise ValueError("No resume checkpoint found")
        old = read_json(args.output / "run.json")
        if old["config"] != config or old["prepared"] != manifest or old["max_steps"] != args.max_steps or old["max_train_samples"] != args.max_train_samples:
            raise ValueError("Resume requires the same configuration, inputs, and sample/step limits")
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output is not empty; choose a new run directory or use --resume auto")
    if (args.output / "final_model").exists():
        raise ValueError("This run already has a final model; evaluate it or choose a new run")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.prepared / "base-model", use_fast=False, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.prepared / "base-model", local_files_only=True, num_labels=2,
        problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id={name: i for i, name in enumerate(LABELS)},
    )
    dataset = select_samples(load_from_disk(args.prepared / "train"), args.max_train_samples)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_json(args.output / "run.json", {
            "config": config, "prepared": manifest, "environment": provenance(),
            "max_steps": args.max_steps, "max_train_samples": args.max_train_samples,
            "train_instances": len(dataset),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        })
    training_args = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=config["epochs"],
        max_steps=args.max_steps, learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        weight_decay=config["weight_decay"], warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="linear", optim="adamw_torch", max_grad_norm=1.0,
        fp16=config["precision"] == "fp16", bf16=config["precision"] == "bf16",
        seed=config["seed"], data_seed=config["seed"], eval_strategy="no",
        save_strategy="steps", save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"], logging_steps=10, report_to="none",
        dataloader_num_workers=0, save_safetensors=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate)
    result = trainer.train(resume_from_checkpoint=previous)
    final_staging = args.output / ".final_model.tmp"
    trainer.save_model(str(final_staging))
    tokenizer.save_pretrained(final_staging)
    write_json(final_staging / "preprocessing.json", {
        "max_length": config["max_length"], "labels": LABELS, "threshold": 0.5,
    })
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    final_staging.rename(args.output / "final_model")
    print(f"Saved model: {args.output / 'final_model'}")


def evaluate(args):
    import torch
    from datasets import load_from_disk
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification

    config = load_config(args.config)
    manifest = check_prepared(config, args.prepared)
    run = read_json(args.model.parent / "run.json")
    if run["prepared"] != manifest:
        raise ValueError("Evaluation inputs differ from the training inputs")
    if args.output.exists():
        raise ValueError("Evaluation output already exists; choose a new path")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is unavailable")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True).to(device).eval()
    dataset = select_samples(load_from_disk(args.prepared / "test"), args.max_test_samples)
    loader = DataLoader(dataset, batch_size=config["eval_batch_size"], collate_fn=collate)
    labels, predictions, probabilities = [], [], []
    forward_seconds, total_flops = 0.0, 0
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            labels.extend(batch.pop("labels").tolist())
            inputs = {key: value.to(device) for key, value in batch.items()}
            synchronize()
            begin = time.perf_counter()
            profile = torch.profiler.profile(with_flops=True) if args.profile_flops else contextlib.nullcontext()
            with profile as profiler:
                for _ in range(args.repetitions):
                    logits = model(**inputs).logits
            synchronize()
            forward_seconds += time.perf_counter() - begin
            if args.profile_flops:
                total_flops += sum(event.flops or 0 for event in profiler.key_averages())
            probs = torch.sigmoid(logits)
            probabilities.extend(probs.cpu().tolist())
            predictions.extend((probs > 0.5).int().cpu().tolist())
    metrics = classification_metrics(labels, predictions)
    metrics.update({
        "instances": len(dataset), "full_public_test": len(dataset) == manifest["split_sizes"]["test"],
        "label_order": LABELS, "threshold": 0.5, "precision": "fp32",
        "repetitions": args.repetitions, "forward_seconds_per_pass": forward_seconds / args.repetitions,
        "wall_seconds": time.perf_counter() - started,
        "gflops_per_pass": total_flops / args.repetitions / 1e9 if args.profile_flops else None,
        "profiling_enabled": args.profile_flops, "eval_batch_size": config["eval_batch_size"],
        "model": str(args.model.resolve()), "prepared": manifest, "environment": provenance(),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    })
    args.output.mkdir(parents=True)
    write_json(args.output / "metrics.json", metrics)
    with (args.output / "predictions.jsonl").open("w") as handle:
        for index, (label, prediction, probs) in enumerate(zip(labels, predictions, probabilities)):
            handle.write(json.dumps(dict(index=index, labels=label, predictions=prediction, probabilities=probs)) + "\n")
    print(json.dumps(metrics, indent=2))


def predict(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.output.exists():
        raise ValueError("Prediction output already exists")
    preprocessing = read_json(args.model / "preprocessing.json")
    max_length = args.max_length or preprocessing["max_length"]
    if max_length != preprocessing["max_length"]:
        raise ValueError("max-length differs from the model's saved preprocessing")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=True)
    if model.config.num_labels != 2 or [model.config.id2label[i] for i in range(2)] != LABELS:
        raise ValueError("Expected a model saved by this pipeline with [Vulnerability, MAT] outputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Read JSONL in batches so hidden-test inference need not fit in host memory.
    with args.input.open() as source, args.output.open("x") as target, torch.inference_mode():
        pending = []
        def emit(rows):
            codes = [row["Function"] for row in rows]
            if any(not isinstance(code, str) for code in codes):
                raise ValueError("Function must contain strings")
            inputs = tokenizer(codes, padding="max_length", truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            probs = torch.sigmoid(model(**inputs).logits)
            for row, probability, prediction in zip(rows, probs.cpu().tolist(), (probs > 0.5).int().cpu().tolist()):
                output = {"predictions": prediction, "probabilities": probability}
                if "id" in row:
                    output["id"] = row["id"]
                target.write(json.dumps(output) + "\n")
        for line in source:
            if not line.strip():
                continue
            pending.append(json.loads(line))
            if len(pending) == args.batch_size:
                emit(pending)
                pending = []
        if pending:
            emit(pending)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=ROOT / "configs/codebert-baseline.json")
        command.add_argument("--prepared", type=Path, required=True)
        if name != "prepare":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--require-cuda", action="store_true")
        if name == "train":
            command.add_argument("--resume", nargs="?", const="auto")
            command.add_argument("--max-steps", type=positive_int, default=-1)
            command.add_argument("--max-train-samples", type=positive_int)
        if name == "evaluate":
            command.add_argument("--model", type=Path, required=True)
            command.add_argument("--max-test-samples", type=positive_int)
            command.add_argument("--repetitions", type=positive_int, default=1)
            command.add_argument("--profile-flops", action="store_true")
    command = commands.add_parser("predict")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--batch-size", type=positive_int, default=16)
    command.add_argument("--max-length", type=positive_int)
    args = parser.parse_args()
    if args.command == "predict" and args.max_length is not None and args.max_length > 512:
        parser.error("max-length cannot exceed 512")
    try:
        globals()[args.command](args)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

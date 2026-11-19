"""Offline pipeline check with tiny ModernBERT/RoBERTa or a downloaded backbone."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from datasets import Dataset, DatasetDict
    from tokenizers import ByteLevelBPETokenizer
    from transformers import (ModernBertConfig, ModernBertForMaskedLM,
                              RobertaConfig, RobertaModel, RobertaTokenizer, RobertaTokenizerFast)
    from safetensors.torch import load_file

    spec = importlib.util.spec_from_file_location("nlbse", ROOT / "scripts/nlbse.py")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["modernbert", "roberta"], default="modernbert")
    parser.add_argument("--backbone", type=Path, help="Optional locally downloaded Ettin checkpoint")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nlbse-integration-") as temporary:
        directory = Path(temporary)
        prepared = directory / "prepared"
        backbone = directory / "fixture-backbone"
        if args.backbone:
            shutil.copytree(args.backbone, backbone, ignore=shutil.ignore_patterns(".cache"))
        else:
            backbone.mkdir(parents=True)
            bpe = ByteLevelBPETokenizer()
            bpe.train_from_iterator(["int f() { /* TODO */ return 0; }", "void g() { return; }"],
                                    vocab_size=280, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
            bpe.save_model(str(backbone))
            tokenizer_class = RobertaTokenizerFast if args.architecture == "modernbert" else RobertaTokenizer
            tokenizer = tokenizer_class(vocab_file=str(backbone / "vocab.json"),
                                        merges_file=str(backbone / "merges.txt"))
            tokenizer.save_pretrained(backbone)
            torch.manual_seed(42)
            if args.architecture == "modernbert":
                model = ModernBertForMaskedLM(ModernBertConfig(
                    vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=32, max_position_embeddings=514,
                    reference_compile=False, pad_token_id=1, cls_token_id=0, bos_token_id=0,
                    eos_token_id=2, sep_token_id=2, classifier_pooling="mean"))
            else:
                model = RobertaModel(RobertaConfig(
                    vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=32, max_position_embeddings=514))
            model.save_pretrained(backbone)
        codes = ["int f() { /* TODO */ return 0; }", "void g() { return; }"] * 4
        config_path_source = pipeline.DEFAULT_CONFIG if args.architecture == "modernbert" else ROOT / "configs/codebert-baseline.json"
        config = pipeline.load_config(config_path_source)
        config.update(max_length=16, batch_size=2, eval_batch_size=2, save_steps=1,
                      model_revision="fixture", dataset_revision="fixture")
        config_path = directory / "config.json"
        pipeline.write_json(config_path, config)
        dataset = Dataset.from_dict({"Function": codes, "Label": [[i % 2, (i // 2) % 2] for i in range(8)],
                                     "Projectname": ["fixture"] * 8, "Filepath": ["test.c"] * 8})
        def snapshot(*args, local_dir, **kwargs):
            shutil.copytree(backbone, local_dir)
        with patch("huggingface_hub.HfApi") as api, \
                patch("huggingface_hub.snapshot_download", side_effect=snapshot), \
                patch("datasets.load_dataset", return_value=DatasetDict(train=dataset, test=dataset)):
            api.return_value.model_info.return_value = SimpleNamespace(sha="fixture")
            api.return_value.dataset_info.return_value = SimpleNamespace(sha="fixture")
            pipeline.prepare(SimpleNamespace(config=config_path, prepared=prepared))
        assert pipeline.read_json(prepared / "manifest.json")["split_sizes"] == {"train": 8, "test": 8}
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="")

        def run(command, *arguments, succeeds=True):
            result = subprocess.run([sys.executable, str(ROOT / "scripts/nlbse.py"), command, *map(str, arguments)],
                                    env=env, capture_output=True, text=True)
            if (result.returncode == 0) != succeeds:
                raise AssertionError(result.stdout + result.stderr)
            return result

        common = ["--config", config_path, "--prepared", prepared]
        output = directory / "run"
        missing = run("train", *common, "--output", output, "--resume", "auto", succeeds=False)
        assert "No resume checkpoint found" in missing.stderr and "FileNotFoundError" not in missing.stderr
        assert not output.exists()
        run("train", *common, "--output", output, "--max-steps", 2)
        # A second launch must not silently overwrite a trained model.
        run("train", *common, "--output", output, "--max-steps", 2, succeeds=False)
        resumed = directory / "resumed"
        resumed.mkdir()
        shutil.copy(output / "run.json", resumed / "run.json")
        shutil.copytree(output / "checkpoint-1", resumed / "checkpoint-1")
        run("train", *common, "--output", resumed, "--max-steps", 2, "--resume", "auto")
        first = load_file(str(output / "final_model/model.safetensors"))
        second = load_file(str(resumed / "final_model/model.safetensors"))
        assert first.keys() == second.keys()
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=1e-7, rtol=1e-6)
        evaluation = directory / "evaluation"
        run("evaluate", *common, "--model", output / "final_model", "--output", evaluation,
            "--repetitions", 2, "--profile-flops")
        metrics = pipeline.read_json(evaluation / "metrics.json")
        assert metrics["instances"] == 8 and metrics["full_public_test"]
        assert metrics["gflops_per_pass"] > 0 and 0 <= metrics["mean_f1"] <= 1
        if args.architecture == "modernbert":
            saved_config = pipeline.read_json(output / "final_model/config.json")
            assert saved_config["model_type"] == "modernbert"
            # Transformers deliberately omits reference_compile when serializing
            # ModernBERT config, so our loader must restore it on every load.
            reloaded = pipeline.load_classifier(output / "final_model")
            assert reloaded.config.reference_compile is False
            del reloaded
            assert pipeline.read_json(output / "final_model/preprocessing.json")["use_fast_tokenizer"]
        assert len((evaluation / "predictions.jsonl").read_text().splitlines()) == 8
        inputs = directory / "input.jsonl"
        inputs.write_text(json.dumps({"id": "example", "Function": codes[0]}) + "\n")
        predictions = directory / "predicted.jsonl"
        run("predict", "--model", output / "final_model", "--input", inputs,
            "--output", predictions)
        predicted = json.loads(predictions.read_text())
        evaluated = json.loads((evaluation / "predictions.jsonl").read_text().splitlines()[0])
        assert predicted["id"] == "example" and len(predicted["predictions"]) == 2
        assert predicted["predictions"] == evaluated["predictions"]
        # Execute the portable Colab cells against the exact HPC export format.
        archive_path = directory / "model-export.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(output / "final_model", arcname="final_model")
            archive.add(output / "run.json", arcname="run.json")
        notebook = pipeline.read_json(ROOT / "colab/Ettin32M_evaluation.ipynb")
        scope = dict(Path=Path, ARCHIVE=archive_path, WORK_DIR=directory / "colab-export",
                     RESULT_DIR=directory / "colab-results", BATCH_SIZE=2, MAX_TEST_SAMPLES=None,
                     REPETITIONS=1, PROFILE_FLOPS=False)
        with patch("datasets.load_dataset", return_value=dataset), patch("torch.cuda.is_available", return_value=False):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code" and not cell["metadata"].get("tags"):
                    exec(compile("".join(cell["source"]), f"colab-cell-{index}", "exec"), scope)
        assert scope["metrics"]["mean_f1"] == metrics["mean_f1"]
        assert scope["predictions"] == [json.loads(line)["predictions"] for line in
                                         (evaluation / "predictions.jsonl").read_text().splitlines()]
    print(f"PASS ({args.architecture}): preparation, offline train, deterministic checkpoint resume, evaluation/FLOPs, inference, Colab export evaluation, overwrite guard")


if __name__ == "__main__":
    main()

"""Offline pipeline check with tiny ModernBERT/Crivo/RoBERTa or a downloaded backbone."""

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
    from transformers import (ModernBertConfig, ModernBertForMaskedLM, ModernBertModel,
                              RobertaConfig, RobertaModel, RobertaTokenizer, RobertaTokenizerFast)
    from safetensors.torch import load_file, save_file

    spec = importlib.util.spec_from_file_location("nlbse", ROOT / "scripts/nlbse.py")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["modernbert", "crivo", "roberta"], default="modernbert")
    parser.add_argument("--backbone", type=Path, help="Optional locally downloaded checkpoint")
    parser.add_argument("--config", type=Path, help="Optional configuration to exercise")
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
            tokenizer_class = RobertaTokenizer if args.architecture == "roberta" else RobertaTokenizerFast
            tokenizer = tokenizer_class(vocab_file=str(backbone / "vocab.json"),
                                        merges_file=str(backbone / "merges.txt"))
            if args.architecture == "crivo":
                tokenizer.add_special_tokens({"additional_special_tokens": ["[OPT]"]})
            tokenizer.save_pretrained(backbone)
            torch.manual_seed(42)
            if args.architecture in ("modernbert", "crivo"):
                model_class = ModernBertModel if args.architecture == "crivo" else ModernBertForMaskedLM
                model = model_class(ModernBertConfig(
                    vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=32, max_position_embeddings=514,
                    reference_compile=False, pad_token_id=1, cls_token_id=0, bos_token_id=0,
                    eos_token_id=2, sep_token_id=2, classifier_pooling="mean"))
            else:
                model = RobertaModel(RobertaConfig(
                    vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=32, max_position_embeddings=514))
            if args.architecture == "crivo":
                # Match Crivo's custom state dictionary and Transformers 5.x
                # config, including a non-default local rotary base.
                state = {f"encoder.{key}": value for key, value in model.state_dict().items()}
                state["type_embedding.weight"] = torch.zeros(3, model.config.hidden_size)
                state["scorer.0.weight"] = torch.ones(model.config.hidden_size)
                save_file(state, str(backbone / "model.safetensors"))
                raw_config = model.config.to_dict()
                raw_config.pop("global_rope_theta", None)
                raw_config.pop("local_rope_theta", None)
                raw_config["rope_parameters"] = {
                    name: {"rope_theta": 160000.0, "rope_type": "default"}
                    for name in ("full_attention", "sliding_attention")}
                raw_config["decision"] = {"opt_token": "[OPT]"}
                pipeline.write_json(backbone / "config.json", raw_config)
                tokenizer_config = pipeline.read_json(backbone / "tokenizer_config.json")
                tokenizer_config["tokenizer_class"] = "TokenizersBackend"
                tokenizer_config.pop("additional_special_tokens", None)
                tokenizer_config["extra_special_tokens"] = ["[OPT]"]
                tokenizer_config["model_input_names"] = ["input_ids", "attention_mask"]
                pipeline.write_json(backbone / "tokenizer_config.json", tokenizer_config)
            else:
                model.save_pretrained(backbone)
        if args.architecture == "crivo":
            transferred = pipeline.load_classifier(backbone, for_training=True)
            original = load_file(str(backbone / "model.safetensors"))
            for key, value in transferred.base_model.state_dict().items():
                torch.testing.assert_close(value, original[f"encoder.{key}"], rtol=0, atol=0)
            assert transferred.config.global_rope_theta == 160000.0
            assert transferred.config.local_rope_theta == 160000.0
            assert transferred.config.num_labels == 2
            assert not hasattr(transferred.config, "decision")
            tokenizer = pipeline.load_tokenizer(backbone, use_fast=True)
            assert len(tokenizer) == transferred.config.vocab_size
            assert tokenizer.convert_tokens_to_ids("[OPT]") != tokenizer.unk_token_id
            from tokenizers import Tokenizer
            graph = Tokenizer.from_file(str(backbone / "tokenizer.json"))
            assert tokenizer("int f() { /* TODO */ return 0; }")["input_ids"] == \
                graph.encode("int f() { /* TODO */ return 0; }").ids
            # A missing encoder tensor must fail rather than leave random weights.
            with patch("safetensors.torch.load_file", return_value={}):
                try:
                    pipeline.load_classifier(backbone, for_training=True)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Incomplete Crivo encoder was accepted")
            del transferred, original
        codes = ["int f() { /* TODO */ return 0; }", "void g() { return; }"] * 4
        config_path_source = {"modernbert": pipeline.DEFAULT_CONFIG,
                              "crivo": ROOT / "configs/crivo-17m.json",
                              "roberta": ROOT / "configs/codebert-baseline.json"}[args.architecture]
        config = pipeline.load_config(args.config or config_path_source)
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
        if args.architecture in ("modernbert", "crivo"):
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

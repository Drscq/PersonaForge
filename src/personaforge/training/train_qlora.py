from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from personaforge.training.dataset_format import load_dpo_rows, load_sft_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    config.update({key: value for key, value in vars(args).items() if value is not None})
    stage = config["stage"]

    if stage in {"sft", "both"}:
        run_sft(config)
    if stage in {"dpo", "both"}:
        run_dpo(config)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT+DPO training for PersonaForge.")
    parser.add_argument("--config", default="configs/colab_qwen_0_5b.json")
    parser.add_argument("--stage", choices=["sft", "dpo", "both"], default="both")
    parser.add_argument("--model-name", dest="model_name")
    parser.add_argument("--sft-data", dest="sft_data")
    parser.add_argument("--dpo-data", dest="dpo_data")
    parser.add_argument("--sft-output-dir", dest="sft_output_dir")
    parser.add_argument("--dpo-output-dir", dest="dpo_output_dir")
    parser.add_argument("--max-steps", dest="max_steps", type=int)
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false", default=None)
    return parser.parse_args(argv)


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("stage", "both")
    config.setdefault("load_in_4bit", True)
    return config


def run_sft(config: dict[str, Any]) -> None:
    torch, Dataset, SFTConfig, SFTTrainer, tokenizer, model, lora_config = load_training_stack(config)
    dataset = Dataset.from_list(load_sft_rows(config["sft_data"]))
    args = build_trl_config(
        SFTConfig,
        {
            "output_dir": config["sft_output_dir"],
            "per_device_train_batch_size": config["per_device_train_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "learning_rate": config["learning_rate"],
            "num_train_epochs": config["num_train_epochs"],
            "max_steps": config["max_steps"],
            "logging_steps": config["logging_steps"],
            "save_steps": config["save_steps"],
            "max_length": config["max_length"],
            "gradient_checkpointing": True,
            "bf16": supports_bf16(torch),
            "fp16": not supports_bf16(torch),
            "report_to": "none",
            "completion_only_loss": True,
        },
    )
    trainer = instantiate_trainer(
        SFTTrainer,
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(config["sft_output_dir"])
    tokenizer.save_pretrained(config["sft_output_dir"])


def run_dpo(config: dict[str, Any]) -> None:
    torch, Dataset, DPOConfig, DPOTrainer, tokenizer, model, lora_config = load_training_stack(
        config, prefer_sft_adapter=True
    )
    dataset = Dataset.from_list(load_dpo_rows(config["dpo_data"]))
    args = build_trl_config(
        DPOConfig,
        {
            "output_dir": config["dpo_output_dir"],
            "per_device_train_batch_size": config["per_device_train_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "learning_rate": config["learning_rate"] / 2,
            "num_train_epochs": config["num_train_epochs"],
            "max_steps": config["max_steps"],
            "logging_steps": config["logging_steps"],
            "save_steps": config["save_steps"],
            "max_length": config["max_length"],
            "gradient_checkpointing": True,
            "bf16": supports_bf16(torch),
            "fp16": not supports_bf16(torch),
            "report_to": "none",
            "beta": 0.1,
        },
    )
    trainer = instantiate_trainer(
        DPOTrainer,
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(config["dpo_output_dir"])
    tokenizer.save_pretrained(config["dpo_output_dir"])


def load_training_stack(config: dict[str, Any], prefer_sft_adapter: bool = False):
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from exc

    model_name = config["model_name"]
    token = os.environ.get("HF_TOKEN")
    quantization_config = None
    if config.get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if supports_bf16(torch) else torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.use_cache = False
    if config.get("load_in_4bit", True):
        model = prepare_model_for_kbit_training(model)

    sft_dir = Path(config["sft_output_dir"])
    if prefer_sft_adapter and sft_dir.exists():
        model = PeftModel.from_pretrained(model, str(sft_dir), is_trainable=True)
        lora_config = None
    else:
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    trainer_cls = DPOTrainer if prefer_sft_adapter else SFTTrainer
    config_cls = DPOConfig if prefer_sft_adapter else SFTConfig
    return torch, Dataset, config_cls, trainer_cls, tokenizer, model, lora_config


def instantiate_trainer(trainer_cls, **kwargs):
    """Handle small TRL API differences across versions."""

    signature = inspect.signature(trainer_cls.__init__)
    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    if "processing_class" in signature.parameters and "tokenizer" in kwargs:
        filtered["processing_class"] = kwargs["tokenizer"]
    if "tokenizer" in signature.parameters and "tokenizer" in kwargs:
        filtered["tokenizer"] = kwargs["tokenizer"]
    return trainer_cls(**filtered)


def build_trl_config(config_cls, kwargs: dict[str, Any]):
    if is_dataclass(config_cls):
        allowed = {field.name for field in fields(config_cls)}
    else:
        allowed = set(inspect.signature(config_cls).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    return config_cls(**filtered)


def supports_bf16(torch_module) -> bool:
    return bool(torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported())


if __name__ == "__main__":
    raise SystemExit(main())

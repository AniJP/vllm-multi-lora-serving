"""Fine-tune a single LoRA adapter for one task on top of a shared base model.

Run once per task:
    python src/train_lora.py --task sql
    python src/train_lora.py --task summarize
    python src/train_lora.py --task extract
"""

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# TRL has renamed SFTConfig fields across releases (e.g. max_seq_length ->
# max_length). Resolve against whichever name the installed version accepts
# instead of hardcoding one, so a version bump doesn't break this again.
SFT_CONFIG_PARAMS = set(inspect.signature(SFTConfig.__init__).parameters)


def _resolve_kwarg(candidates, value):
    for name in candidates:
        if name in SFT_CONFIG_PARAMS:
            return {name: value}
    raise TypeError(f"None of {candidates} are accepted by the installed trl.SFTConfig")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def to_chat_text(tokenizer, row):
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
        {"role": "assistant", "content": row["assistant"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sql", "summarize", "extract"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="adapters")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    rows = load_jsonl(Path(args.data_dir) / f"{args.task}_train.jsonl")
    dataset = Dataset.from_list([{"text": to_chat_text(tokenizer, r)} for r in rows])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    output_dir = Path(args.output_dir) / f"{args.task}-lora"
    sft_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )
    sft_kwargs.update(_resolve_kwarg(["max_length", "max_seq_length"], args.max_seq_len))
    sft_kwargs.update(_resolve_kwarg(["dataset_text_field"], "text"))
    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"saved adapter to {output_dir}")


if __name__ == "__main__":
    main()

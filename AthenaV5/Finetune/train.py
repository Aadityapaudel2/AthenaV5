#!/usr/bin/env python3
"""
train.py

Minimal supervised finetuning for a causal LM on conversation JSONL produced by prepare_data.py.

Dependencies:
  pip install transformers datasets accelerate

Usage:
  accelerate launch train.py --model_name_or_path <BASE> --train_file train_athena.jsonl --output_dir outputs/athena_v5

This script:
  - Adds simple special tokens for roles (<|user|>, <|assistant|>)
  - Builds a single training sequence per conversation
  - Masks loss on user tokens; computes loss only on assistant tokens
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments)

ROLE_USER = "<|user|>"
ROLE_ASSISTANT = "<|assistant|>"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_MODEL = str(PROJECT_ROOT / "checkpoints" / "Qwen3-0.6B-Base")
DEFAULT_TRAIN_FILE = str(BASE_DIR / "trainingdata" / "samples" / "athena_commandments_train.jsonl")
DEFAULT_OUTPUT = str(BASE_DIR / "trainingdata" / "output" / "qwen_commandment")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    p.add_argument("--train_file", default=DEFAULT_TRAIN_FILE, help="Conversation JSONL from prepare_data.py")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT)

    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def format_conversation(messages: List[Dict[str, str]]) -> str:
    # Simple linearization.
    chunks = []
    for m in messages:
        role = m["role"]
        content = (m.get("content") or "").strip()
        if role == "user":
            chunks.append(f"{ROLE_USER}\n{content}\n")
        elif role == "assistant":
            chunks.append(f"{ROLE_ASSISTANT}\n{content}\n")
    return "".join(chunks).strip() + "\n"

@dataclass
class DataCollatorForCausalChat:
    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Build labels: mask out everything that is not inside assistant segments.
        labels = input_ids.clone()
        labels[:] = -100

        # We detect assistant spans by scanning for ROLE_ASSISTANT token sequence.
        # This works because ROLE_ASSISTANT is a single token after adding it as special token.
        assist_id = self.tokenizer.convert_tokens_to_ids(ROLE_ASSISTANT)
        user_id = self.tokenizer.convert_tokens_to_ids(ROLE_USER)

        for i in range(input_ids.size(0)):
            seq = input_ids[i].tolist()

            # Find all role token positions
            role_positions = []
            for j, tid in enumerate(seq):
                if tid == user_id:
                    role_positions.append(("user", j))
                elif tid == assist_id:
                    role_positions.append(("assistant", j))

            # Sort by position
            role_positions.sort(key=lambda x: x[1])

            # For each assistant segment, label tokens after the assistant marker up to next role marker (or end)
            for idx, (role, pos) in enumerate(role_positions):
                if role != "assistant":
                    continue
                start = pos + 1
                end = role_positions[idx + 1][1] if idx + 1 < len(role_positions) else len(seq)
                for k in range(start, end):
                    if attention_mask[i, k].item() == 1:
                        labels[i, k] = input_ids[i, k]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ds = load_dataset("json", data_files={"train": args.train_file})["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Add role tokens
    special_tokens = {"additional_special_tokens": [ROLE_USER, ROLE_ASSISTANT]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def map_fn(example):
        text = format_conversation(example["messages"])
        return {"text": text}

    ds = ds.map(map_fn, remove_columns=ds.column_names)

    collator = DataCollatorForCausalChat(tokenizer=tokenizer, max_length=args.max_seq_length)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

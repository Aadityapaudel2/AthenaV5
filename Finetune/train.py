#!/usr/bin/env python3
"""
Lean SFT trainer for conversation JSONL produced by prepare_data.py.

Expected input JSONL row:
  {"messages": [{"role": "user|assistant", "content": "..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROLE_USER = "<|user|>"
ROLE_ASSISTANT = "<|assistant|>"
ROLE_SYSTEM = "<|system|>"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_MODEL = str(PROJECT_ROOT / "models" / "Qwen3-1.7B")
DEFAULT_TRAIN_FILE = str(BASE_DIR / "trainingdata" / "bhagavadgita" / "bhagavaggitatrainingdata_train.jsonl")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "models" / "tuned")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    p.add_argument("--train_file", default=DEFAULT_TRAIN_FILE)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT)

    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_only_model", action="store_true")

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def render_chat(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    """Render messages using the tokenizer's native chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)
    # Fallback if tokenizer has no chat template.
    chunks: list[str] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            chunks.append(f"{ROLE_SYSTEM}\n{content}\n")
        elif role == "user":
            chunks.append(f"{ROLE_USER}\n{content}\n")
        elif role == "assistant":
            chunks.append(f"{ROLE_ASSISTANT}\n{content}\n")
    if add_generation_prompt:
        chunks.append(f"{ROLE_ASSISTANT}\n")
    return "".join(chunks).strip() + "\n"


class JsonlChatDataset(Dataset):
    def __init__(self, path: str):
        self.samples: list[list[dict[str, str]]] = []
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Train file not found: {p}")

        with p.open("r", encoding="utf-8-sig") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                messages = obj.get("messages")
                if not isinstance(messages, list):
                    raise ValueError(f"Line {line_no}: missing 'messages' list")
                normalized = normalize_messages(messages)
                if normalized:
                    self.samples.append(normalized)

        if not self.samples:
            raise ValueError("No usable training samples found in train_file")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[dict[str, str]]]:
        return {"messages": self.samples[idx]}


class DataCollatorForCausalChat:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            messages = feature["messages"]
            full_text = render_chat(self.tokenizer, messages, add_generation_prompt=False)
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            labels = [-100] * len(full_ids)

            # Label assistant spans using native template boundaries.
            for idx, msg in enumerate(messages):
                if msg["role"] != "assistant":
                    continue
                prefix_text = render_chat(self.tokenizer, messages[:idx], add_generation_prompt=True)
                turn_text = render_chat(self.tokenizer, messages[: idx + 1], add_generation_prompt=False)
                prefix_len = len(self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
                turn_len = len(self.tokenizer(turn_text, add_special_tokens=False)["input_ids"])
                start = min(prefix_len, len(labels))
                end = min(turn_len, len(labels))
                for pos in range(start, end):
                    labels[pos] = full_ids[pos]

            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
                labels = labels[: self.max_length]

            batch_input_ids.append(full_ids)
            batch_labels.append(labels)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has no pad_token_id; set pad token before collator use.")

        max_len = max(len(x) for x in batch_input_ids)
        input_ids_t = torch.full((len(batch_input_ids), max_len), pad_id, dtype=torch.long)
        attention_t = torch.zeros((len(batch_input_ids), max_len), dtype=torch.long)
        labels_t = torch.full((len(batch_input_ids), max_len), -100, dtype=torch.long)

        for i, (ids, labs) in enumerate(zip(batch_input_ids, batch_labels)):
            l = len(ids)
            input_ids_t[i, :l] = torch.tensor(ids, dtype=torch.long)
            attention_t[i, :l] = 1
            labels_t[i, :l] = torch.tensor(labs, dtype=torch.long)

        return {"input_ids": input_ids_t, "attention_mask": attention_t, "labels": labels_t}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_ds = JsonlChatDataset(args.train_file)
    print(f"Loaded samples: {len(train_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    added_special_tokens = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added_special_tokens = True

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    uses_native_template = hasattr(tokenizer, "apply_chat_template")
    if not uses_native_template:
        tokenizer.add_special_tokens({"additional_special_tokens": [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT]})
        added_special_tokens = True
    if added_special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
        save_total_limit=2,
        save_only_model=args.save_only_model,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROLES = {"system", "user", "assistant"}
NUM_ARGS = [
    ("max_seq_length", int, 2048), ("per_device_train_batch_size", int, 1), ("gradient_accumulation_steps", int, 8),
    ("learning_rate", float, 2e-5), ("num_train_epochs", float, 3.0), ("warmup_ratio", float, 0.03),
    ("lr_scheduler_type", str, "linear"), ("weight_decay", float, 0.0), ("max_grad_norm", float, 1.0),
    ("logging_steps", int, 10), ("save_steps", int, 200), ("save_total_limit", int, 2),
    ("expected_samples", int, 0), ("seed", int, 777),
]
FLAG_ARGS = ("save_only_model", "strict_no_truncation", "bf16", "fp16", "gradient_checkpointing")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact Qwen chat SFT trainer")
    for n in ("model_name_or_path", "train_file", "output_dir"): p.add_argument(f"--{n}", required=True)
    for n, t, d in NUM_ARGS: p.add_argument(f"--{n}", type=t, default=d)
    for n in FLAG_ARGS: p.add_argument(f"--{n}", action="store_true")
    p.add_argument("--resume_from_checkpoint", default="")
    return p.parse_args()


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for m in messages:
        role, content = str(m.get("role") or "").strip(), str(m.get("content") or "").strip()
        if role in ROLES and content: out.append({"role": role, "content": content})
    return out


def render(tok, messages: list[dict[str, str]], gen: bool = False) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)


def ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


class ChatDataset(Dataset):
    def __init__(self, path: str):
        p = Path(path)
        if not p.is_file(): raise FileNotFoundError(f"Train file not found: {p}")
        self.samples: list[list[dict[str, str]]] = []
        with p.open("r", encoding="utf-8-sig") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                msgs = obj.get("messages")
                if not isinstance(msgs, list): raise ValueError(f"Line {i}: missing 'messages' list")
                cleaned = normalize_messages(msgs)
                if cleaned: self.samples.append(cleaned)
        if not self.samples: raise ValueError("No usable training samples found")

    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, i: int) -> dict: return {"messages": self.samples[i]}


def make_collator(tok, max_len: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        packed_ids, packed_lbl = [], []
        for row in batch:
            msgs = row["messages"]
            x = ids(tok, render(tok, msgs))
            y = [-100] * len(x)
            for i, m in enumerate(msgs):
                if m["role"] != "assistant": continue
                s = min(len(ids(tok, render(tok, msgs[:i], True))), len(x))
                e = min(len(ids(tok, render(tok, msgs[: i + 1]))), len(x))
                if e > s: y[s:e] = x[s:e]
            packed_ids.append(x[:max_len]); packed_lbl.append(y[:max_len])
        pad = tok.pad_token_id
        if pad is None: raise ValueError("Tokenizer pad_token_id is required")
        mlen = max(len(v) for v in packed_ids)
        x = torch.full((len(batch), mlen), pad, dtype=torch.long)
        a = torch.zeros((len(batch), mlen), dtype=torch.long)
        y = torch.full((len(batch), mlen), -100, dtype=torch.long)
        for i, (v, l) in enumerate(zip(packed_ids, packed_lbl)):
            n = len(v)
            x[i, :n] = torch.tensor(v)
            a[i, :n] = 1
            y[i, :n] = torch.tensor(l)
        return {"input_ids": x, "attention_mask": a, "labels": y}
    return collate


def main() -> None:
    args = parse_args(); torch.manual_seed(args.seed)
    ds = ChatDataset(args.train_file); print(f"Loaded samples: {len(ds)}")
    if args.expected_samples and len(ds) != args.expected_samples: raise ValueError(f"Expected {args.expected_samples}, got {len(ds)}")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if not hasattr(tok, "apply_chat_template"): raise ValueError("Tokenizer must support apply_chat_template()")
    if tok.pad_token is None: tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|pad|>"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if args.gradient_checkpointing: model.gradient_checkpointing_enable()

    lens = sorted(len(ids(tok, render(tok, s))) for s in ds.samples)
    over = sum(1 for n in lens if n > args.max_seq_length)
    print(f"Token length stats: min={lens[0]} p95={lens[int(0.95*(len(lens)-1))]} max={lens[-1]} over_limit({args.max_seq_length})={over}")
    if args.strict_no_truncation and over: raise ValueError(f"strict_no_truncation enabled but {over} samples exceed max_seq_length={args.max_seq_length}")

    targs = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs, warmup_ratio=args.warmup_ratio, lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_total_limit=args.save_total_limit, save_only_model=args.save_only_model,
        bf16=args.bf16, fp16=args.fp16, report_to=[], remove_unused_columns=False, seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=make_collator(tok, args.max_seq_length), tokenizer=tok)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint.strip() or None)
    trainer.save_model(args.output_dir); tok.save_pretrained(args.output_dir)


if __name__ == "__main__": main()

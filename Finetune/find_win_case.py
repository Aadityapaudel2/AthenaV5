#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class QAItem:
    question: str
    answer: int


def build_items(n: int, seed: int) -> list[QAItem]:
    rng = random.Random(seed)
    out: list[QAItem] = []

    # Integer arithmetic expressions (objective, quick to verify).
    while len(out) < n:
        a = rng.randint(3, 99)
        b = rng.randint(2, 40)
        c = rng.randint(2, 30)
        d = rng.randint(2, 20)
        expr_type = rng.randint(0, 2)
        if expr_type == 0:
            q = f"Compute exactly: ({a} + {b}) * {c} - {d}"
            ans = (a + b) * c - d
        elif expr_type == 1:
            q = f"Compute exactly: {a} * ({b} - {c}) + {d}"
            ans = a * (b - c) + d
        else:
            e = rng.randint(2, 12)
            q = f"Compute exactly: ({a} * {b}) - ({c} * {d}) + {e}"
            ans = (a * b) - (c * d) + e
        out.append(QAItem(question=q, answer=ans))

    return out


def extract_last_int(text: str) -> int | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    matches = re.findall(r"[-+]?\d+", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def generate_answer(
    model,
    tokenizer,
    device: torch.device,
    question: str,
    max_new_tokens: int,
) -> str:
    user_prompt = (
        "Solve this math question. Return only the final integer answer, no words.\n"
        f"Question: {question}"
    )
    msgs = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"<|user|>\n{user_prompt}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


def run_model(model_dir: Path, items: list[QAItem], max_new_tokens: int) -> list[int | None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    preds: list[int | None] = []
    for it in items:
        txt = generate_answer(model, tokenizer, device, it.question, max_new_tokens=max_new_tokens)
        preds.append(extract_last_int(txt))

    del model
    torch.cuda.empty_cache()
    return preds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--tuned_model", required=True)
    p.add_argument("--num_questions", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=32)
    args = p.parse_args()

    items = build_items(args.num_questions, args.seed)
    print(f"Generated {len(items)} objective questions.")

    print(f"Running base: {args.base_model}")
    base_preds = run_model(Path(args.base_model), items, max_new_tokens=args.max_new_tokens)

    print(f"Running tuned: {args.tuned_model}")
    tuned_preds = run_model(Path(args.tuned_model), items, max_new_tokens=args.max_new_tokens)

    base_correct = sum(int(p == it.answer) for p, it in zip(base_preds, items))
    tuned_correct = sum(int(p == it.answer) for p, it in zip(tuned_preds, items))
    print(f"Base accuracy:  {base_correct}/{len(items)}")
    print(f"Tuned accuracy: {tuned_correct}/{len(items)}")

    wins = []
    for i, (it, bp, tp) in enumerate(zip(items, base_preds, tuned_preds), start=1):
        if tp == it.answer and bp != it.answer:
            wins.append((i, it, bp, tp))

    print(f"Tuned-only wins: {len(wins)}")
    for i, it, bp, tp in wins[:10]:
        print(f"[{i}] Q: {it.question}")
        print(f"    gold={it.answer} base={bp} tuned={tp}")


if __name__ == "__main__":
    main()

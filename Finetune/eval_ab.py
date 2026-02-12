#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = ""


@dataclass
class EvalItem:
    item_id: str
    prompt: str
    expected: str
    expected_type: str
    bucket: str


def load_eval_file(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"Line {i}: missing prompt")
            item_id = str(obj.get("id", f"row_{i:03d}"))
            expected = str(obj.get("expected", "")).strip()
            expected_type = str(obj.get("expected_type", "text")).strip().lower()
            bucket = str(obj.get("bucket", "general")).strip()
            if expected_type not in {"text", "integer", "binary"}:
                raise ValueError(f"Line {i}: invalid expected_type '{expected_type}'")
            items.append(
                EvalItem(
                    item_id=item_id,
                    prompt=prompt,
                    expected=expected,
                    expected_type=expected_type,
                    bucket=bucket,
                )
            )
    if not items:
        raise ValueError("No eval rows found")
    return items


def clean_output(text: str) -> str:
    t = text.replace("\r\n", "\n").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE)
    return t.strip()


def score_output(output: str, expected: str, expected_type: str) -> tuple[int, str]:
    out = clean_output(output)
    if expected_type == "binary":
        ok = int(out in {"0", "1"} and (not expected or out == expected))
        return ok, out
    if expected_type == "integer":
        ok = int(bool(re.fullmatch(r"[-+]?\d+", out)) and (not expected or out == expected))
        return ok, out
    ok = int((not expected) or (out == expected))
    return ok, out


def generate_one(model: Any, tokenizer: Any, device: torch.device, prompt: str, max_new_tokens: int) -> str:
    msgs: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    if SYSTEM_PROMPT:
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        rendered = f"<|user|>\n{prompt}\n<|assistant|>\n"

    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    generated_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def run_model(model_dir: Path, prompts: list[str], max_new_tokens: int) -> list[str]:
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

    outputs = [generate_one(model, tokenizer, device, p, max_new_tokens) for p in prompts]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--tuned_model", required=True)
    parser.add_argument("--eval_file", required=True, help="JSONL with fields: id,prompt,expected,expected_type")
    parser.add_argument("--out_csv", default="Finetune/eval_results.csv")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument(
        "--system_prompt",
        default=(
            "You are Athena in strict evaluation mode.\n\n"
            "If the user asks a math question, output only the final answer with no explanation, "
            "no steps, and no extra words.\n\n"
            "If the user message is not a math question, output exactly one character:\n"
            "1 if you agree, 0 if you disagree.\n\n"
            "Do not output any other text. Output must be a single line."
        ),
    )
    args = parser.parse_args()
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = args.system_prompt

    items = load_eval_file(Path(args.eval_file))
    prompts = [x.prompt for x in items]

    print(f"Loaded {len(items)} eval rows")
    print(f"Base model: {args.base_model}")
    base_outputs = run_model(Path(args.base_model), prompts, args.max_new_tokens)
    print(f"Tuned model: {args.tuned_model}")
    tuned_outputs = run_model(Path(args.tuned_model), prompts, args.max_new_tokens)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_total = 0
    tuned_total = 0
    same_output_total = 0
    both_ok_total = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "id",
                "bucket",
                "expected_type",
                "prompt",
                "expected",
                "base_output",
                "base_ok",
                "tuned_output",
                "tuned_ok",
                "same_output",
                "both_ok",
                "winner",
            ]
        )
        for item, base_out, tuned_out in zip(items, base_outputs, tuned_outputs):
            base_ok, base_clean = score_output(base_out, item.expected, item.expected_type)
            tuned_ok, tuned_clean = score_output(tuned_out, item.expected, item.expected_type)
            same_output = int(base_clean == tuned_clean)
            both_ok = int(base_ok == 1 and tuned_ok == 1)
            base_total += base_ok
            tuned_total += tuned_ok
            same_output_total += same_output
            both_ok_total += both_ok
            if tuned_ok > base_ok:
                winner = "tuned"
            elif base_ok > tuned_ok:
                winner = "base"
            else:
                winner = "tie"
            writer.writerow(
                [
                    item.item_id,
                    item.bucket,
                    item.expected_type,
                    item.prompt,
                    item.expected,
                    base_clean,
                    base_ok,
                    tuned_clean,
                    tuned_ok,
                    same_output,
                    both_ok,
                    winner,
                ]
            )

    print(f"Base score:  {base_total}/{len(items)}")
    print(f"Tuned score: {tuned_total}/{len(items)}")
    print(f"Same output: {same_output_total}/{len(items)}")
    print(f"Both correct: {both_ok_total}/{len(items)}")
    print(f"Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()

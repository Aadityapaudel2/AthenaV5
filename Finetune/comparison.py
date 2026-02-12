#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Run vanilla vs finetuned comparison and write CSV.")
    p.add_argument("--base_model", default=str(root / "models" / "Qwen3-1.7B"))
    p.add_argument("--tuned_model", default=str(root / "models" / "tuned" / "1.7mathsimple"))
    p.add_argument("--eval_file", default=str(root / "Finetune" / "eval_sanity_check.jsonl"))
    p.add_argument("--out_csv", default=str(root / "Finetune" / "eval_sanity_results.csv"))
    p.add_argument("--max_new_tokens", type=int, default=96)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve().parent / "eval_ab.py"

    cmd = [
        sys.executable,
        str(script),
        "--base_model",
        args.base_model,
        "--tuned_model",
        args.tuned_model,
        "--eval_file",
        args.eval_file,
        "--out_csv",
        args.out_csv,
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]

    print("Running comparison:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nCSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "checkpoints" / "Qwen3-0.6B-Base"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Athena V5 CLI chat (Qwen 0.6B · GPU)", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to the Qwen 0.6B checkpoint folder.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (float).")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling (top-p).")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 disables).")
    parser.add_argument("--context", default="", help="Optional prefix appended to every prompt.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    accelerator = Accelerator()
    device = accelerator.device
    print("Loading tokenizer and model (this may take a moment)...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    print(f"Athena V5 ready on {device}. Press Ctrl+C to exit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not prompt:
            print(" (empty prompt, try again)")
            continue
        full_prompt = f"{args.context}\n{prompt}".strip()
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        gen_cfg = GenerationConfig(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            top_k=args.top_k if args.top_k > 0 else None,
            do_sample=True,
        )
        with torch.no_grad(), accelerator.autocast():
            outputs = model.generate(**inputs, generation_config=gen_cfg)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text[len(full_prompt) :].strip() if text.startswith(full_prompt) else text.strip()
        print("\nAthena:", textwrap.fill(response, width=80))


if __name__ == "__main__":
    main()

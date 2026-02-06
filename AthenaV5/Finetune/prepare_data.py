#!/usr/bin/env python3
"""
prepare_data.py

Converts message-level JSONL (athena_apprentice.jsonl) into conversation JSONL suitable for chat SFT.

- Input lines contain: role=student|teacher, metadata{dialogue_id, turn, scene, mode, topic}, content.
- Output lines contain: {"meta": {...}, "messages": [{"role":"user"/"assistant","content":...}, ...]}

You choose which role the *assistant* should learn to generate:
  --assistant_role student   (Athena as assistant; Neohm as user)
  --assistant_role teacher   (Neohm as assistant; Athena as user)

This script intentionally treats sigils/passcodes as fictional mnemonics, not real security.
"""

import argparse
import json
from collections import defaultdict

DEFAULT_INPUT = "V5/finetuning_data/athena_commandments.jsonl"
DEFAULT_OUTPUT = "V5/finetuning_data/athena_commandments_train.jsonl"
DEFAULT_ROLE = "teacher"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT, help=f"Input JSONL (default {DEFAULT_INPUT})")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output JSONL (default {DEFAULT_OUTPUT})")
    p.add_argument("--assistant_role", choices=["student", "teacher"], default=DEFAULT_ROLE,
                   help=f"Which original role becomes the assistant (default {DEFAULT_ROLE})")
    p.add_argument("--drop_empty", action="store_true", help="Drop dialogues with no assistant messages after mapping")
    return p.parse_args()

def main():
    args = parse_args()

    by_dialogue = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            did = obj.get("metadata", {}).get("dialogue_id")
            by_dialogue[did].append(obj)

    assistant_role = args.assistant_role
    user_role = "teacher" if assistant_role == "student" else "student"

    with open(args.output, "w", encoding="utf-8") as out:
        for did, turns in sorted(by_dialogue.items(), key=lambda x: (x[0] is None, x[0])):
            turns_sorted = sorted(turns, key=lambda t: t.get("metadata", {}).get("turn", 0))

            # meta: take from first item
            meta = dict(turns_sorted[0].get("metadata", {}))
            meta["dialogue_id"] = did

            messages = []
            assistant_count = 0
            for t in turns_sorted:
                r = t.get("role")
                content = t.get("content", "")
                if r == assistant_role:
                    messages.append({"role": "assistant", "content": content})
                    assistant_count += 1
                elif r == user_role:
                    messages.append({"role": "user", "content": content})
                else:
                    # unknown role; skip
                    continue

            if args.drop_empty and assistant_count == 0:
                continue

            out.write(json.dumps({"meta": meta, "messages": messages}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

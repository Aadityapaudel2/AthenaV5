#!/usr/bin/env python3
"""
Build chat-SFT artifacts from turn-level JSONL.

Input row:
  {"role":"student|teacher","metadata":{"dialogue_id":...,"turn":...},"content":"..."}

Output row:
  {"meta": {...}, "messages": [{"role":"user|assistant","content":"..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "trainingdata" / "bhagavadgita" / "bhagavaggitatrainingdata.jsonl"
DEFAULT_OUTPUT = ROOT / "trainingdata" / "bhagavadgita" / "bhagavaggitatrainingdata_train.jsonl"
DEFAULT_ROLE = "teacher"
DEFAULT_STYLE = "assistant_turn"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--assistant_role", choices=["student", "teacher"], default=DEFAULT_ROLE)
    p.add_argument("--artifact_style", choices=["dialogue", "assistant_turn"], default=DEFAULT_STYLE)
    p.add_argument("--max_context_messages", type=int, default=12)
    p.add_argument("--min_messages", type=int, default=2)
    p.add_argument("--drop_empty", action="store_true")
    p.add_argument("--require_user_before_assistant", action="store_true")
    p.add_argument("--merge_consecutive_same_role", action="store_true")
    p.add_argument("--strip_role_prefixes", action="store_true")
    return p.parse_args()


def clean_text(text: str, strip_role_prefixes: bool) -> str:
    t = (text or "").replace("\r\n", "\n").strip()
    if not strip_role_prefixes:
        return t
    prefixes = ("user:", "assistant:", "athena:", "teacher:", "student:")
    low = t.lower()
    for pfx in prefixes:
        if low.startswith(pfx):
            return t[len(pfx) :].lstrip()
    return t


def load_by_dialogue(path: Path) -> dict[Any, list[dict[str, Any]]]:
    by_dialogue: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            did = obj.get("metadata", {}).get("dialogue_id")
            by_dialogue[did].append(obj)
    return by_dialogue


def map_messages(
    turns_sorted: list[dict[str, Any]],
    assistant_role: str,
    strip_role_prefixes: bool,
) -> list[dict[str, Any]]:
    user_role = "teacher" if assistant_role == "student" else "student"
    out: list[dict[str, Any]] = []
    for t in turns_sorted:
        src_role = t.get("role")
        content = clean_text(t.get("content", ""), strip_role_prefixes)
        if not content:
            continue
        if src_role == assistant_role:
            role = "assistant"
        elif src_role == user_role:
            role = "user"
        else:
            continue
        out.append({"role": role, "content": content, "source_turn": t.get("metadata", {}).get("turn")})
    return out


def merge_adjacent_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return messages
    merged = [dict(messages[0])]
    for msg in messages[1:]:
        prev = merged[-1]
        if prev["role"] == msg["role"]:
            prev["content"] = f"{prev['content']}\n\n{msg['content']}".strip()
            prev["source_turn"] = msg.get("source_turn", prev.get("source_turn"))
        else:
            merged.append(dict(msg))
    return merged


def emit_dialogue_sample(
    did: Any,
    meta: dict[str, Any],
    messages: list[dict[str, Any]],
    min_messages: int,
    drop_empty: bool,
) -> list[dict[str, Any]]:
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    if drop_empty and assistant_count == 0:
        return []
    if len(messages) < min_messages:
        return []
    out_meta = dict(meta)
    out_meta["dialogue_id"] = did
    out_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    return [{"meta": out_meta, "messages": out_messages}]


def emit_assistant_turn_samples(
    did: Any,
    meta: dict[str, Any],
    messages: list[dict[str, Any]],
    max_context_messages: int,
    min_messages: int,
    drop_empty: bool,
    require_user_before_assistant: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    assistant_positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if drop_empty and not assistant_positions:
        return out

    for target_idx in assistant_positions:
        start_idx = max(0, target_idx - max_context_messages + 1)
        window = messages[start_idx : target_idx + 1]

        first_user_rel = next((i for i, m in enumerate(window) if m["role"] == "user"), None)
        if first_user_rel is not None and first_user_rel > 0:
            window = window[first_user_rel:]

        if len(window) < min_messages:
            continue
        if require_user_before_assistant and not any(m["role"] == "user" for m in window[:-1]):
            continue

        out_meta = dict(meta)
        out_meta["dialogue_id"] = did
        out_meta["sample_style"] = "assistant_turn"
        out_meta["target_index_in_window"] = len(window) - 1
        out_meta["target_source_turn"] = messages[target_idx].get("source_turn")
        out_messages = [{"role": m["role"], "content": m["content"]} for m in window]
        out.append({"meta": out_meta, "messages": out_messages})
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_dialogue = load_by_dialogue(input_path)

    out_samples = 0
    out_messages = 0
    src_turns = 0

    with output_path.open("w", encoding="utf-8") as out:
        for did, turns in sorted(by_dialogue.items(), key=lambda x: (x[0] is None, x[0])):
            src_turns += len(turns)
            turns_sorted = sorted(turns, key=lambda t: t.get("metadata", {}).get("turn", 0))
            meta = dict(turns_sorted[0].get("metadata", {})) if turns_sorted else {}
            meta["dialogue_id"] = did

            messages = map_messages(
                turns_sorted,
                assistant_role=args.assistant_role,
                strip_role_prefixes=bool(args.strip_role_prefixes),
            )
            if args.merge_consecutive_same_role:
                messages = merge_adjacent_same_role(messages)

            if args.artifact_style == "dialogue":
                samples = emit_dialogue_sample(
                    did=did,
                    meta=meta,
                    messages=messages,
                    min_messages=args.min_messages,
                    drop_empty=bool(args.drop_empty),
                )
            else:
                samples = emit_assistant_turn_samples(
                    did=did,
                    meta=meta,
                    messages=messages,
                    max_context_messages=max(2, args.max_context_messages),
                    min_messages=max(2, args.min_messages),
                    drop_empty=bool(args.drop_empty),
                    require_user_before_assistant=bool(args.require_user_before_assistant),
                )

            for sample in samples:
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out_samples += 1
                out_messages += len(sample.get("messages", []))

    avg_messages = (out_messages / out_samples) if out_samples else 0.0
    print(f"Input dialogues: {len(by_dialogue)}")
    print(f"Input turns: {src_turns}")
    print(f"Output samples: {out_samples}")
    print(f"Average messages/sample: {avg_messages:.2f}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

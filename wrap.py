"""Prompt wrapping utilities for Qwen-style chat models.

Why this exists:
- The GUI previously built prompts like "User: ...\nAthena: ..." which is *not*
  the native chat format for Qwen.
- The streaming UI also displayed the *prompt itself* because the streamer was
  not configured to skip the prompt.

This module provides a small, self-contained wrapper that:
- Builds role-based messages (system/user/assistant)
- Renders them with the tokenizer's chat template when available
- Falls back to a simple ChatML-style template when needed

Designed to be imported by ui.py / cli_chat.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Optional file users can create/edit to customize the system prompt without
# touching Python files.
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.txt"
PRIMER_PATH = Path(__file__).resolve().parent / "primer_25.json"

DEFAULT_SYSTEM_PROMPT = (
    "You are Athena, a helpful assistant. "
    "Follow the user's instructions. "
    "Be concise unless asked for depth. "
    "If you are unsure or missing information, say so and ask a clarifying question. "
    "Do not invent facts, sources, or citations." 
)


def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    """Load a system prompt from disk, falling back to DEFAULT_SYSTEM_PROMPT."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or DEFAULT_SYSTEM_PROMPT
    except FileNotFoundError:
        return DEFAULT_SYSTEM_PROMPT


def load_primer_messages(path: Path = PRIMER_PATH) -> List[Dict[str, str]]:
    """Load a few-shot primer from disk.

    File format (JSON): a list of objects like:
      {"role": "user", "content": "..."}
      {"role": "assistant", "content": "..."}

    Any malformed entries are ignored.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except Exception:
        return []

    try:
        data = __import__("json").loads(raw)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    out: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def build_messages_from_history(
    history: Sequence[Tuple[str, str]],
    user_text: str,
    *,
    system_prompt: str,
    primer_messages: Optional[Sequence[Dict[str, str]]] = None,
    max_turns: int = 6,
) -> List[Dict[str, str]]:
    """Convert a simple (user, assistant) history into chat messages."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Optional few-shot primer (inserted after the system message, before history).
    if primer_messages:
        for msg in primer_messages:
            role = (msg.get("role") or "").strip()
            content = (msg.get("content") or "").strip()
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    # Keep only the last N turns (a turn = user+assistant)
    for u, a in list(history)[-max_turns:]:
        u = (u or "").strip()
        a = (a or "").strip()
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_text.strip()})
    return messages


def render_prompt(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Render messages into a single model prompt.

    Primary path: tokenizer.apply_chat_template (model-specific; best for Qwen).
    Fallback path: ChatML-style wrapper.
    """

    # Transformers chat-template path.
    if hasattr(tokenizer, "apply_chat_template"):
        # Newer Qwen templates accept enable_thinking=... (Qwen3 hard switch).
        # We try a few signatures for compatibility.
        try:
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            if enable_thinking is not None:
                kwargs["enable_thinking"] = bool(enable_thinking)
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Older/newer signatures: try dropping add_generation_prompt.
            try:
                kwargs = {"tokenize": False}
                if enable_thinking is not None:
                    kwargs["enable_thinking"] = bool(enable_thinking)
                return tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:
                pass
        except Exception:
            pass

    # Generic ChatML fallback.
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    parts: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").strip()
        content = msg.get("content") or ""
        parts.append(f"{im_start}{role}\n{content}{im_end}")

    if add_generation_prompt:
        parts.append(f"{im_start}assistant\n")

    return "\n".join(parts)


def build_prompt(
    tokenizer,
    history: Sequence[Tuple[str, str]],
    user_text: str,
    *,
    system_prompt: str,
    primer_messages: Optional[Sequence[Dict[str, str]]] = None,
    max_turns: int = 6,
    enable_thinking: Optional[bool] = None,
) -> str:
    """One-call helper used by the GUI."""
    messages = build_messages_from_history(
        history,
        user_text,
        system_prompt=system_prompt,
        primer_messages=primer_messages,
        max_turns=max_turns,
    )
    return render_prompt(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


_CHATML_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>")


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> sections (Qwen3 reasoning) from a text blob."""
    if not text:
        return ""

    t = text
    # Remove full blocks first.
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE)

    # If a stray closing tag remains (e.g. some templates auto-insert <think> in the prompt),
    # drop everything up to it.
    if "</think>" in t.lower():
        # Case-insensitive split keeping original content.
        m = re.search(r"</think>", t, flags=re.IGNORECASE)
        if m:
            t = t[m.end() :]

    # Remove any remaining tag literals.
    t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE)
    return t


class ThinkStripper:
    """Streaming filter that hides <think>...</think> while still streaming the final answer.

    This is useful when enable_thinking=True, but you don't want chain-of-thought (CoT)
    to appear in the UI.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._in_think = False
        self._pending = ""

    def feed(self, chunk: str) -> str:
        if (not self.enabled) or (not chunk):
            return chunk or ""

        self._pending += chunk
        out: List[str] = []

        while self._pending:
            if self._in_think:
                idx = self._pending.lower().find(self._CLOSE)
                if idx != -1:
                    # Discard think content including the closing tag.
                    self._pending = self._pending[idx + len(self._CLOSE) :]
                    self._in_think = False
                    continue

                # No closing tag yet: discard everything except a small tail in case the
                # closing tag is split across chunks.
                keep = len(self._CLOSE) - 1
                if len(self._pending) > keep:
                    self._pending = self._pending[-keep:]
                break

            # Not currently inside a think block.
            idx_open = self._pending.lower().find(self._OPEN)
            idx_close = self._pending.lower().find(self._CLOSE)

            # Handle edge case: stray closing tag without an opening tag.
            if idx_close != -1 and (idx_open == -1 or idx_close < idx_open):
                # Drop everything up to and including the closing tag.
                self._pending = self._pending[idx_close + len(self._CLOSE) :]
                continue

            if idx_open != -1:
                # Emit text before <think>, then enter think mode.
                if idx_open > 0:
                    out.append(self._pending[:idx_open])
                self._pending = self._pending[idx_open + len(self._OPEN) :]
                self._in_think = True
                continue

            # No tags found: emit most content, keep a small tail for split-tag safety.
            keep = len(self._OPEN) - 1
            if len(self._pending) > keep:
                out.append(self._pending[:-keep])
                self._pending = self._pending[-keep:]
            break

        return "".join(out)

    def flush(self) -> str:
        """Call at end of generation to emit any remaining non-think text."""
        if not self.enabled:
            t = self._pending
            self._pending = ""
            self._in_think = False
            return t

        if self._in_think:
            # Drop any remaining buffered think content.
            self._pending = ""
            self._in_think = False
            return ""

        t = self._pending
        self._pending = ""
        # Clean any partial tag fragments.
        t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE)
        return t


def clean_assistant_text(text: str) -> str:
    """Best-effort cleanup for streamed assistant text."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n")
    t = _CHATML_TOKEN_RE.sub("", t)

    # Remove any chain-of-thought blocks.
    t = strip_think_blocks(t)

    # Remove an echoed role prefix if the model included it.
    t = re.sub(r"^\s*(assistant|athena)\s*:\s*", "", t, flags=re.IGNORECASE)

    return t.strip()

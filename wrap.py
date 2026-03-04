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

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Optional file users can create/edit to customize the system prompt without
# touching Python files.
SYSTEM_PROMPT_JSON_PATH = Path(__file__).resolve().parent / "system_prompt.json"
SYSTEM_PROMPT_TXT_PATH = Path(__file__).resolve().parent / "system_prompt.txt"
SYSTEM_PROMPT_PATH = SYSTEM_PROMPT_TXT_PATH

DEFAULT_SYSTEM_PROMPT = (
    "Answer in exactly one clear sentence."
)


def _as_str_lines(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                t = item.strip()
                if t:
                    out.append(t)
        return out
    return []


def _render_system_prompt_from_json(cfg: dict[str, Any]) -> str:
    direct = cfg.get("system_prompt")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: List[str] = []

    persona = cfg.get("persona")
    if isinstance(persona, str) and persona.strip():
        chunks.append(persona.strip())

    section_specs = [
        ("core_behavior", "Core behavior:"),
        ("math_response_protocol", "Math response protocol:"),
        ("formatting_rules", "Formatting rules:"),
        ("default_mode", "Default mode:"),
    ]
    for key, label in section_specs:
        lines = _as_str_lines(cfg.get(key))
        if lines:
            body = "\n".join(f"- {line}" for line in lines)
            chunks.append(f"{label}\n{body}")

    few_shots = cfg.get("few_shots")
    if isinstance(few_shots, list):
        examples: List[str] = []
        idx = 1
        for item in few_shots:
            if not isinstance(item, dict):
                continue
            user = item.get("user")
            assistant = item.get("assistant")
            if not isinstance(user, str) or not isinstance(assistant, str):
                continue
            u = user.strip()
            a = assistant.strip()
            if not u or not a:
                continue
            examples.append(f"Example {idx}\nUser: {u}\nAssistant:\n{a}")
            idx += 1
        if examples:
            chunks.append("Few-shot style examples:\n\n" + "\n\n".join(examples))

    text = "\n\n".join(chunks).strip()
    return text or DEFAULT_SYSTEM_PROMPT


def load_system_prompt(path: Optional[Path] = None) -> str:
    """Load a system prompt from disk, falling back to DEFAULT_SYSTEM_PROMPT.

    Resolution order:
    1) Explicit `path` argument if provided.
    2) `system_prompt.json` (JSON-first config).
    3) `system_prompt.txt` (legacy plain text).
    """
    if path is not None:
        if not path.exists():
            return DEFAULT_SYSTEM_PROMPT
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict):
                    return _render_system_prompt_from_json(data)
                return DEFAULT_SYSTEM_PROMPT
            except Exception:
                return DEFAULT_SYSTEM_PROMPT
        try:
            text = path.read_text(encoding="utf-8-sig").strip()
            return text or DEFAULT_SYSTEM_PROMPT
        except Exception:
            return DEFAULT_SYSTEM_PROMPT

    if SYSTEM_PROMPT_JSON_PATH.exists():
        try:
            data = json.loads(SYSTEM_PROMPT_JSON_PATH.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                text = _render_system_prompt_from_json(data)
                if text:
                    return text
        except Exception:
            pass

    try:
        text = SYSTEM_PROMPT_TXT_PATH.read_text(encoding="utf-8-sig").strip()
        return text or DEFAULT_SYSTEM_PROMPT
    except FileNotFoundError:
        return DEFAULT_SYSTEM_PROMPT


def build_messages_from_history(
    history: Sequence[Tuple[str, str]],
    user_text: str,
    *,
    system_prompt: str,
    max_turns: int = 6,
    user_images: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert a simple (user, assistant) history into chat messages."""
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # Keep only the last N turns (a turn = user+assistant)
    for u, a in list(history)[-max_turns:]:
        u = (u or "").strip()
        a = (a or "").strip()
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})

    clean_user = user_text.strip()
    if user_images:
        content: List[Dict[str, Any]] = []
        for image in user_images:
            if isinstance(image, Path):
                content.append({"type": "image", "image": str(image)})
                continue
            if isinstance(image, str):
                image_s = image.strip()
                content.append({"type": "image", "image": image_s})
                continue
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": clean_user or "Describe this image."})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": clean_user})
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
    max_turns: int = 6,
    enable_thinking: Optional[bool] = None,
    user_images: Optional[Sequence[Any]] = None,
) -> str:
    """One-call helper used by the GUI."""
    messages = build_messages_from_history(
        history,
        user_text,
        system_prompt=system_prompt,
        max_turns=max_turns,
        user_images=user_images,
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

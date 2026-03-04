from __future__ import annotations

import base64
import html
import re
from typing import Iterable

from markdown_it import MarkdownIt


def _safe_unicode(text: str) -> str:
    # Preserve emoji and replace broken surrogates defensively.
    try:
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    except Exception:
        return text.encode("utf-8", "replace").decode("utf-8", "replace")


def _new_markdown_renderer() -> MarkdownIt:
    md = MarkdownIt(
        "commonmark",
        {
            "html": False,
            "linkify": True,
            "typographer": False,
            "breaks": True,
        },
    )
    # Allow local file URIs for pasted image previews in the desktop UI transcript.
    # Keep default validation for all other schemes.
    default_validate_link = md.validateLink
    md.validateLink = lambda url: url.lower().startswith("file://") or default_validate_link(url)  # type: ignore[assignment]
    return md


_TEX_DELIM_RE = re.compile(r"(?<!\\)\\([()\[\]])")


def _preserve_tex_delimiters(text: str) -> str:
    # markdown-it treats \\(...\\) and \\[...\\] escapes as plain brackets unless
    # we preserve the TeX delimiters before markdown conversion.
    return _TEX_DELIM_RE.sub(r"\\\\\\1", text)


def _role_label(role: str) -> str:
    if role == "user":
        return "User"
    if role == "assistant":
        return "Athena"
    if role == "system":
        return "System"
    return role.capitalize()


def _role_icon(role: str) -> str:
    if role == "user":
        return "\U0001F9D1"
    if role == "assistant":
        return "\U0001F9E0"
    if role == "system":
        return "\u2699\ufe0f"
    return "\U0001F4AC"


def render_message_body_html(content: str) -> str:
    md = _new_markdown_renderer()
    safe_content = _preserve_tex_delimiters(_safe_unicode(content or ""))
    return md.render(safe_content)


def render_transcript_html(messages: Iterable[dict[str, str]]) -> str:
    chunks: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "assistant").strip().lower()
        raw_content = msg.get("content") or ""
        raw_b64 = base64.b64encode(raw_content.encode("utf-8")).decode("ascii")
        body = render_message_body_html(raw_content)
        chunks.append(
            (
                f'<article class="msg {html.escape(role)}">'
                f'<aside class="avatar" aria-hidden="true">{html.escape(_role_icon(role))}</aside>'
                '<div class="bubble">'
                '<div class="bubble-head">'
                f'<span class="role-pill"><span class="role-icon">{html.escape(_role_icon(role))}</span>{html.escape(_role_label(role))}</span>'
                '<button class="copy-msg-btn" type="button" title="Copy raw message" aria-label="Copy raw message">Copy</button>'
                "</div>"
                f'<section class="msg-body" data-raw-b64="{html.escape(raw_b64)}">{body}</section>'
                "</div>"
                "</article>"
            )
        )
    return "\n".join(chunks)

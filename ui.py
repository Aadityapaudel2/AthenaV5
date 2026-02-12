#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def ensure_project_venv() -> None:
    """Re-launch with the workspace venv Python when available."""
    if os.environ.get("ATHENA_UI_VENV_BOOTSTRAPPED") == "1":
        return
    if not VENV_PYTHON.exists():
        return

    current = Path(sys.executable).resolve()
    target = VENV_PYTHON.resolve()
    if current == target:
        return

    os.environ["ATHENA_UI_VENV_BOOTSTRAPPED"] = "1"
    os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_project_venv()

sys.path.append(str(ROOT))

from athena_paths import get_default_chat_model_dir
from tk_chat import GuiSettings, LocalStreamer, CONFIG_PATH
import wrap

BG = "#0d1117"
FG = "#f5f5f5"
ENTRY_BG = "#1a1f2b"
ENTRY_FG = "#e0e6ff"
PANEL_BORDER = 1

LOG_DIR = ROOT / "logs"
RAW_LOG = LOG_DIR / "raw.log"
CLEAN_LOG = LOG_DIR / "clean.log"


def ensure_logs() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    RAW_LOG.touch(exist_ok=True)
    CLEAN_LOG.touch(exist_ok=True)


def append_log(path: Path, label: str, text: str) -> None:
    # Preserve multi-line content, but keep a single timestamp/label prefix.
    safe = (text or "").replace("\r\n", "\n")
    with open(path, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {label}: {safe}\n")


class AthenaUI:
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        ensure_logs()

        self.settings = GuiSettings.load(CONFIG_PATH)
        resolved_model_dir = model_dir if model_dir is not None else get_default_chat_model_dir()
        self.streamer = LocalStreamer(self.settings, model_dir=resolved_model_dir)

        self.system_prompt = wrap.load_system_prompt()
        # Optional few-shot primer messages (25 examples by default).
        primer_path = (ROOT / getattr(self.settings, "primer_path", "primer_25.json")).resolve()
        self.primer_messages = wrap.load_primer_messages(primer_path)
        self.history: List[Tuple[str, str]] = []

        self.is_streaming = False

        # Internal: if we hide thoughts, Qwen3 may output a long <think>...</think> block
        # before the final answer. We keep the UI responsive by showing a temporary
        # "Thinking..." placeholder in the chat window.
        self._thinking_placeholder: tuple[str, str] | None = None

        self.root = tk.Tk()
        self.root.title(f"Athena V5  -  {self.streamer.model_label}")
        self.root.geometry("820x520")
        self.root.configure(bg=BG)

        # Tk variables must be created after the root window exists.
        self.var_thinking = tk.BooleanVar(value=bool(self.settings.enable_thinking))
        self.var_show_thoughts = tk.BooleanVar(value=not bool(self.settings.hide_thoughts))
        self.var_use_primer = tk.BooleanVar(value=bool(getattr(self.settings, "use_primer", True)))

        self.output: ScrolledText
        self.entry: tk.Text
        self.status: ttk.Label
        self.btn_send: ttk.Button
        self.btn_stop: ttk.Button
        self.btn_clear: ttk.Button
        self.chk_thinking: ttk.Checkbutton
        self.chk_show_thoughts: ttk.Checkbutton
        self.chk_primer: ttk.Checkbutton

        self._build_ui()
        self._append_output(f"Loaded model: {self.streamer.model_dir}\n")

    def _build_ui(self) -> None:
        self.output = ScrolledText(
            self.root,
            wrap=tk.WORD,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            insertbackground=ENTRY_FG,
            bd=PANEL_BORDER,
            relief=tk.SUNKEN,
            state=tk.DISABLED,
            pady=8,
            font=("Consolas", 10),
        )
        self.output.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 6))

        controls = ttk.Frame(self.root)
        controls.pack(fill=tk.X, padx=12, pady=(0, 8))
        controls.configure(style="Custom.TFrame")

        self.entry = tk.Text(
            controls,
            height=4,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            insertbackground=ENTRY_FG,
            bd=PANEL_BORDER,
            relief=tk.SUNKEN,
            font=("Segoe UI", 11),
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Keybinds: Enter sends, Shift+Enter inserts newline
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<Shift-Return>", self._on_shift_enter)

        btns = ttk.Frame(controls)
        btns.pack(side=tk.LEFT, padx=(8, 0))
        btns.configure(style="Custom.TFrame")

        self.btn_send = ttk.Button(btns, text="Send", command=self.send)
        self.btn_send.pack(fill=tk.X)

        self.chk_thinking = ttk.Checkbutton(
            btns,
            text="Thinking",
            variable=self.var_thinking,
            command=self._toggle_thinking,
        )
        self.chk_thinking.pack(fill=tk.X, pady=(6, 0))

        self.chk_show_thoughts = ttk.Checkbutton(
            btns,
            text="Show thoughts",
            variable=self.var_show_thoughts,
            command=self._toggle_show_thoughts,
        )
        self.chk_show_thoughts.pack(fill=tk.X, pady=(4, 0))

        self.chk_primer = ttk.Checkbutton(
            btns,
            text="Style primer",
            variable=self.var_use_primer,
            command=self._toggle_primer,
        )
        self.chk_primer.pack(fill=tk.X, pady=(4, 0))

        self.btn_stop = ttk.Button(btns, text="Stop", command=self.stop)
        self.btn_stop.pack(fill=tk.X, pady=(6, 0))
        self.btn_clear = ttk.Button(btns, text="Clear", command=self.clear)
        self.btn_clear.pack(fill=tk.X, pady=(6, 0))

        self.status = ttk.Label(
            self.root,
            text=self._status_text("Ready"),
            anchor="w",
            background=BG,
            foreground=FG,
        )
        self.status.pack(fill=tk.X, padx=12, pady=(0, 12))

        style = ttk.Style(self.root)
        style.configure("TButton", background=BG, foreground=FG)
        style.configure("Custom.TFrame", background=BG)

    def _status_text(self, state: str) -> str:
        thinking = "on" if self.var_thinking.get() else "off"
        show_thoughts = "on" if self.var_show_thoughts.get() else "off"
        primer = "on" if self.var_use_primer.get() else "off"
        return (
            f"{state}  |  thinking={thinking}  show_thoughts={show_thoughts}  primer={primer}  |  "
            f"model={self.streamer.model_dir}  temp={self.settings.temperature}  top_p={self.settings.top_p}  "
            f"top_k={self.settings.top_k}  max_new_tokens={self.settings.max_new_tokens}"
        )

    def set_status(self, state: str) -> None:
        self.status.config(text=self._status_text(state))

    def _append_output(self, text: str) -> None:
        if not text:
            return
        self.output.config(state=tk.NORMAL)
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.config(state=tk.DISABLED)

    def _on_enter(self, event) -> str:
        # Prevent Text widget from inserting a newline.
        self.send()
        return "break"

    def _on_shift_enter(self, event) -> None:
        # Allow newline insertion.
        return None

    def build_prompt(self, user_text: str) -> str:
        return wrap.build_prompt(
            self.streamer.tokenizer,
            self.history,
            user_text,
            system_prompt=self.system_prompt,
            primer_messages=(self.primer_messages if self.var_use_primer.get() else None),
            max_turns=6,
            enable_thinking=self.var_thinking.get(),
        )

    def _toggle_thinking(self) -> None:
        # Persist in gui_config.json so the toggle survives restarts.
        self.settings.enable_thinking = bool(self.var_thinking.get())
        try:
            self.settings.save(CONFIG_PATH)
        except Exception:
            # Non-fatal: UI still works.
            pass
        self.set_status("Ready")

    def _toggle_show_thoughts(self) -> None:
        # Persist in gui_config.json.
        self.settings.hide_thoughts = not bool(self.var_show_thoughts.get())
        try:
            self.settings.save(CONFIG_PATH)
        except Exception:
            pass
        self.set_status("Ready")

    def _toggle_primer(self) -> None:
        # Persist in gui_config.json.
        self.settings.use_primer = bool(self.var_use_primer.get())
        try:
            self.settings.save(CONFIG_PATH)
        except Exception:
            pass
        self.set_status("Ready")

    def _show_thinking_placeholder(self) -> None:
        if self._thinking_placeholder is not None:
            return
        placeholder = "<thinking...>"
        self.output.config(state=tk.NORMAL)
        start = self.output.index(tk.END)
        self.output.insert(tk.END, placeholder)
        end = self.output.index(tk.END)
        self.output.see(tk.END)
        self.output.config(state=tk.DISABLED)
        self._thinking_placeholder = (start, end)

    def _clear_thinking_placeholder(self) -> None:
        if self._thinking_placeholder is None:
            return
        start, end = self._thinking_placeholder
        self.output.config(state=tk.NORMAL)
        try:
            self.output.delete(start, end)
        except Exception:
            # If indices are stale for any reason, ignore.
            pass
        self.output.config(state=tk.DISABLED)
        self._thinking_placeholder = None

    def send(self) -> None:
        if self.is_streaming:
            return

        user_text = self.entry.get("1.0", "end-1c").strip()
        if not user_text:
            return

        self.entry.delete("1.0", tk.END)

        # Log + UI preface
        append_log(RAW_LOG, "USER", user_text)
        self._append_output(f"\nUser: {user_text}\nAthena: ")

        self.is_streaming = True
        self.btn_send.config(state=tk.DISABLED)
        self.chk_thinking.config(state=tk.DISABLED)
        self.chk_show_thoughts.config(state=tk.DISABLED)
        self.chk_primer.config(state=tk.DISABLED)

        hide_thoughts = not bool(self.var_show_thoughts.get())
        if self.var_thinking.get() and hide_thoughts:
            self.set_status("Thinking... (hidden)")
            # Avoid a "frozen" looking UI while the model is emitting <think>.
            self._show_thinking_placeholder()
        else:
            self.set_status("Streaming...")

        assistant_chunks: List[str] = []
        think_stripper = wrap.ThinkStripper(enabled=hide_thoughts)

        def on_chunk(chunk: str) -> None:
            visible = think_stripper.feed(chunk)
            if not visible:
                return
            assistant_chunks.append(visible)

            def ui_update(c: str = visible) -> None:
                # First visible tokens: clear the placeholder if it was shown.
                self._clear_thinking_placeholder()
                self._append_output(c)

            self.root.after(0, ui_update)

        def worker() -> None:
            try:
                prompt = self.build_prompt(user_text)
                self.streamer.stream(prompt, on_chunk)
            finally:
                tail = think_stripper.flush()
                if tail:
                    assistant_chunks.append(tail)
                    self.root.after(0, lambda c=tail: (self._clear_thinking_placeholder(), self._append_output(c)))

                assistant_text = wrap.clean_assistant_text("".join(assistant_chunks))

                if assistant_text:
                    self.history.append((user_text, assistant_text))
                    # Keep a small rolling window.
                    if len(self.history) > 12:
                        self.history = self.history[-12:]

                append_log(RAW_LOG, "ASSISTANT", assistant_text)
                append_log(CLEAN_LOG, "ASSISTANT", assistant_text)

                # Finish the UI update on the main thread.
                self.root.after(0, self._finish_stream)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_stream(self) -> None:
        self._clear_thinking_placeholder()
        self._append_output("\n")
        self.is_streaming = False
        self.btn_send.config(state=tk.NORMAL)
        self.chk_thinking.config(state=tk.NORMAL)
        self.chk_show_thoughts.config(state=tk.NORMAL)
        self.chk_primer.config(state=tk.NORMAL)
        self.set_status("Ready")

    def stop(self) -> None:
        if not self.is_streaming:
            self.set_status("Ready")
            return
        self.streamer.stop()
        self._clear_thinking_placeholder()
        self.set_status("Stopped")

    def clear(self) -> None:
        if self.is_streaming:
            self.streamer.stop()
        self.history.clear()
        self._clear_thinking_placeholder()
        self.output.config(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.config(state=tk.DISABLED)
        self.set_status("Ready")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="", help="Optional model path override for this UI instance.")
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else None
    AthenaUI(model_dir=model_dir).run()


if __name__ == "__main__":
    main()

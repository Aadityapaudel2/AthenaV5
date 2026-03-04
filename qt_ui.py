#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT

# Quiet + software-safe defaults for Windows QtWebEngine.
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
os.environ.setdefault(
    "QT_LOGGING_RULES",
    "qt.webenginecontext.warning=false;qt.webenginecontext.info=false;qt.webenginecontext.debug=false;qt.qpa.gl=false",
)
_required_qt_flags = [
    "--no-sandbox",
    "--disable-gpu-sandbox",
    "--disable-gpu",
    "--disable-gpu-compositing",
    "--use-angle=swiftshader",
    "--disable-logging",
    "--log-level=3",
]
_existing_qt_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
_flag_set = set(_existing_qt_flags.split()) if _existing_qt_flags else set()
for _flag in _required_qt_flags:
    if _flag not in _flag_set:
        _existing_qt_flags = f"{_existing_qt_flags} {_flag}".strip()
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = _existing_qt_flags
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("QT_QUICK_BACKEND", "software")


def _candidate_venv_pythons() -> list[Path]:
    return [
        p
        for p in (
            PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
            PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe",
        )
        if p.exists()
    ]


def ensure_project_venv() -> None:
    if os.environ.get("ATHENA_UI_VENV_BOOTSTRAPPED") == "1":
        return
    candidates = _candidate_venv_pythons()
    if not candidates:
        return
    current = Path(sys.executable).resolve()
    target = candidates[0].resolve()
    if current == target:
        return
    os.environ["ATHENA_UI_VENV_BOOTSTRAPPED"] = "1"
    os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_project_venv()

try:
    from PySide6.QtCore import QObject, QTimer, Qt, Signal  # pyright: ignore[reportMissingImports]
    from PySide6.QtGui import QFont, QImage, QKeyEvent, QPixmap  # pyright: ignore[reportMissingImports]
    from PySide6.QtWebEngineWidgets import QWebEngineView  # pyright: ignore[reportMissingImports]
    from PySide6.QtWidgets import (  # pyright: ignore[reportMissingImports]
        QApplication,
        QCheckBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:
    print(
        "PySide6 + QtWebEngine is required for qt_ui.py.\n"
        "Install in your project venv:\n"
        "  D:\\AthenaPlayground\\.venv\\Scripts\\python.exe -m pip install PySide6\n"
        f"Import error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)

sys.path.append(str(ROOT))

from athena_paths import get_default_chat_model_dir  # noqa: E402  # pyright: ignore[reportMissingImports]
from qt_render import render_message_body_html, render_transcript_html  # noqa: E402  # pyright: ignore[reportMissingImports]
from tk_chat import CONFIG_PATH, GuiSettings, LocalStreamer  # noqa: E402  # pyright: ignore[reportMissingImports]
import wrap  # noqa: E402  # pyright: ignore[reportMissingImports]

BG = "#0d1117"
FG = "#f5f5f5"
LOG_DIR = Path(os.environ.get("ATHENA_LOG_DIR", str(Path.home() / ".athena_v5" / "logs"))).expanduser()
RAW_LOG = LOG_DIR / "raw.log"
CLEAN_LOG = LOG_DIR / "clean.log"
UI_EVENTS_LOG = LOG_DIR / "ui_events.jsonl"
ASSETS_HTML = (ROOT / "assets" / "chat_shell.html").resolve()
LOGGING_ENABLED = True
CHAR_STREAM_INTERVAL_MS = 14
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
ASCII_AVATAR_TEXT = "[^_^]"
ASCII_AVATAR_BY_STATE = {
    "idle": "[^_^]",
    "thinking": "[-_-]",
    "speaking": "[o_o]",
    "happy": "[^o^]",
    "confused": "[?_?]",
    "stopped": "[x_x]",
    "error": "[>_<]",
}


def ensure_logs() -> None:
    global LOGGING_ENABLED
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        RAW_LOG.touch(exist_ok=True)
        CLEAN_LOG.touch(exist_ok=True)
        UI_EVENTS_LOG.touch(exist_ok=True)
        LOGGING_ENABLED = True
    except Exception:
        LOGGING_ENABLED = False


def append_log(path: Path, label: str, text: str) -> None:
    if not LOGGING_ENABLED:
        return
    safe = (text or "").replace("\r\n", "\n")
    with open(path, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {label}: {safe}\n")


def append_ui_event(event: str, mode: str, model_dir: str, details: Optional[dict] = None) -> None:
    if not LOGGING_ENABLED:
        return
    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": event,
        "mode": mode,
        "model_dir": model_dir,
        "details": details or {},
    }
    with open(UI_EVENTS_LOG, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_model_proof_text(model_proof: dict) -> str:
    sha256 = str(model_proof.get("config_sha256") or "")
    arch = model_proof.get("architectures") or []
    if isinstance(arch, (list, tuple)):
        arch_text = ", ".join(str(x) for x in arch) if arch else "n/a"
    else:
        arch_text = str(arch)
    model_dir = str(model_proof.get("model_dir") or "")
    name_or_path = str(model_proof.get("name_or_path") or "n/a")
    model_type = str(model_proof.get("model_type") or "n/a")
    hidden_size = model_proof.get("hidden_size")
    num_layers = model_proof.get("num_hidden_layers")
    num_heads = model_proof.get("num_attention_heads")
    vocab_size = model_proof.get("vocab_size")
    max_pos = model_proof.get("max_position_embeddings")
    target_model = get_default_chat_model_dir().name
    target_key = target_model.lower()
    target_hit = (target_key in model_dir.lower()) or (target_key in name_or_path.lower())
    return (
        "Model proof:\n"
        f"target={target_model} match={'YES' if target_hit else 'NO'}\n"
        f"path={model_dir}\n"
        f"name_or_path={name_or_path}\n"
        f"model_type={model_type}\n"
        f"architectures={arch_text}\n"
        f"layers={num_layers} hidden={hidden_size} heads={num_heads} vocab={vocab_size} max_pos={max_pos}\n"
        f"config_sha256={sha256 if sha256 else 'n/a'}"
    )


def format_runtime_config_text(runtime_cfg: dict) -> str:
    return (
        "Loaded runtime config:\n"
        f"model_dir={runtime_cfg.get('model_dir')}\n"
        f"model_label={runtime_cfg.get('model_label')}\n"
        f"device={runtime_cfg.get('device')} dtype={runtime_cfg.get('dtype')}\n"
        f"temperature={runtime_cfg.get('temperature')} max_new_tokens={runtime_cfg.get('max_new_tokens')}\n"
        f"top_p={runtime_cfg.get('top_p')} top_k={runtime_cfg.get('top_k')} repetition_penalty={runtime_cfg.get('repetition_penalty')}\n"
        f"do_sample={runtime_cfg.get('do_sample')} eos_token_id={runtime_cfg.get('eos_token_id')} pad_token_id={runtime_cfg.get('pad_token_id')}\n"
        f"enable_thinking={runtime_cfg.get('enable_thinking')} hide_thoughts={runtime_cfg.get('hide_thoughts')}\n"
        f"renderer_mode={runtime_cfg.get('renderer_mode')} render_throttle_ms={runtime_cfg.get('render_throttle_ms')}\n"
        f"supports_vision={runtime_cfg.get('supports_vision')} image_processor_loaded={runtime_cfg.get('image_processor_loaded')}"
    )


class InputTextEdit(QTextEdit):
    sendRequested = Signal()
    imagesPasted = Signal(object)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                event.accept()
                self.sendRequested.emit()
            return
        super().keyPressEvent(event)

    def insertFromMimeData(self, source) -> None:  # type: ignore[override]
        pasted: list[object] = []
        try:
            if source.hasImage():
                image_data = source.imageData()
                if isinstance(image_data, QImage) and not image_data.isNull():
                    pasted.append(image_data)
        except Exception:
            pass

        try:
            if source.hasUrls():
                for url in source.urls():
                    if not url.isLocalFile():
                        continue
                    local_path = url.toLocalFile()
                    if not local_path:
                        continue
                    if Path(local_path).suffix.lower() in IMAGE_SUFFIXES:
                        pasted.append(local_path)
        except Exception:
            pass

        if pasted:
            self.imagesPasted.emit(pasted)
            return
        super().insertFromMimeData(source)


class StreamSignals(QObject):
    chunk = Signal(str)
    finished = Signal(str, str)
    error = Signal(str)


class AthenaQtUI(QMainWindow):
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        super().__init__()
        ensure_logs()

        self.settings = GuiSettings.load(CONFIG_PATH)
        if self.settings.renderer_mode != "qt_web":
            self.settings.renderer_mode = "qt_web"
            try:
                self.settings.save(CONFIG_PATH)
            except Exception:
                pass

        resolved_model_dir = model_dir if model_dir is not None else get_default_chat_model_dir()
        self.streamer = LocalStreamer(self.settings, model_dir=resolved_model_dir)
        self.system_prompt = wrap.load_system_prompt()
        self.history: list[tuple[str, str]] = []
        self._loaded_runtime_cfg = self.streamer.runtime_config()
        self.transcript: list[dict[str, str]] = [
            {"role": "system", "content": f"Loaded model: {self.streamer.model_dir}"},
            {"role": "system", "content": format_model_proof_text(self.streamer.model_proof)},
            {"role": "system", "content": format_runtime_config_text(self._loaded_runtime_cfg)},
        ]

        self.is_streaming = False
        self._current_assistant_idx: Optional[int] = None
        self._web_ready = False
        self._stop_requested = False
        self._last_full_html: str = ""
        self._last_assistant_body_html: str = ""
        self._stream_char_queue: deque[str] = deque()
        self._pending_finish_payload: Optional[tuple[str, str]] = None
        self._avatar_state: str = "idle"
        self._stream_had_error = False
        self._last_stream_error_message = ""
        self._pending_image_paths: list[Path] = []
        self._session_temp_image_paths: list[Path] = []
        self._image_stage_dir = (ROOT / "data" / "clipboard_images").resolve()
        self._image_stage_dir.mkdir(parents=True, exist_ok=True)

        self.signals = StreamSignals()
        self.signals.chunk.connect(self._on_stream_chunk)
        self.signals.finished.connect(self._on_stream_finished)
        self.signals.error.connect(self._on_stream_error)

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._flush_render)
        self.render_throttle_ms = max(20, min(500, int(self.settings.render_throttle_ms)))

        self.char_timer = QTimer(self)
        self.char_timer.setInterval(CHAR_STREAM_INTERVAL_MS)
        self.char_timer.timeout.connect(self._drain_stream_chars)

        self._build_ui()
        self._refresh_image_label()
        self._render_now()

        append_ui_event(
            "launch_start",
            mode="qt-web",
            model_dir=str(self.streamer.model_dir),
            details={"renderer_mode": self.settings.renderer_mode},
        )

    def _build_ui(self) -> None:
        self.setWindowTitle(f"Athena V5  -  {self.streamer.model_label}")
        self.resize(1080, 760)

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        top = QWidget(self)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        self.web = QWebEngineView(self)
        self.web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.web.loadFinished.connect(self._on_web_loaded)
        self.web.setUrl(ASSETS_HTML.as_uri())
        top_layout.addWidget(self.web, stretch=1)

        side = QWidget(self)
        side.setFixedWidth(220)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        self.avatar_frame = QLabel(side)
        self.avatar_frame.setObjectName("AvatarFrame")
        self.avatar_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.avatar_frame.setMargin(3)
        self.avatar_frame.setFixedSize(196, 196)
        self.avatar_frame.setTextFormat(Qt.TextFormat.PlainText)
        self.avatar_frame.setFont(QFont("Cascadia Mono", 22))
        self.avatar_frame.setText(ASCII_AVATAR_TEXT)
        side_layout.addWidget(self.avatar_frame)
        side_layout.addStretch(1)

        top_layout.addWidget(side, stretch=0)
        layout.addWidget(top, stretch=1)

        bottom = QWidget(self)
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        left = QWidget(bottom)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.entry = InputTextEdit(left)
        self.entry.setMinimumHeight(92)
        self.entry.setPlaceholderText("Message Athena...  (Enter = send, Shift+Enter = newline)")
        self.entry.setFont(QFont("Segoe UI Emoji", 11))
        self.entry.sendRequested.connect(self.send)
        self.entry.imagesPasted.connect(self._on_images_pasted)
        left_layout.addWidget(self.entry, stretch=1)

        image_row = QWidget(left)
        image_row_layout = QHBoxLayout(image_row)
        image_row_layout.setContentsMargins(0, 0, 0, 0)
        image_row_layout.setSpacing(8)

        self.image_label = QLabel("Images: none", image_row)
        self.image_label.setStyleSheet("color: #9eb2d7;")
        image_row_layout.addWidget(self.image_label, stretch=1)

        self.image_clear_btn = QPushButton("Clear Images")
        self.image_clear_btn.clicked.connect(self._clear_pending_images)
        self.image_clear_btn.setEnabled(False)
        image_row_layout.addWidget(self.image_clear_btn, stretch=0)

        left_layout.addWidget(image_row, stretch=0)

        self.image_preview = QLabel(left)
        self.image_preview.setFixedHeight(120)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.image_preview.setStyleSheet(
            "border: 1px solid #2d4b7a; border-radius: 8px; background: #0f1728; color: #9eb2d7; padding: 4px;"
        )
        self.image_preview.setText("No image preview")
        self.image_preview.setVisible(False)
        left_layout.addWidget(self.image_preview, stretch=0)

        bottom_layout.addWidget(left, stretch=1)

        controls = QWidget(bottom)
        ctl_layout = QVBoxLayout(controls)
        ctl_layout.setContentsMargins(0, 0, 0, 0)
        ctl_layout.setSpacing(6)

        self.btn_send = QPushButton("Send")
        self.btn_send.clicked.connect(self.send)
        ctl_layout.addWidget(self.btn_send)

        self.chk_thinking = QCheckBox("Thinking")
        self.chk_thinking.setChecked(bool(self.settings.enable_thinking))
        self.chk_thinking.stateChanged.connect(self._toggle_thinking)
        ctl_layout.addWidget(self.chk_thinking)

        self.chk_show_thoughts = QCheckBox("Show thoughts")
        self.chk_show_thoughts.setChecked(not bool(self.settings.hide_thoughts))
        self.chk_show_thoughts.stateChanged.connect(self._toggle_show_thoughts)
        ctl_layout.addWidget(self.chk_show_thoughts)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop)
        ctl_layout.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear)
        ctl_layout.addWidget(self.btn_clear)
        ctl_layout.addStretch(1)

        bottom_layout.addWidget(controls, stretch=0)
        layout.addWidget(bottom, stretch=0)

        self.status = QLabel(self._status_text("Ready"), self)
        self.status.setStyleSheet(f"color: {FG}; background: {BG}; padding: 2px;")
        layout.addWidget(self.status)

        root.setStyleSheet(
            """
            QWidget { background: #0b111d; color: #eef4ff; }
            QTextEdit {
                background: #121c30;
                color: #e8f0ff;
                border: 1px solid #2d4b7a;
                border-radius: 8px;
                padding: 8px;
            }
            QTextEdit:focus { border: 1px solid #58a6ff; }
            QPushButton {
                min-height: 30px;
                padding: 3px 10px;
                border-radius: 7px;
                border: 1px solid #35598f;
                background: #1a2b48;
                color: #eff5ff;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #234172; }
            QPushButton:disabled {
                color: #9eb2d7;
                border: 1px solid #2f405f;
                background: #182033;
            }
            QCheckBox { font-size: 12px; color: #cfe1ff; }
            QCheckBox::indicator { width: 14px; height: 14px; }
            #AvatarFrame {
                border: 1px solid #2d4b7a;
                border-radius: 12px;
                background: #121c30;
            }
            """
        )

    def _on_web_loaded(self, ok: bool) -> None:
        self._web_ready = bool(ok)
        self._render_now()
        if ok and not (ROOT / "assets" / "mathjax" / "es5" / "tex-mml-chtml.js").exists():
            self.web.page().runJavaScript("window.AthenaUI && window.AthenaUI.notifyMathjaxMissing();")

    def _status_text(self, state: str) -> str:
        thinking = "on" if self.chk_thinking.isChecked() else "off"
        show_thoughts = "on" if self.chk_show_thoughts.isChecked() else "off"
        return (
            f"{state}  |  thinking={thinking}  show_thoughts={show_thoughts}  |  "
            f"model={self.streamer.model_dir}  temp={self.settings.temperature}  "
            f"top_p={self.settings.top_p}  top_k={self.settings.top_k}  "
            f"max_new_tokens={self.settings.max_new_tokens}"
        )

    def set_status(self, state: str) -> None:
        self.status.setText(self._status_text(state))

    def _set_avatar_state(self, state: str) -> None:
        face = ASCII_AVATAR_BY_STATE.get(state, ASCII_AVATAR_BY_STATE["idle"])
        if self._avatar_state == state and self.avatar_frame.text() == face:
            return
        self._avatar_state = state
        self.avatar_frame.setText(face)

    def _refresh_image_label(self) -> None:
        count = len(self._pending_image_paths)
        if count <= 0:
            self.image_label.setText("Images: none")
            self.image_clear_btn.setEnabled(False)
            self.image_preview.setVisible(False)
            self.image_preview.clear()
            return
        name_list = [p.name for p in self._pending_image_paths[:2]]
        if count > 2:
            name_list.append(f"+{count - 2} more")
        self.image_label.setText(f"Images attached: {count}  ({', '.join(name_list)})")
        self.image_clear_btn.setEnabled(True)
        first = self._pending_image_paths[0]
        pix = QPixmap(str(first))
        if pix.isNull():
            self.image_preview.setText(f"Preview unavailable: {first.name}")
            self.image_preview.setPixmap(QPixmap())
        else:
            scaled = pix.scaled(220, 112, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_preview.setPixmap(scaled)
        self.image_preview.setVisible(True)

    def _persist_pasted_image(self, item: object) -> Optional[Path]:
        try:
            if isinstance(item, str):
                src = Path(item).expanduser().resolve()
                if src.is_file():
                    return src
                return None
            if isinstance(item, QImage):
                if item.isNull():
                    return None
                out = self._image_stage_dir / f"clip_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.png"
                if item.save(str(out), "PNG"):
                    return out
                return None
        except Exception:
            return None
        return None

    def _on_images_pasted(self, items: object) -> None:
        if not isinstance(items, list):
            return
        added = 0
        for item in items:
            persisted = self._persist_pasted_image(item)
            if persisted is None:
                continue
            self._pending_image_paths.append(persisted)
            added += 1
        if added > 0:
            self._refresh_image_label()
            if self.streamer.supports_vision and self.streamer.processor is None:
                self.set_status(
                    f"Attached {added} image(s), but vision backend is unavailable. "
                    "Check torchvision and restart."
                )
            else:
                self.set_status(f"Attached {added} image(s).")

    def _cleanup_temp_images(self, paths: list[Path]) -> None:
        for path in paths:
            try:
                if path.parent.resolve() == self._image_stage_dir and path.exists():
                    path.unlink(missing_ok=True)
            except Exception:
                pass

    def _clear_pending_images(self, *, remove_files: bool = True) -> None:
        paths = list(self._pending_image_paths)
        self._pending_image_paths.clear()
        if remove_files:
            self._cleanup_temp_images(paths)
        self._refresh_image_label()

    def _toggle_thinking(self, _state: int) -> None:
        self.settings.enable_thinking = bool(self.chk_thinking.isChecked())
        try:
            self.settings.save(CONFIG_PATH)
        except Exception:
            pass
        self.set_status("Ready")

    def _toggle_show_thoughts(self, _state: int) -> None:
        self.settings.hide_thoughts = not bool(self.chk_show_thoughts.isChecked())
        try:
            self.settings.save(CONFIG_PATH)
        except Exception:
            pass
        self.set_status("Ready")

    def _render_now(self) -> None:
        if not self._web_ready:
            return
        body_html = render_transcript_html(self.transcript)
        if body_html == self._last_full_html:
            return
        payload = json.dumps(body_html)
        self.web.page().runJavaScript(f"window.AthenaUI && window.AthenaUI.setTranscriptHtml({payload});")
        self._last_full_html = body_html

    def _render_latest_assistant_body(self, force_typeset: bool = False) -> None:
        if not self._web_ready or self._current_assistant_idx is None:
            return
        if self._current_assistant_idx >= len(self.transcript):
            return
        content = self.transcript[self._current_assistant_idx].get("content") or ""
        body_html = render_message_body_html(content)
        if body_html == self._last_assistant_body_html and not force_typeset:
            return
        payload = json.dumps(body_html)
        force = "true" if force_typeset else "false"
        fallback_full = json.dumps(render_transcript_html(self.transcript))
        self.web.page().runJavaScript(
            "if (window.AthenaUI && typeof window.AthenaUI.updateLatestAssistantBody === 'function') {"
            f"  window.AthenaUI.updateLatestAssistantBody({payload}, {force});"
            "} else if (window.AthenaUI && typeof window.AthenaUI.setTranscriptHtml === 'function') {"
            f"  window.AthenaUI.setTranscriptHtml({fallback_full});"
            "}"
        )
        self._last_assistant_body_html = body_html

    def _schedule_render(self) -> None:
        if not self.render_timer.isActive():
            self.render_timer.start(self.render_throttle_ms)

    def _flush_render(self) -> None:
        if self.is_streaming and self._current_assistant_idx is not None:
            self._render_latest_assistant_body(force_typeset=False)
            return
        self._render_now()

    def _drain_stream_chars(self) -> None:
        if not self.is_streaming or self._current_assistant_idx is None:
            self.char_timer.stop()
            return
        if self._current_assistant_idx >= len(self.transcript):
            self.char_timer.stop()
            return
        if not self._stream_char_queue:
            self.char_timer.stop()
            self._maybe_finalize_stream()
            return

        next_char = self._stream_char_queue.popleft()
        self.transcript[self._current_assistant_idx]["content"] += next_char
        self._render_latest_assistant_body(force_typeset=False)

        if not self._stream_char_queue:
            self._maybe_finalize_stream()

    def _maybe_finalize_stream(self) -> None:
        if self._pending_finish_payload is None:
            return
        if self._stream_char_queue:
            return
        cleaned_user_text, assistant_text = self._pending_finish_payload
        self._pending_finish_payload = None
        self._finalize_stream(cleaned_user_text, assistant_text)

    def build_prompt(self, user_text: str, image_paths: Optional[list[Path]] = None) -> tuple[str, str]:
        cleaned_user_text = (user_text or "").strip()
        vision_inputs = image_paths if (image_paths and self.streamer.supports_vision and self.streamer.processor is not None) else None
        prompt = wrap.build_prompt(
            self.streamer.tokenizer,
            self.history,
            cleaned_user_text,
            system_prompt=self.system_prompt,
            max_turns=6,
            enable_thinking=self.chk_thinking.isChecked(),
            user_images=vision_inputs,
        )
        return prompt, cleaned_user_text

    def send(self) -> None:
        if self.is_streaming:
            return
        user_text = self.entry.toPlainText().strip()
        image_paths = list(self._pending_image_paths)
        if not user_text and not image_paths:
            return
        self.entry.clear()
        # Keep staged files alive for inference; worker cleans them up after streaming.
        self._clear_pending_images(remove_files=False)

        display_user = user_text if user_text else "[image input]"
        if image_paths:
            image_md_lines: list[str] = []
            for idx, path in enumerate(image_paths, start=1):
                try:
                    image_uri = path.resolve().as_uri()
                    image_md_lines.append(f"![attached image {idx}]({image_uri})")
                except Exception:
                    image_md_lines.append(f"[image {idx}: {path.name}]")
                try:
                    if path.parent.resolve() == self._image_stage_dir:
                        self._session_temp_image_paths.append(path)
                except Exception:
                    pass
            display_user = f"{display_user}\n\n" + "\n\n".join(image_md_lines)

        log_user = user_text if user_text else "[image input]"
        if image_paths:
            log_user = f"{log_user}\n[attached images: {len(image_paths)}]"

        append_log(RAW_LOG, "USER", log_user)
        self.transcript.append({"role": "user", "content": display_user})
        self.transcript.append({"role": "assistant", "content": ""})
        self._current_assistant_idx = len(self.transcript) - 1
        self._stop_requested = False
        self._stream_had_error = False
        self._last_stream_error_message = ""
        self._last_assistant_body_html = ""
        self._stream_char_queue.clear()
        self._pending_finish_payload = None
        self._render_now()

        self.is_streaming = True
        self.btn_send.setEnabled(False)
        self.chk_thinking.setEnabled(False)
        self.chk_show_thoughts.setEnabled(False)
        if self.settings.hide_thoughts and self.chk_thinking.isChecked():
            self._set_avatar_state("thinking")
        else:
            self._set_avatar_state("speaking")
        if self.settings.hide_thoughts and self.chk_thinking.isChecked():
            self.set_status("Thinking... (hidden)")
        elif image_paths and (not self.streamer.supports_vision or self.streamer.processor is None):
            self.set_status("Streaming... (image ignored: vision backend unavailable)")
        else:
            self.set_status("Streaming...")
        append_ui_event(
            "stream_start",
            mode="qt-web",
            model_dir=str(self.streamer.model_dir),
            details={
                "input_chars": len(user_text),
                "images_attached": len(image_paths),
                "thinking_enabled": bool(self.chk_thinking.isChecked()),
                "show_thoughts": bool(self.chk_show_thoughts.isChecked()),
            },
        )

        threading.Thread(target=self._stream_worker, args=(user_text, image_paths), daemon=True).start()

    def _stream_worker(self, user_text: str, image_paths: list[Path]) -> None:
        assistant_chunks: list[str] = []
        think_stripper = wrap.ThinkStripper(enabled=not self.chk_show_thoughts.isChecked())
        try:
            use_multimodal_path = bool(image_paths and self.streamer.supports_vision and self.streamer.processor is not None)
            prompt, cleaned_user_text = self.build_prompt(user_text, image_paths)
            if not cleaned_user_text and image_paths:
                cleaned_user_text = "[image input]"

            def on_chunk(chunk: str) -> None:
                visible = think_stripper.feed(chunk)
                if not visible:
                    return
                assistant_chunks.append(visible)
                self.signals.chunk.emit(visible)

            if use_multimodal_path:
                messages = wrap.build_messages_from_history(
                    self.history,
                    user_text,
                    system_prompt=self.system_prompt,
                    max_turns=6,
                    user_images=image_paths,
                )
                self.streamer.stream_messages(
                    messages,
                    on_chunk,
                    enable_thinking=self.chk_thinking.isChecked(),
                )
            else:
                self.streamer.stream(prompt, on_chunk)
            tail = think_stripper.flush()
            if tail:
                assistant_chunks.append(tail)
                self.signals.chunk.emit(tail)
            assistant_text = wrap.clean_assistant_text("".join(assistant_chunks))
            self.signals.finished.emit(cleaned_user_text, assistant_text)
        except Exception as exc:
            msg = str(exc).strip() or f"{exc.__class__.__name__}: {repr(exc)}"
            self.signals.error.emit(msg)
            cleaned_fallback = user_text.strip()
            assistant_text = wrap.clean_assistant_text("".join(assistant_chunks))
            self.signals.finished.emit(cleaned_fallback, assistant_text)

    def _on_stream_chunk(self, chunk: str) -> None:
        if self._current_assistant_idx is None:
            return
        if self._current_assistant_idx >= len(self.transcript):
            return
        if "?" in chunk:
            self._set_avatar_state("confused")
        elif "!" in chunk:
            self._set_avatar_state("happy")
        else:
            self._set_avatar_state("speaking")
        for ch in chunk:
            self._stream_char_queue.append(ch)
        if not self.char_timer.isActive():
            self.char_timer.start()

    def _on_stream_error(self, message: str) -> None:
        if not self.is_streaming and self._current_assistant_idx is None:
            return
        self._set_avatar_state("error")
        self._stream_had_error = True
        self._last_stream_error_message = message or "stream failed"
        append_ui_event(
            "stream_error",
            mode="qt-web",
            model_dir=str(self.streamer.model_dir),
            details={"message": message},
        )
        if self._current_assistant_idx is None:
            self.transcript.append({"role": "assistant", "content": f"[stream error] {message}"})
            self._schedule_render()
            self.set_status("Stream error")
            return
        if self._current_assistant_idx < len(self.transcript):
            existing = self.transcript[self._current_assistant_idx]["content"]
            if existing.strip():
                self.transcript[self._current_assistant_idx]["content"] = existing.rstrip() + f"\n[stream error] {message}"
            else:
                self.transcript[self._current_assistant_idx]["content"] = f"[stream error] {message}"
            self._schedule_render()
            self.set_status("Stream error (partial)")

    def _finalize_stream(self, cleaned_user_text: str, assistant_text: str, *, final_state: str = "Ready") -> None:
        if self._current_assistant_idx is not None and self._current_assistant_idx < len(self.transcript):
            existing = self.transcript[self._current_assistant_idx].get("content", "")
            if assistant_text or (not self._stream_had_error):
                self.transcript[self._current_assistant_idx]["content"] = assistant_text
            elif not existing.strip():
                msg = self._last_stream_error_message or "stream failed"
                self.transcript[self._current_assistant_idx]["content"] = f"[stream error] {msg}"
        if assistant_text:
            self.history.append((cleaned_user_text, assistant_text))
            if len(self.history) > 12:
                self.history = self.history[-12:]
        append_log(RAW_LOG, "ASSISTANT", assistant_text)
        append_log(CLEAN_LOG, "ASSISTANT", assistant_text)
        append_ui_event(
            "stream_stop",
            mode="qt-web",
            model_dir=str(self.streamer.model_dir),
            details={
                "reason": "user_stop" if self._stop_requested else "completed",
                "output_chars": len(assistant_text),
            },
        )

        self._render_latest_assistant_body(force_typeset=True)
        self._current_assistant_idx = None
        self._stop_requested = False
        self._stream_had_error = False
        self._last_stream_error_message = ""
        self.is_streaming = False
        self.btn_send.setEnabled(True)
        self.chk_thinking.setEnabled(True)
        self.chk_show_thoughts.setEnabled(True)
        self.entry.setFocus()
        if final_state == "Stopped":
            self._set_avatar_state("stopped")
        else:
            self._set_avatar_state("idle")
        self.set_status(final_state)

    def _on_stream_finished(self, cleaned_user_text: str, assistant_text: str) -> None:
        if not self.is_streaming or self._current_assistant_idx is None:
            return
        self._pending_finish_payload = (cleaned_user_text, assistant_text)
        self._maybe_finalize_stream()

    def stop(self) -> None:
        if not self.is_streaming:
            self.set_status("Ready")
            return
        self._stop_requested = True
        self._set_avatar_state("stopped")
        append_ui_event(
            "stream_stop",
            mode="qt-web",
            model_dir=str(self.streamer.model_dir),
            details={"reason": "stop_requested"},
        )
        self.streamer.stop()
        self.char_timer.stop()

        # Flush any queued visible characters immediately so the user keeps partial output.
        if self._current_assistant_idx is not None and self._current_assistant_idx < len(self.transcript):
            if self._stream_char_queue:
                self.transcript[self._current_assistant_idx]["content"] += "".join(self._stream_char_queue)
        self._stream_char_queue.clear()
        self._pending_finish_payload = None

        # Finalize immediately so Stop is deterministic even if worker thread lingers.
        cleaned_user_text = ""
        for msg in reversed(self.transcript):
            if msg.get("role") == "user":
                cleaned_user_text = (msg.get("content") or "").strip()
                break
        assistant_text = ""
        if self._current_assistant_idx is not None and self._current_assistant_idx < len(self.transcript):
            assistant_text = self.transcript[self._current_assistant_idx].get("content", "")
        self._finalize_stream(cleaned_user_text, assistant_text, final_state="Stopped")

    def clear(self) -> None:
        if self.is_streaming:
            self.streamer.stop()
        self.char_timer.stop()
        if self._session_temp_image_paths:
            self._cleanup_temp_images(self._session_temp_image_paths)
            self._session_temp_image_paths.clear()
        self._clear_pending_images()
        self.history.clear()
        self.transcript = [
            {"role": "system", "content": f"Loaded model: {self.streamer.model_dir}"},
            {"role": "system", "content": format_model_proof_text(self.streamer.model_proof)},
            {"role": "system", "content": format_runtime_config_text(self._loaded_runtime_cfg)},
        ]
        self._current_assistant_idx = None
        self._last_assistant_body_html = ""
        self._last_full_html = ""
        self._stream_char_queue.clear()
        self._pending_finish_payload = None
        self._set_avatar_state("idle")
        self._render_now()
        self.entry.clear()
        self.entry.setFocus()
        self.set_status("Ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="", help="Optional model path override for this UI instance.")
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else None

    app = QApplication(sys.argv)
    ui = AthenaQtUI(model_dir=model_dir)
    ui.show()
    try:
        raise SystemExit(app.exec())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

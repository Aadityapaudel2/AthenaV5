#!/usr/bin/env python3
from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from athena_paths import get_default_chat_model_dir
import wrap
from qt_render import render_transcript_html

try:
    from authlib.integrations.starlette_client import OAuth
except Exception:  # pragma: no cover - handled at startup
    OAuth = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
PORTAL_DIR = ROOT / "portal"
TEMPLATES_DIR = PORTAL_DIR / "templates"
STATIC_DIR = PORTAL_DIR / "static"
DEFAULT_LOG_ROOT = ROOT / "data" / "users"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_user_key(email: str) -> str:
    safe = re.sub(r"[^a-z0-9._@-]+", "_", (email or "anonymous").lower())
    return safe.strip("_") or "anonymous"


def _sanitize_user_history_for_model(content: str) -> str:
    text = content or ""
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", "", text)
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        lower = line.lower()
        if lower.startswith("[attached image"):
            continue
        if lower.startswith("[attached images"):
            continue
        if lower.startswith("(file:///"):
            continue
        lines.append(raw)
    clean = "\n".join(lines).strip()
    return clean or "Image attached."


@dataclass(frozen=True)
class PortalConfig:
    host: str
    port: int
    path_prefix: str
    load_model: bool
    auth_required: bool
    google_client_id: str
    google_client_secret: str
    auth_redirect_uri: str
    session_secret: str
    log_root: Path
    cookie_secure: bool
    log_deltas: bool

    @staticmethod
    def load() -> "PortalConfig":
        raw_prefix = os.getenv("ATHENA_PORTAL_PATH_PREFIX", "/AthenaV5").strip() or "/AthenaV5"
        path_prefix = raw_prefix if raw_prefix.startswith("/") else f"/{raw_prefix}"
        path_prefix = path_prefix.rstrip("/") or "/AthenaV5"
        return PortalConfig(
            host=os.getenv("ATHENA_PORTAL_HOST", "0.0.0.0"),
            port=int(os.getenv("ATHENA_PORTAL_PORT") or os.getenv("PORT") or "8000"),
            path_prefix=path_prefix,
            load_model=_env_bool("ATHENA_WEB_LOAD_MODEL", False),
            auth_required=_env_bool("ATHENA_AUTH_REQUIRED", True),
            google_client_id=(os.getenv("ATHENA_GOOGLE_CLIENT_ID") or "").strip(),
            google_client_secret=(os.getenv("ATHENA_GOOGLE_CLIENT_SECRET") or "").strip(),
            auth_redirect_uri=(
                os.getenv("ATHENA_AUTH_REDIRECT_URI")
                or "https://portal.neohmlabs.com/AthenaV5/auth/callback"
            ).strip(),
            session_secret=(os.getenv("ATHENA_PORTAL_SESSION_SECRET") or "").strip(),
            log_root=Path(os.getenv("ATHENA_LOG_ROOT") or str(DEFAULT_LOG_ROOT)).expanduser().resolve(),
            cookie_secure=_env_bool("ATHENA_PORTAL_COOKIE_SECURE", True),
            log_deltas=_env_bool("ATHENA_LOG_DELTAS", False),
        )


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=20000)


class ChatImage(BaseModel):
    name: str = Field(default="image.png", max_length=256)
    content_type: str = Field(default="image/png", max_length=128)
    data_url: str = Field(min_length=1, max_length=12_000_000)


class ChatRequest(BaseModel):
    prompt: str = Field(default="", max_length=12000)
    history: list[ChatMessage] = Field(default_factory=list)
    images: list[ChatImage] = Field(default_factory=list, max_length=6)


class ChatResponse(BaseModel):
    assistant: str
    history: list[ChatMessage]
    transcript_html: str
    smoke_mode: bool
    model_loaded: bool


def _chat_msg_dict(msg: ChatMessage) -> dict[str, Any]:
    if hasattr(msg, "model_dump"):
        return msg.model_dump()  # type: ignore[attr-defined]
    return msg.dict()


class UserLogStore:
    def __init__(self, root: Path):
        self.root = root
        self._lock = Lock()

    def user_key(self, email: str) -> str:
        return _normalize_user_key(email)

    def _session_file(self, email: str) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        user_dir = self.root / self.user_key(email)
        return user_dir / "sessions" / f"{day}.ndjson"

    def _error_file(self, email: str) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        user_dir = self.root / self.user_key(email)
        return user_dir / "errors" / f"{day}.ndjson"

    def ensure_profile(self, user: dict[str, Any]) -> None:
        email = str(user.get("email") or "anonymous@local")
        user_dir = self.root / self.user_key(email)
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (user_dir / "errors").mkdir(parents=True, exist_ok=True)
        profile_path = user_dir / "profile.json"
        if profile_path.exists():
            return
        profile = {
            "email": user.get("email"),
            "name": user.get("name"),
            "picture": user.get("picture"),
            "sub": user.get("sub"),
            "created_at_utc": _utc_now_iso(),
        }
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    def log_event(self, user_email: str, event: dict[str, Any], error_log: bool = False) -> None:
        try:
            with self._lock:
                target = self._error_file(user_email) if error_log else self._session_file(user_email)
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(event)
                payload.setdefault("ts_utc", _utc_now_iso())
                line = json.dumps(payload, ensure_ascii=False)
                with target.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as exc:  # pragma: no cover
            print(f"[portal-log] failed to write event: {exc}")


class ChatEngine:
    def __init__(self, cfg: PortalConfig):
        self._cfg = cfg
        self._lock = Lock()
        self._streamer: Any | None = None
        self._model_load_error: str = ""
        self._system_prompt = wrap.load_system_prompt()
        self._configured_model_dir = str(get_default_chat_model_dir())
        self._startup_model_warmed = False

    @property
    def smoke_mode(self) -> bool:
        return not self._cfg.load_model

    @property
    def model_loaded(self) -> bool:
        return self._streamer is not None

    @property
    def model_load_error(self) -> str:
        return self._model_load_error

    @property
    def startup_model_warmed(self) -> bool:
        return self._startup_model_warmed

    @property
    def configured_model_dir(self) -> str:
        return self._configured_model_dir

    @property
    def configured_model_label(self) -> str:
        return Path(self._configured_model_dir).name

    @property
    def active_model_dir(self) -> str:
        if self._streamer is None:
            return ""
        return str(self._streamer.model_dir)

    @property
    def active_model_label(self) -> str:
        if self._streamer is None:
            return ""
        return str(self._streamer.model_label)

    def _ensure_loaded(self) -> None:
        if self._streamer is not None:
            return
        try:
            from tk_chat import CONFIG_PATH, GuiSettings, LocalStreamer

            settings = GuiSettings.load(CONFIG_PATH)
            self._streamer = LocalStreamer(settings)
        except Exception as exc:
            self._model_load_error = str(exc)
            raise

    def warm_start(self) -> None:
        if self.smoke_mode:
            return
        self._ensure_loaded()
        self._startup_model_warmed = True

    def _history_to_turns(self, history: list[ChatMessage]) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        pending_user: str | None = None
        for msg in history[-24:]:
            role = msg.role
            content = msg.content.strip()
            if not content:
                continue
            if role == "user":
                pending_user = _sanitize_user_history_for_model(content)
                continue
            if role == "assistant" and pending_user is not None:
                turns.append((pending_user, content))
                pending_user = None
        return turns[-12:]

    def _build_prompt(self, prompt: str, history: list[ChatMessage]) -> str:
        clean_prompt = prompt.strip()
        turns = self._history_to_turns(history)
        return wrap.build_prompt(
            self._streamer.tokenizer,  # type: ignore[union-attr]
            turns,
            clean_prompt,
            system_prompt=self._system_prompt,
            max_turns=6,
            enable_thinking=False,
        )

    def _build_messages(self, prompt: str, history: list[ChatMessage], image_paths: list[str]) -> list[dict[str, Any]]:
        turns = self._history_to_turns(history)
        return wrap.build_messages_from_history(
            turns,
            prompt.strip() or "Describe this image.",
            system_prompt=self._system_prompt,
            max_turns=6,
            user_images=image_paths or None,
        )

    def reply(self, prompt: str, history: list[ChatMessage], image_paths: list[str] | None = None) -> str:
        clean_prompt = prompt.strip()
        use_images = list(image_paths or [])
        if not clean_prompt and not use_images:
            return "Please enter a prompt."
        if self.smoke_mode:
            return (
                "Smoke mode is active. Portal routing and UI are live, but model loading is disabled. "
                "Set ATHENA_WEB_LOAD_MODEL=1 to enable live inference."
            )
        with self._lock:
            self._ensure_loaded()
            think_stripper = wrap.ThinkStripper(enabled=True)
            chunks: list[str] = []

            def on_chunk(chunk: str) -> None:
                visible = think_stripper.feed(chunk)
                if visible:
                    chunks.append(visible)

            if use_images:
                messages = self._build_messages(clean_prompt, history, use_images)
                self._streamer.stream_messages(messages, on_chunk, enable_thinking=False)  # type: ignore[union-attr]
            else:
                self._streamer.stream(self._build_prompt(clean_prompt, history), on_chunk)  # type: ignore[union-attr]
            tail = think_stripper.flush()
            if tail:
                chunks.append(tail)
            return wrap.clean_assistant_text("".join(chunks))

    def stream_reply(
        self,
        prompt: str,
        history: list[ChatMessage],
        on_delta: Any,
        image_paths: list[str] | None = None,
    ) -> str:
        clean_prompt = prompt.strip()
        use_images = list(image_paths or [])
        if not clean_prompt and not use_images:
            return "Please enter a prompt."
        if self.smoke_mode:
            smoke = (
                "Smoke mode is active. Portal routing and UI are live, but model loading is disabled. "
                "Set ATHENA_WEB_LOAD_MODEL=1 to enable live inference."
            )
            on_delta(smoke)
            return smoke
        with self._lock:
            self._ensure_loaded()
            think_stripper = wrap.ThinkStripper(enabled=True)
            chunks: list[str] = []

            def on_chunk(chunk: str) -> None:
                visible = think_stripper.feed(chunk)
                if visible:
                    chunks.append(visible)
                    on_delta(visible)

            if use_images:
                messages = self._build_messages(clean_prompt, history, use_images)
                self._streamer.stream_messages(messages, on_chunk, enable_thinking=False)  # type: ignore[union-attr]
            else:
                self._streamer.stream(self._build_prompt(clean_prompt, history), on_chunk)  # type: ignore[union-attr]
            tail = think_stripper.flush()
            if tail:
                chunks.append(tail)
                on_delta(tail)
            return wrap.clean_assistant_text("".join(chunks))


def _build_initial_system_messages(cfg: PortalConfig, engine: ChatEngine) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"Athena V5 portal online at {cfg.path_prefix}. "
                f"Smoke mode={'on' if engine.smoke_mode else 'off'}."
            ),
        }
    ]


def _client_meta(request: Request) -> dict[str, str]:
    return {
        "client_ip": request.client.host if request.client else "",
        "user_agent": request.headers.get("user-agent", ""),
    }


def _session_user(request: Request) -> dict[str, Any] | None:
    try:
        raw = request.session.get("user")
    except AssertionError:
        return None
    return raw if isinstance(raw, dict) else None


def _decode_data_url_image(data_url: str) -> tuple[bytes, str]:
    match = re.match(r"^data:([a-zA-Z0-9.+/-]+);base64,(.+)$", data_url.strip(), re.DOTALL)
    if not match:
        raise ValueError("Invalid image data URL.")
    mime = match.group(1).strip().lower()
    payload = re.sub(r"\s+", "", match.group(2))
    try:
        blob = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 image payload.") from exc
    if not blob:
        raise ValueError("Empty image payload.")
    return blob, mime


def _image_ext_from_mime(mime: str, fallback_name: str) -> str:
    ext = mimetypes.guess_extension(mime) or ""
    if not ext and "." in fallback_name:
        ext = "." + fallback_name.rsplit(".", 1)[-1].lower()
    if not ext:
        ext = ".png"
    return ext


def _persist_request_images(
    payload_images: list[ChatImage],
    *,
    user_email: str,
    request_id: str,
) -> tuple[list[str], list[str]]:
    if not payload_images:
        return [], []
    if len(payload_images) > 6:
        raise ValueError("Too many images in one request (max 6).")

    user_key = logs.user_key(user_email)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    image_dir = cfg.log_root / user_key / "uploads" / day
    image_dir.mkdir(parents=True, exist_ok=True)

    model_paths: list[str] = []
    portal_urls: list[str] = []
    for idx, item in enumerate(payload_images, start=1):
        blob, mime = _decode_data_url_image(item.data_url)
        if len(blob) > 8 * 1024 * 1024:
            raise ValueError("Image exceeds 8MB limit.")
        if not mime.startswith("image/"):
            raise ValueError("Only image uploads are supported.")
        ext = _image_ext_from_mime(mime, item.name or "")
        fname = f"{request_id}_{idx:02d}{ext}"
        out_path = image_dir / fname
        out_path.write_bytes(blob)
        model_paths.append(str(out_path))
        rel = out_path.relative_to(cfg.log_root).as_posix()
        portal_urls.append(f"{cfg.path_prefix}/api/uploads/{rel}")
    return model_paths, portal_urls


def _format_user_message_content(prompt: str, image_urls: list[str]) -> str:
    clean_prompt = prompt.strip()
    parts: list[str] = []
    if clean_prompt:
        parts.append(clean_prompt)
    if image_urls:
        marker = f"[attached image {len(image_urls)}]" if len(image_urls) == 1 else f"[attached images: {len(image_urls)}]"
        parts.append(marker)
        for i, url in enumerate(image_urls, start=1):
            parts.append(f"![attached image {i}]({url})")
    if not parts:
        return "Image attached."
    return "\n\n".join(parts)


cfg = PortalConfig.load()
engine = ChatEngine(cfg)
logs = UserLogStore(cfg.log_root)
oauth: Any | None = None

app = FastAPI(title="Athena V5 Portal", version="2.0.0")
app.add_middleware(
    SessionMiddleware,
    secret_key=cfg.session_secret or "athena-insecure-dev-session-secret",
    same_site="lax",
    https_only=cfg.cookie_secure,
    session_cookie="athena_portal_session",
)
app.mount(f"{cfg.path_prefix}/static", StaticFiles(directory=str(STATIC_DIR)), name="portal-static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
async def on_startup() -> None:
    global oauth

    cfg.log_root.mkdir(parents=True, exist_ok=True)
    probe = cfg.log_root / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)

    if cfg.auth_required:
        missing = []
        if not cfg.google_client_id:
            missing.append("ATHENA_GOOGLE_CLIENT_ID")
        if not cfg.google_client_secret:
            missing.append("ATHENA_GOOGLE_CLIENT_SECRET")
        if not cfg.auth_redirect_uri:
            missing.append("ATHENA_AUTH_REDIRECT_URI")
        if not cfg.session_secret:
            missing.append("ATHENA_PORTAL_SESSION_SECRET")
        if missing:
            raise RuntimeError(f"Missing required auth env vars: {', '.join(missing)}")
        if OAuth is None:
            raise RuntimeError("Auth is required, but authlib is not installed. Add authlib to requirements.")
        oauth = OAuth()
        oauth.register(
            name="google",
            client_id=cfg.google_client_id,
            client_secret=cfg.google_client_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )

    if cfg.load_model:
        engine.warm_start()
    print(
        "[portal-startup] "
        f"auth_required={cfg.auth_required} "
        f"path_prefix={cfg.path_prefix} "
        f"log_root={cfg.log_root} "
        f"log_deltas={cfg.log_deltas} "
        f"model_dir={engine.configured_model_dir} "
        f"model_warmed={engine.startup_model_warmed}"
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "path_prefix": cfg.path_prefix,
        "smoke_mode": engine.smoke_mode,
        "model_loaded": engine.model_loaded,
        "startup_model_warmed": engine.startup_model_warmed,
        "configured_model_label": engine.configured_model_label,
        "configured_model_dir": engine.configured_model_dir,
        "active_model_label": engine.active_model_label,
        "active_model_dir": engine.active_model_dir,
        "auth_required": cfg.auth_required,
        "auth_configured": bool(cfg.google_client_id and cfg.google_client_secret and cfg.session_secret),
        "log_root": str(cfg.log_root),
        "log_deltas": cfg.log_deltas,
    }


@app.get("/", include_in_schema=False)
def root_redirect(request: Request) -> RedirectResponse:
    if cfg.auth_required and _session_user(request) is None:
        return RedirectResponse(url=f"{cfg.path_prefix}/login")
    return RedirectResponse(url=cfg.path_prefix)


@app.get(f"{cfg.path_prefix}/login", response_class=HTMLResponse)
def login_page(request: Request) -> Any:
    if not cfg.auth_required:
        return RedirectResponse(url=cfg.path_prefix)
    if _session_user(request):
        return RedirectResponse(url=cfg.path_prefix)
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "path_prefix": cfg.path_prefix,
            "title": "Athena V5 Login",
        },
    )


@app.get(f"{cfg.path_prefix}/auth/login")
async def auth_login(request: Request) -> Any:
    if not cfg.auth_required:
        return RedirectResponse(url=cfg.path_prefix)
    if _session_user(request):
        return RedirectResponse(url=cfg.path_prefix)
    if oauth is None:
        raise HTTPException(status_code=500, detail="OAuth is not initialized.")
    return await oauth.google.authorize_redirect(request, cfg.auth_redirect_uri)


@app.get(f"{cfg.path_prefix}/auth/callback")
async def auth_callback(request: Request) -> Any:
    if oauth is None:
        raise HTTPException(status_code=500, detail="OAuth is not initialized.")
    try:
        token = await oauth.google.authorize_access_token(request)
        userinfo = token.get("userinfo")
        if not userinfo:
            userinfo = await oauth.google.parse_id_token(request, token)
        user = {
            "sub": str((userinfo or {}).get("sub") or ""),
            "email": str((userinfo or {}).get("email") or ""),
            "name": str((userinfo or {}).get("name") or ""),
            "picture": str((userinfo or {}).get("picture") or ""),
            "issued_at": _utc_now_iso(),
        }
        if not user["email"]:
            raise ValueError("Google account did not return email.")
        # Current policy: allow any Google account. Domain restrictions can be added later.
        request.session["user"] = user
        logs.ensure_profile(user)
        logs.log_event(user["email"], {"event_type": "auth_login", "user_email": user["email"]})
        return RedirectResponse(url=cfg.path_prefix)
    except Exception as exc:
        return HTMLResponse(
            f"<h3>Google login failed</h3><pre>{str(exc)}</pre>",
            status_code=400,
        )


@app.post(f"{cfg.path_prefix}/auth/logout")
def auth_logout(request: Request) -> dict[str, Any]:
    user = _session_user(request)
    if user and user.get("email"):
        logs.log_event(str(user["email"]), {"event_type": "auth_logout", "user_email": str(user["email"])})
    request.session.clear()
    return {"ok": True}


@app.get(cfg.path_prefix, response_class=HTMLResponse)
def portal_index(request: Request) -> HTMLResponse:
    if cfg.auth_required and _session_user(request) is None:
        return RedirectResponse(url=f"{cfg.path_prefix}/login")
    initial_transcript_html = render_transcript_html(_build_initial_system_messages(cfg, engine))
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "path_prefix": cfg.path_prefix,
            "title": "Athena V5 Portal",
            "initial_transcript_html": initial_transcript_html,
        },
    )


@app.get(f"{cfg.path_prefix}/api/me")
def api_me(request: Request) -> dict[str, Any]:
    if not cfg.auth_required:
        return {"user": {"email": "anonymous@local", "name": "Anonymous", "sub": "", "picture": ""}}
    user = _session_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return {"user": user}


@app.get(f"{cfg.path_prefix}/api/config")
def api_config(request: Request) -> dict[str, Any]:
    if cfg.auth_required and _session_user(request) is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return {
        "path_prefix": cfg.path_prefix,
        "smoke_mode": engine.smoke_mode,
        "model_loaded": engine.model_loaded,
        "startup_model_warmed": engine.startup_model_warmed,
        "model_load_error": engine.model_load_error,
        "configured_model_label": engine.configured_model_label,
        "configured_model_dir": engine.configured_model_dir,
        "active_model_label": engine.active_model_label,
        "active_model_dir": engine.active_model_dir,
        "auth_required": cfg.auth_required,
        "log_deltas": cfg.log_deltas,
    }


@app.get(f"{cfg.path_prefix}/api/uploads/{{relative_path:path}}")
def api_upload_file(relative_path: str, request: Request) -> FileResponse:
    if cfg.auth_required and _session_user(request) is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    rel = relative_path.strip().lstrip("/")
    if not rel:
        raise HTTPException(status_code=404, detail="File not found.")
    if ".." in rel.replace("\\", "/"):
        raise HTTPException(status_code=400, detail="Invalid path.")
    target = (cfg.log_root / rel).resolve()
    root_resolved = cfg.log_root.resolve()
    if not str(target).startswith(str(root_resolved)):
        raise HTTPException(status_code=403, detail="Forbidden.")
    if cfg.auth_required:
        user = _session_user(request) or {}
        user_key = logs.user_key(str(user.get("email") or "anonymous@local"))
        expected_prefix = str((cfg.log_root / user_key).resolve())
        if not str(target).startswith(expected_prefix):
            raise HTTPException(status_code=403, detail="Forbidden.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=str(target))


# Deprecated compatibility endpoint. Prefer /api/chat/stream.
@app.post(f"{cfg.path_prefix}/api/chat", response_model=ChatResponse)
def api_chat(payload: ChatRequest, request: Request) -> ChatResponse:
    if cfg.auth_required and _session_user(request) is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    prompt = payload.prompt.strip()
    if not prompt and not payload.images:
        raise HTTPException(status_code=400, detail="Prompt is empty.")

    user = _session_user(request) or {}
    user_email = str(user.get("email") or "anonymous@local")
    req_id = str(uuid4())
    meta = _client_meta(request)
    t0 = perf_counter()
    try:
        model_image_paths, image_urls = _persist_request_images(payload.images, user_email=user_email, request_id=req_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    user_content = _format_user_message_content(prompt, image_urls)
    logs.log_event(
        user_email,
        {
            "event_type": "request_start",
            "request_id": req_id,
            "user_email": user_email,
            "path_prefix": cfg.path_prefix,
            "prompt": user_content,
            "image_count": len(model_image_paths),
            "model_label": engine.active_model_label or engine.configured_model_label,
            "model_dir": engine.active_model_dir or engine.configured_model_dir,
            "smoke_mode": engine.smoke_mode,
            **meta,
        },
    )

    try:
        assistant = engine.reply(prompt=prompt, history=payload.history, image_paths=model_image_paths)
        next_history = list(payload.history)
        next_history.append(ChatMessage(role="user", content=user_content))
        next_history.append(ChatMessage(role="assistant", content=assistant))
        transcript_html = render_transcript_html([_chat_msg_dict(item) for item in next_history])
        latency_ms = int((perf_counter() - t0) * 1000)
        logs.log_event(
            user_email,
            {
                "event_type": "request_done",
                "request_id": req_id,
                "user_email": user_email,
                "assistant_final": assistant,
                "latency_ms": latency_ms,
                "image_count": len(model_image_paths),
                "model_label": engine.active_model_label or engine.configured_model_label,
                "model_dir": engine.active_model_dir or engine.configured_model_dir,
                "smoke_mode": engine.smoke_mode,
                **meta,
            },
        )
        return ChatResponse(
            assistant=assistant,
            history=next_history,
            transcript_html=transcript_html,
            smoke_mode=engine.smoke_mode,
            model_loaded=engine.model_loaded,
        )
    except Exception as exc:
        latency_ms = int((perf_counter() - t0) * 1000)
        logs.log_event(
            user_email,
            {
                "event_type": "request_error",
                "request_id": req_id,
                "user_email": user_email,
                "error": str(exc),
                "latency_ms": latency_ms,
                "image_count": len(model_image_paths),
                "model_label": engine.active_model_label or engine.configured_model_label,
                "model_dir": engine.active_model_dir or engine.configured_model_dir,
                **meta,
            },
            error_log=True,
        )
        raise


@app.post(f"{cfg.path_prefix}/api/chat/stream")
def api_chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    if cfg.auth_required and _session_user(request) is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    prompt = payload.prompt.strip()
    if not prompt and not payload.images:
        raise HTTPException(status_code=400, detail="Prompt is empty.")

    user = _session_user(request) or {}
    user_email = str(user.get("email") or "anonymous@local")
    req_id = str(uuid4())
    meta = _client_meta(request)
    t0 = perf_counter()
    try:
        model_image_paths, image_urls = _persist_request_images(payload.images, user_email=user_email, request_id=req_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    user_content = _format_user_message_content(prompt, image_urls)

    logs.log_event(
        user_email,
        {
            "event_type": "request_start",
            "request_id": req_id,
            "user_email": user_email,
            "path_prefix": cfg.path_prefix,
            "prompt": user_content,
            "image_count": len(model_image_paths),
            "model_label": engine.active_model_label or engine.configured_model_label,
            "model_dir": engine.active_model_dir or engine.configured_model_dir,
            "smoke_mode": engine.smoke_mode,
            **meta,
        },
    )

    q: "Queue[dict[str, Any]]" = Queue()

    def worker() -> None:
        try:
            def on_delta(txt: str) -> None:
                q.put({"type": "delta", "text": txt})
                if cfg.log_deltas:
                    logs.log_event(
                        user_email,
                        {
                            "event_type": "delta",
                            "request_id": req_id,
                            "user_email": user_email,
                            "delta": txt,
                            "model_label": engine.active_model_label or engine.configured_model_label,
                            "model_dir": engine.active_model_dir or engine.configured_model_dir,
                            **meta,
                        },
                    )

            assistant = engine.stream_reply(
                prompt=prompt,
                history=payload.history,
                image_paths=model_image_paths,
                on_delta=on_delta,
            )
            next_history = list(payload.history)
            next_history.append(ChatMessage(role="user", content=user_content))
            next_history.append(ChatMessage(role="assistant", content=assistant))
            transcript_html = render_transcript_html([_chat_msg_dict(item) for item in next_history])
            latency_ms = int((perf_counter() - t0) * 1000)
            logs.log_event(
                user_email,
                {
                    "event_type": "request_done",
                    "request_id": req_id,
                    "user_email": user_email,
                    "assistant_final": assistant,
                    "latency_ms": latency_ms,
                    "image_count": len(model_image_paths),
                    "model_label": engine.active_model_label or engine.configured_model_label,
                    "model_dir": engine.active_model_dir or engine.configured_model_dir,
                    "smoke_mode": engine.smoke_mode,
                    **meta,
                },
            )
            q.put(
                {
                    "type": "done",
                    "assistant": assistant,
                    "history": [_chat_msg_dict(item) for item in next_history],
                    "transcript_html": transcript_html,
                    "smoke_mode": engine.smoke_mode,
                    "model_loaded": engine.model_loaded,
                }
            )
        except Exception as exc:
            latency_ms = int((perf_counter() - t0) * 1000)
            logs.log_event(
                user_email,
                {
                    "event_type": "request_error",
                    "request_id": req_id,
                    "user_email": user_email,
                    "error": str(exc),
                    "latency_ms": latency_ms,
                    "image_count": len(model_image_paths),
                    "model_label": engine.active_model_label or engine.configured_model_label,
                    "model_dir": engine.active_model_dir or engine.configured_model_dir,
                    **meta,
                },
                error_log=True,
            )
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put({"type": "eof"})

    Thread(target=worker, daemon=True).start()

    def iter_events() -> Any:
        while True:
            item = q.get()
            if item.get("type") == "eof":
                break
            payload_json = json.dumps(item, ensure_ascii=False)
            yield f"data: {payload_json}\n\n"

    return StreamingResponse(
        iter_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("portal_server:app", host=cfg.host, port=cfg.port, reload=False)

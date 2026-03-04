# Athena V5 Portal

Web runtime for Athena V5 (`/AthenaV5` by default).

## Components
- `portal_server.py` (repo root): FastAPI server, auth, SSE, logging, model bridge.
- `templates/index.html`: portal UI shell.
- `templates/login.html`: Google sign-in page.
- `static/portal.css`: portal styles.
- `static/portal.js`: client streaming + composer logic.

## Current Features
- Google OAuth login for portal routes (when enabled).
- Auth-gated API surface:
  - `/AthenaV5/api/me`
  - `/AthenaV5/api/config`
  - `/AthenaV5/api/chat`
  - `/AthenaV5/api/chat/stream` (SSE)
- Streaming chat with first-token/stall status UX.
- Enter to send, Shift+Enter for newline.
- Image attach support:
  - file picker,
  - clipboard paste in composer,
  - image previews in composer and transcript.
- Per-user local storage under `data/users/<user_key>/...`:
  - `profile.json`
  - NDJSON event logs
  - uploaded image files used for multimodal prompts.

## Route Prefix
- Default: `/AthenaV5`
- Override with `ATHENA_PORTAL_PATH_PREFIX=/your/path`

## Local Start

Smoke mode (no model load):
```powershell
Set-Location D:\AthenaPlayground\AthenaV5
.\run_portal.ps1
```

Live model:
```powershell
Set-Location D:\AthenaPlayground\AthenaV5
.\run_portal.ps1 -LoadModel
```

Health endpoint:
- `http://localhost:8000/healthz`

## Tunnel Start

One-shot launcher (portal + cloudflared):
```powershell
.\run_portal_v5.ps1
```

Direct tunnel script:
```powershell
.\cloudflared_athenav5.ps1
```

## Auth Environment

Required when `ATHENA_AUTH_REQUIRED=1`:
- `ATHENA_GOOGLE_CLIENT_ID`
- `ATHENA_GOOGLE_CLIENT_SECRET`
- `ATHENA_AUTH_REDIRECT_URI`
- `ATHENA_PORTAL_SESSION_SECRET`

Recommended:
- copy `portal_auth.env.example` to `portal_auth.env` in repo root
- fill values once
- `run_portal_v5.ps1` auto-loads this file

Optional:
- `ATHENA_LOG_ROOT`
- `ATHENA_MODEL_DIR`
- `ATHENA_PORTAL_COOKIE_SECURE`
- `ATHENA_LOG_DELTAS` (`0` default; enable only for stream debugging)

## Notes
- Smoke mode defaults to offloading GPU usage until explicitly requested.
- Server emits SSE with `Cache-Control: no-cache` and `X-Accel-Buffering: no`.
- Upload guardrails: max 6 images/request, max 8MB/image.

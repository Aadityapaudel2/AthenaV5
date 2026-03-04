# Athena V5

Athena V5 is a local-first LLM workspace with:
- desktop chat (Qt primary, Tk fallback),
- FastAPI + Cloudflared portal,
- finetuning scripts for iterative SFT runs.

This README is the operational source of truth as of **March 4, 2026**.

## Current State
- Desktop UI is stable on the active `.venv` and supports:
  - streaming output,
  - Markdown + LaTeX rendering,
  - clipboard/file image attach for multimodal-capable models.
- Portal is stable with:
  - Google sign-in,
  - authenticated chat routes,
  - SSE token streaming,
  - per-user raw NDJSON logging on local disk,
  - image upload support in composer (attach button + clipboard paste).
- Finetune pipeline is stable with resumable checkpoints and explicit args-file based launches.

## Core Directories
- `Finetune/`: SFT training scripts, args files, evaluation helpers.
- `portal/`: web UI templates/static assets.
- `assets/`: desktop renderer assets (chat shell, CSS, MathJax bundle).
- `models/`: local base/tuned checkpoints.
- `data/users/`: per-user portal profiles, logs, and uploaded images.

## Canonical Entrypoints
- Desktop UI: `run_ui.ps1`
- Portal local: `run_portal.ps1`
- Portal + Cloudflare tunnel: `run_portal_v5.ps1`
- Finetune orchestrator: `Finetune/run_training.ps1`

## Pivot Log (What Changed and Why)

1. Training stability pivot
- Early SFT runs stalled or crashed during long jobs.
- Recovery path moved to checkpoint resume with tighter save cadence and explicit config snapshots.
- Result: resumable training became routine instead of manual recovery.

2. Dataset quality pivot
- Previous mixed persona/math data caused style drift, repetition, and weak coherence.
- Workflow switched to manual curation and stricter sample control.
- Result: cleaner target behavior, less random persona bleed.

3. Script simplification pivot
- Launch scripts had redundant wrappers and path ambiguity.
- Consolidated to canonical launchers and stricter path resolution.
- Result: one-command workflows (`run_ui.ps1`, `run_portal_v5.ps1`, `train_fast_sft.ps1` style).

4. Portal architecture pivot
- Portal moved from simple route shell to authenticated production-like stack.
- Added startup preflight, model warm-start option, auth gating, health reporting, SSE streaming, user logging.
- Result: portal became deployable and debuggable, not just a demo.

5. Auth + user-data pivot
- Requirement changed from open access to controlled entry.
- Added Google OAuth login and per-user local folders under `data/users`.
- Result: each user session can be audited and attributed.

6. Runtime tooling pivot (RTX 50 / Blackwell)
- Stable CUDA wheels were not fully aligned with `sm_120` during the transition.
- Moved to nightly PyTorch CUDA wheels (`cu130` currently active) and verified CUDA runtime explicitly.
- Result: GPU execution restored and validated (`cuda=True`, `cap=(12,0)`).

7. Multimodal pivot
- Desktop initially only emitted image markers (`[attached images: N]`) without reliable payload handling.
- Restored processor-backed multimodal flow and image retention lifecycle.
- Added portal-side image upload path + transcript preview rendering.
- Result: user-attached images are now sent to the model and visible in chat transcript.

## Environment Baseline

```powershell
Set-Location D:\AthenaPlayground\AthenaV5
& D:\AthenaPlayground\.venv\Scripts\Activate.ps1
```

## Canonical Commands

Desktop UI (Qt default):
```powershell
.\run_ui.ps1
```

Desktop UI (Tk fallback forced):
```powershell
.\run_ui.ps1 -LegacyTk
```

Portal local (no tunnel):
```powershell
.\run_portal.ps1 -LoadModel
```

Portal full stack (server + cloudflared):
```powershell
.\run_portal_v5.ps1
```

Finetune (args-driven):
```powershell
Set-Location .\Finetune
.\run_training.ps1 -ArgsFile .\finetune_args_aimo_fast.json
```

## Portal Auth and Logging

When `ATHENA_AUTH_REQUIRED=1`, these env vars are required:
- `ATHENA_GOOGLE_CLIENT_ID`
- `ATHENA_GOOGLE_CLIENT_SECRET`
- `ATHENA_AUTH_REDIRECT_URI`
- `ATHENA_PORTAL_SESSION_SECRET`

Recommended local setup:
- copy `portal_auth.env.example` to `portal_auth.env`
- fill values once
- `run_portal_v5.ps1` auto-loads `portal_auth.env` (or `.env.portal` / `.env`)
- delta token logging is off by default (`ATHENA_LOG_DELTAS=0`) to prevent NDJSON bloat

Log root defaults to:
- `D:\AthenaPlayground\AthenaV5\data\users`

Per-user layout:
- `data/users/<user_key>/profile.json`
- `data/users/<user_key>/sessions/YYYY-MM-DD.ndjson`
- `data/users/<user_key>/errors/YYYY-MM-DD.ndjson`
- `data/users/<user_key>/uploads/YYYY-MM-DD/*`

## Model Path Source of Truth
- Canonical default model path is resolved by `athena_paths.py`.
- Launchers may accept override flags/env vars, but `athena_paths.py` is the base fallback.

## System Prompt Source of Truth
- Prompt loading is JSON-first:
  1. `system_prompt.json`
  2. fallback to `system_prompt.txt`
- Use `system_prompt.json` for structured rules and few-shot examples.

## Known Constraints
- The active environment is tuned for local inference and portal serving first.
- Nightly CUDA stacks can shift quickly; pin only after a stable release supports `sm_120` natively.
- Very large image payloads are rejected (8MB per image, max 6 images per request on portal).

## Immediate Execution Plan

1. Portal-first hardening
- Keep portal as primary interaction surface.
- Validate auth/login flow and SSE under tunnel daily.

2. AIMO SFT loop
- Continue curated math-only SFT runs with explicit args snapshots.
- Evaluate after every run; do not merge datasets blindly.

3. Swarm prep
- Keep base single-model stack stable while preparing role-specialized models (NT/ALG/COMB/GEO) and router logic.

4. Kaggle path
- Maintain a minimal reproducible submission pipeline while model quality improves.

## Related Docs
- Portal details: `portal/README.md`
- Finetune details: `Finetune/README.md`
- Project planning notes: `agentic_project_planning.txt`

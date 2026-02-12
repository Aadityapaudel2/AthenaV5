from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
# Paste your exact model folder path here when you switch models.
CHAT_MODEL_DIR = Path(r"D:\AthenaPlayground\AthenaV5\models\Qwen3-1.7B")

GUI_CONFIG_PATH = ROOT_DIR / "gui_config.json"


def get_default_chat_model_dir() -> Path:
    """Resolve chat model directory from env override or the configured default."""
    override = os.environ.get("ATHENA_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return CHAT_MODEL_DIR.resolve()

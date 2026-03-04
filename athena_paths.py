from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
# Paste your exact model folder path here when you switch models.
CHAT_MODEL_DIR = Path(r"D:\AthenaPlayground\AthenaV5\models\Qwen3.5-2B")

GUI_CONFIG_PATH = ROOT_DIR / "gui_config.json"


def get_default_chat_model_dir() -> Path:
    """Canonical model path for chat runtimes."""
    return CHAT_MODEL_DIR.resolve()

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from athena_paths import GUI_CONFIG_PATH, get_default_chat_model_dir

MODEL_DIR = get_default_chat_model_dir()
CONFIG_PATH = GUI_CONFIG_PATH


@dataclass
class GuiSettings:
    temperature: float
    max_new_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    # Qwen3 chat template switch (hard toggle). Default is True on the model side,
    # but we default to False here because most users prefer faster responses.
    enable_thinking: bool = False
    # UI preference: hide <think>...</think> in the chat window.
    hide_thoughts: bool = True

    # Few-shot style primer (prepended to every prompt, after the system message).
    # This is optional and can be disabled.
    use_primer: bool = True
    # Path is resolved relative to the UI folder.
    primer_path: str = "primer_25.json"

    @staticmethod
    def load(path: Path) -> "GuiSettings":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {}
        return GuiSettings(
            # Qwen3 docs suggest:
            # - thinking:    temp=0.6, top_p=0.95
            # - non-thinking: temp=0.7, top_p=0.8
            # We default to non-thinking-friendly parameters here.
            temperature=float(data.get("temperature", 0.7)),
            max_new_tokens=int(data.get("max_new_tokens", 16000)),
            top_p=float(data.get("top_p", 0.8)),
            top_k=int(data.get("top_k", 20)),
            repetition_penalty=float(data.get("repetition_penalty", 1.05)),
            enable_thinking=bool(data.get("enable_thinking", False)),
            hide_thoughts=bool(data.get("hide_thoughts", True)),
            use_primer=bool(data.get("use_primer", True)),
            primer_path=str(data.get("primer_path", "primer_25.json")),
        )

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "enable_thinking": self.enable_thinking,
            "hide_thoughts": self.hide_thoughts,
            "use_primer": self.use_primer,
            "primer_path": self.primer_path,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class _StopOnEvent(StoppingCriteria):
    """Stops generation when a threading.Event is set."""

    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self._stop_event = stop_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        return self._stop_event.is_set()


class LocalStreamer:
    def __init__(self, settings: GuiSettings, model_dir: Optional[Path | str] = None):
        self.settings = settings
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        if model_dir is None:
            resolved_model_dir = MODEL_DIR
        else:
            resolved_model_dir = Path(model_dir).expanduser().resolve()
        if not resolved_model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {resolved_model_dir}")

        self.model_dir = resolved_model_dir
        self.model_label = self.model_dir.name

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=False)
        # Many Qwen checkpoints do not define a pad token; make generation happy.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        self.active_streamer: Optional[TextIteratorStreamer] = None
        self.stop_event = threading.Event()

    def _eos_token_ids(self) -> Optional[list[int] | int]:
        """Return a robust EOS token id/list.

        For ChatML-style Qwen templates, the end-of-message token is often <|im_end|>.
        We include it if present so generation stops cleanly at end-of-turn.
        """
        eos_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))

        # ChatML end token used by many Qwen chat templates.
        try:
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            unk_id = getattr(self.tokenizer, "unk_token_id", None)
            if (
                im_end_id is not None
                and im_end_id >= 0
                and im_end_id not in eos_ids
                and (unk_id is None or im_end_id != unk_id)
            ):
                eos_ids.append(int(im_end_id))
        except Exception:
            pass

        if not eos_ids:
            return None
        return eos_ids[0] if len(eos_ids) == 1 else eos_ids

    def stream(self, prompt: str, callback: Callable[[str], None]) -> None:
        """Stream model output for a *fully-rendered* prompt string."""
        if not prompt.strip():
            callback("Type a prompt before streaming.\n")
            return

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        cfg = GenerationConfig(
            temperature=self.settings.temperature,
            max_new_tokens=self.settings.max_new_tokens,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k if self.settings.top_k > 0 else None,
            do_sample=True,
            repetition_penalty=self.settings.repetition_penalty,
            eos_token_id=self._eos_token_ids(),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,  # CRITICAL: don't echo the prompt back into the UI
            timeout=5.0,
        )
        self.active_streamer = streamer

        stopping_criteria = StoppingCriteriaList([_StopOnEvent(self.stop_event)])

        def generate() -> None:
            with torch.no_grad(), self.accelerator.autocast():
                self.model.generate(
                    **inputs,
                    generation_config=cfg,
                    streamer=streamer,
                    stopping_criteria=stopping_criteria,
                )

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

        for chunk in streamer:
            if not chunk:
                continue
            callback(chunk)

        self.active_streamer = None
        self.stop_event.clear()

    def stop(self) -> None:
        self.stop_event.set()
        if self.active_streamer:
            # End the iterator promptly; generation itself is stopped via stopping_criteria.
            self.active_streamer.end()

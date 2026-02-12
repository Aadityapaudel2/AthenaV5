# Athena V5 Workspace

`AthenaV5/` contains everything needed to prepare data, finetune local Qwen checkpoints, and run the local UI.

## Layout
- `Finetune/prepare_data.py`: converts turn-level JSONL into conversation-level SFT JSONL expected by `train.py`.
- `Finetune/train.py`: lean supervised finetuning trainer (PyTorch + Transformers) for `{"messages":[...]}` rows.
- `Finetune/run_training.ps1`: canonical training entrypoint with resolved absolute paths.
- `ui.py` + `tk_chat.py`: local streaming chat UI and model loader.
- `athena_paths.py`: single default model path for runtime chat loading.
- `Finetune/trainingdata/`: source and processed datasets.
- `models/`: base checkpoints and tuned outputs.

## Environment
```powershell
Set-Location D:\AthenaPlayground\AthenaV5
.\.venv\Scripts\Activate.ps1
```

## Canonical Commands
Run data preparation:
```powershell
python .\Finetune\prepare_data.py
```

Run finetuning:
```powershell
. .\Finetune\run_training.ps1
```

Run UI:
```powershell
python .\ui.py
```

## Data Pipeline Deep Dive
### Why `prepare_data.py` Exists
Raw training files like `bhagavaggitatrainingdata.jsonl` are turn-level records (`teacher`/`student` on each line). `train.py` does not train from single turns. It expects each line to already be one conversation object with `messages` in order.

`prepare_data.py` solves that mismatch by:
1. Grouping rows by `metadata.dialogue_id`.
2. Sorting each group by `metadata.turn`.
3. Mapping source roles into training roles:
- source `assistant_role` -> model `assistant`
- opposite role -> model `user`
4. Writing one output JSON object per dialogue:
- `{"meta": {...}, "messages": [{"role":"user|assistant","content":"..."}, ...]}`

Without this conversion, `train.py` cannot correctly build prompt/target spans for supervised learning.

### `prepare_data.py` Arguments
- `--input`
Purpose: source turn-level JSONL file.
Effect: selects which raw corpus is converted.

- `--output`
Purpose: destination conversation-level JSONL.
Effect: file used directly by `train.py --train_file`.

- `--assistant_role` (`student` or `teacher`)
Purpose: chooses whose voice the model learns to generate.
Effect: this is the highest-impact data setting for persona direction.

- `--drop_empty`
Purpose: drop conversations with no mapped assistant messages.
Effect: avoids training on unusable rows.

## Training Pipeline Deep Dive
`run_training.ps1` resolves paths, creates output folder if needed, then launches:
- `accelerate launch Finetune/train.py ...`

`train.py` then:
1. Reads each JSONL line and extracts `messages`.
2. Formats each conversation into role-tagged text using `<|user|>` and `<|assistant|>`.
3. Tokenizes with truncation at `max_seq_length`.
4. Builds labels so loss is computed only on assistant spans.
5. Runs HF `Trainer` with your configured hyperparameters.

## `train.py` / `run_training.ps1` Argument Literature Review
### Model and IO Arguments
- `--model_name_or_path`
Meaning: base checkpoint to finetune.
Behavior impact: defines starting capabilities and coherence ceiling.

- `--train_file`
Meaning: prepared conversation JSONL path.
Behavior impact: strongest driver of final style and knowledge adaptation.

- `--output_dir`
Meaning: where tuned model and checkpoints are saved.
Behavior impact: none on quality; critical for versioning and reproducibility.

### Sequence and Batch Geometry
- `--max_seq_length`
Meaning: max tokens per sample after tokenization.
Behavior impact: higher values preserve longer context, but increase compute and memory. Too low truncates supervision.

- `--per_device_train_batch_size`
Meaning: samples processed per device per micro-step.
Behavior impact: higher batch reduces gradient noise but raises VRAM usage.

- `--gradient_accumulation_steps`
Meaning: number of micro-steps accumulated before optimizer update.
Behavior impact: sets effective batch size without increasing VRAM proportionally.
Formula: effective batch = `per_device_train_batch_size * gradient_accumulation_steps * num_devices`.

### Optimization Arguments
- `--learning_rate`
Meaning: optimizer step size.
Behavior impact: most sensitive knob. Too high destabilizes or overfits; too low under-updates.

- `--num_train_epochs`
Meaning: full passes over the dataset.
Behavior impact: `1` epoch often under-adapts small corpora; more epochs increase adaptation and overfit risk.

- `--warmup_ratio`
Meaning: fraction of total steps used to ramp LR from near-zero to target LR.
Behavior impact: improves early-step stability, especially on small datasets.

### Logging and Checkpointing
- `--logging_steps`
Meaning: print/train-log cadence in optimizer steps.
Behavior impact: no direct quality impact; affects observability.

- `--save_steps`
Meaning: checkpoint save cadence.
Behavior impact: no direct quality impact; affects resume granularity and disk usage.

### Precision and Memory
- `--bf16`
Meaning: bfloat16 mixed precision.
Behavior impact: usually preferred when supported due better numeric range than fp16 at similar memory cost.

- `--fp16`
Meaning: float16 mixed precision.
Behavior impact: good speed/memory; can be less numerically stable than bf16.

- `--gradient_checkpointing`
Meaning: recompute activations during backward pass to save memory.
Behavior impact: enables larger sequence/model training on limited VRAM, but increases wall-clock time.

- `--seed`
Meaning: random seed.
Behavior impact: improves reproducibility of data order and initialization-sensitive effects.

## Current Training Profile (as of this README update)
Check `Finetune/run_training.ps1` for the source of truth. Typical values currently used in this workspace:
- `max_seq_length=512`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- `learning_rate=1e-5`
- `num_train_epochs=4`
- `warmup_ratio=0.05`
- `logging_steps=10`
- `save_steps=999999` (`--save_only_model` enabled; effectively save-at-end only)
- `bf16`
- `gradient_checkpointing`

## Practical Notes Learned in This Project
- Small datasets can visibly overfit small models quickly; larger bases are often more stable.
- Behavior change is mostly driven by data quality + total useful updates, not checkpoint save cadence.
- `save_steps` controls recovery snapshots, not model quality.
- Warnings seen during training (`use_cache` with checkpointing, tokenizer deprecation, accelerate defaults) are typically non-fatal.
- Use fixed eval prompts after each run to compare versions before promoting a checkpoint.

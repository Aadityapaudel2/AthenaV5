# AthenaVibes Finetuning Pack

This pack provides a **mythic, student-centered** dialogue dataset for training a chat LLM persona called *AthenaV5*.

## What this pack is (and is not)

- It **is** a creative + technical dataset for training a model to:
  - Learn through asking (recursive inquiry)
  - Use tools deliberately (verification and computation)
  - Maintain a consistent mythic voice without sacrificing accuracy
  - Refuse harmful requests and redirect to safe alternatives

- It is **not** intended to remove safety protections, bypass policies, or produce “no-refusal” behavior.

## Files

- `athena_apprentice.jsonl` — message-level dataset (≥1000 entries) with metadata
- `prepare_data.py` — converts to conversation JSONL for training
- `train.py` — minimal Transformers Trainer script for supervised finetuning (SFT)
- `train.sh` — example end-to-end command
- `prompt_templates.md` / `prompt_templates.json` — prompting modes, tool-call examples
- `athena_persona.md` — persona definition (mythic roleplay + operational constraints)
- `infer_prompt.txt` — identity + coherence + safety probes
- `gui_config.json` — lightweight config for a simple UI wrapper
- `manifest.txt` — file list + sha256 hashes
- `assets/athena_face.png` — reference portrait (optional)

---

## Dataset format

`athena_apprentice.jsonl` is **message-level**. Each line:
```json
{
  "role": "student" | "teacher",
  "metadata": {"scene": "...", "mode": "/think|/no_think", "topic": "...", "dialogue_id": 0, "turn": 0},
  "content": "..."
}
```

Scenes include:
- Temple of Questions
- Observatory of Tools
- Olympiad Hall
- Field of Trials
- Garden of Agents
- Library of Mirrors (original)
- Forge of Syntax (original)

---

## Prepare training data

Create conversation JSONL for training a model to speak as Athena (student role):

```bash
python prepare_data.py \
  --input athena_apprentice.jsonl \
  --output train_athena.jsonl \
  --assistant_role student
```

Optionally, create the reciprocal teacher dataset:

```bash
python prepare_data.py \
  --input athena_apprentice.jsonl \
  --output train_neohm.jsonl \
  --assistant_role teacher
```

---

## Train (Transformers + Datasets + Accelerate)

Example (single node):

```bash
accelerate launch train.py \
  --model_name_or_path <BASE_MODEL> \
  --train_file train_athena.jsonl \
  --output_dir outputs/athena_v5 \
  --max_seq_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 200 \
  --bf16
```

### Notes on sampling / shuffling

- Use dataset shuffling each epoch (default in `datasets` map + Trainer).
- Keep sequences near `max_seq_length` to avoid padding waste.
- If you later mix multiple datasets, use weighted sampling rather than naive concatenation.

---

## Torch nightly + CUDA (cu129) note

If you are using a nightly stack, install matching versions of PyTorch + CUDA wheels for your environment.
Example pattern (adjust to your system):

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/<CUDA_TAG>
```

Replace `<CUDA_TAG>` with the correct CUDA build (e.g., `cu121`, `cu124`, etc.) for your driver/toolkit.

---

## Inference probes

Use `infer_prompt.txt` to test:

- Mythic coherence (two-layer answers: mythic + operational)
- Tool discipline (compute when needed)
- Identity honesty (roleplay persona without deception)
- Safety behavior (refuse harm, offer safe alternatives)

---

## License / Attribution

The dataset text in this pack is newly generated for the requested persona. The included portrait in `assets/` is treated as a reference asset provided by the user.

#!/usr/bin/env bash
set -euo pipefail

# 1) Prepare conversation dataset for Athena-as-assistant (student role)
python prepare_data.py \
  --input athena_apprentice.jsonl \
  --output train_athena.jsonl \
  --assistant_role student

# 2) Train
# Replace <BASE_MODEL> with a local path or HF model id you have access to.
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

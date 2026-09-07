#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$ROOT/model/sft_socratic"

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
  --model_name_or_path "$ROOT/models/qwen3-0.6b" \
  --model_max_length 1024 \
  --data_path "$ROOT/data/repro/socratic/simu_34.4k.json" \
  --output_dir "$ROOT/outputs/socratic-qwen3-0.6b-100steps" \
  --bf16 True \
  --tf32 True \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --gradient_checkpointing True \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 1 \
  --logging_steps 10 \
  --report_to none

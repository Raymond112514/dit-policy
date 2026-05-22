#!/usr/bin/env bash
# LIBERO-90 DiT finetune with task conditioning.
#
# Prerequisites:
#   - Converted buffer: DSRL/data_buffers/buf.pkl (with task_name in each obs)
#   - conda env with dit-policy deps (e.g. dsrl_libero)
#
# Task conditioning modes (set MODE below):
#   text     - 90-d task one-hot from obs task_name  (recommended for LIBERO-90)
#   scene    - 23-d scene one-hot from obs task_name
#   language - 768-d BERT; needs obs['lang'] or pickle keyed by task_name
#              (default DSRL pickle is keyed by instruction bytes, not HDF5 names)
#
# Usage:
#   cd /home/raymond112514/dit-policy
#   bash scripts/train_libero_task_cond.sh
#
# Override any Hydra flag:
#   MODE=scene AC_CHUNK=10 bash scripts/train_libero_task_cond.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# ── user knobs ────────────────────────────────────────────────────────────────
MODE="${MODE:-text}"                    # text | scene | language
AC_CHUNK="${AC_CHUNK:-20}"
MAX_ITERS="${MAX_ITERS:-150000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1050}"
SEED="${SEED:-20346}"
EXP_NAME="${EXP_NAME:-bc_ac${AC_CHUNK}_resnet18_libero_task_${MODE}_150k}"

BUFFER_PATH="${BUFFER_PATH:-/home/raymond112514/DSRL/data_buffers/buf.pkl}"
LANG_EMB_PATH="${LANG_EMB_PATH:-/home/raymond112514/DSRL/libero_commands_bert_embeddings.pkl}"
WANDB_ENTITY="${WANDB_ENTITY:-raymond1123581321-university-of-california-berkeley}"

# ── train ─────────────────────────────────────────────────────────────────────
python finetune.py \
  task=libero \
  agent=diffusion \
  agent/features=resnet_gn \
  trainer=bc_cos_sched \
  exp_name="${EXP_NAME}" \
  wandb.name="${EXP_NAME}" \
  wandb.entity="${WANDB_ENTITY}" \
  buffer_path="${BUFFER_PATH}" \
  ac_chunk="${AC_CHUNK}" \
  agent.features.size=18 \
  max_iterations="${MAX_ITERS}" \
  save_freq="${SAVE_FREQ}" \
  batch_size="${BATCH_SIZE}" \
  seed="${SEED}" \
  task_conditioning=true \
  task_conditioning_mode="${MODE}" \
  language_embedding_path="${LANG_EMB_PATH}"

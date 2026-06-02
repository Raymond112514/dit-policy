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

#MODE="${MODE:-text}"                    # text | scene | language
#AC_CHUNK="${AC_CHUNK:-30}"
#MAX_ITERS="${MAX_ITERS:-150000}"
#SAVE_FREQ="${SAVE_FREQ:-10000}"
#BATCH_SIZE="${BATCH_SIZE:-512}"
#SEED="${SEED:-20346}"
#EXP_NAME="${EXP_NAME:-bc_ac${AC_CHUNK}_resnet18_libero_task_${MODE}_150k}"

#CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/raymond112514/DSRL/checkpoints/${EXP_NAME}}"
USE_RUN_TIMESTAMP="${USE_RUN_TIMESTAMP:-1}"
if [[ "${USE_RUN_TIMESTAMP}" == "1" ]]; then
  RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y-%m-%d_%H-%M-%S)}"
  RUN_DIR="${RUN_DIR:-${CHECKPOINT_DIR}/wandb_${EXP_NAME}_libero_resnet_gn_${RUN_SUFFIX}}"
else
  RUN_DIR="${RUN_DIR:-${CHECKPOINT_DIR}}"
fi
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_DIR}/latest.ckpt}"

#mkdir -p "${RUN_DIR}"

#BUFFER_PATH="${BUFFER_PATH:-/global/scratch/users/r112358/libero_dataset/buf.pkl}"
#LANG_EMB_PATH="${LANG_EMB_PATH:-/global/home/users/r112358/dsrl/libero_commands_bert_embeddings.pkl}"
#WANDB_ENTITY="${WANDB_ENTITY:-raymond1123581321-university-of-california-berkeley}"

python finetune.py \
  hydra/launcher=basic \
  hydra.run.dir="${RUN_DIR}" \
  task=libero \
  agent=diffusion \
  agent/features=resnet_gn \
  trainer=bc_cos_sched \
  exp_name="${EXP_NAME}" \
  wandb.name="${EXP_NAME}" \
  checkpoint_path="${CHECKPOINT_PATH}" \
  wandb.entity="${WANDB_ENTITY}" \
  buffer_path="${BUFFER_PATH}" \
  ac_chunk="${AC_CHUNK}" \
  agent.features.size=18 \
  max_iterations="${MAX_ITERS}" \
  save_freq="${SAVE_FREQ}" \
  batch_size="${BATCH_SIZE}" \
  seed="${SEED}" \
  #task_conditioning=true \
  #task_conditioning_mode="${MODE}" \
  #language_embedding_path="${LANG_EMB_PATH}"

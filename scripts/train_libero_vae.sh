#!/usr/bin/env bash
# LIBERO-90 DiT finetune with VAE latent actions.
#
# Action chunks (ac_chunk) are encoded into z_dim-d latents by a pre-trained
# ActionVAE at dataset build time.  The BC policy is then trained to predict
# those latents instead of raw action chunks.
#
# Prerequisites:
#   - Converted buffer at BUFFER_PATH (with image obs)
#   - Trained VAE checkpoint at VAE_CHECKPOINT
#   - conda env with dit-policy deps (e.g. dsrl_libero)
#
# Usage:
#   cd /home/raymond112514/dit-policy
#   bash scripts/train_libero_vae.sh
#
# Override any variable on the command line:
#   Z_DIM=32 AC_CHUNK=10 bash scripts/train_libero_vae.sh

set -euo pipefail

cd "$(dirname "$0")/.."

#AC_CHUNK="${AC_CHUNK:-10}"               # must match VAE action_chunk_size
Z_DIM="${Z_DIM:-16}"                    # must match VAE z_dim
MAX_ITERS="${MAX_ITERS:-150000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEED="${SEED:-20346}"

VAE_ROOT="${VAE_ROOT:-/global/scratch/users/r112358/vae_checkpoints}"
#VAE_EXP="${VAE_EXP:-action_vae_img_encdec_ac10_zdim16_beta1e-3}"
#VAE_STEP="${VAE_STEP:-40000}"

BUFFER_PATH="${BUFFER_PATH:-/global/scratch/users/r112358/libero_dataset/buf.pkl}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-${VAE_ROOT}/${VAE_EXP}/step_${VAE_STEP}.pt}"
WANDB_ENTITY="${WANDB_ENTITY:-raymond1123581321-university-of-california-berkeley}"

VAE_TAG="${VAE_TAG:-${VAE_EXP#action_vae_}}"   # e.g. img_encdec_ac10_zdim16_beta1e-3
EXP_NAME="${EXP_NAME:-bc_vae_ac${AC_CHUNK}_z${Z_DIM}_${VAE_TAG}_step${VAE_STEP}}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/global/scratch/users/r112358/libero_dit_checkpoints/${EXP_NAME}}"
USE_RUN_TIMESTAMP="${USE_RUN_TIMESTAMP:-1}"
if [[ "${USE_RUN_TIMESTAMP}" == "1" ]]; then
  RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y-%m-%d_%H-%M-%S)}"
  RUN_DIR="${RUN_DIR:-${CHECKPOINT_DIR}/wandb_${EXP_NAME}_libero_resnet_gn_${RUN_SUFFIX}}"
else
  RUN_DIR="${RUN_DIR:-${CHECKPOINT_DIR}}"
fi
mkdir -p "${RUN_DIR}"
echo "RUN_DIR=${RUN_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_DIR}/latest.ckpt}"

python finetune.py \
  hydra/launcher=basic \
  hydra.run.dir="${RUN_DIR}" \
  task=libero \
  agent=diffusion \
  agent/features=resnet_gn \
  trainer=bc_cos_sched \
  exp_name="${EXP_NAME}" \
  wandb.name="${EXP_NAME}" \
  wandb.entity="${WANDB_ENTITY}" \
  checkpoint_path="${CHECKPOINT_PATH}" \
  buffer_path="${BUFFER_PATH}" \
  ac_chunk="${AC_CHUNK}" \
  agent.features.size=18 \
  max_iterations="${MAX_ITERS}" \
  save_freq="${SAVE_FREQ}" \
  batch_size="${BATCH_SIZE}" \
  seed="${SEED}" \
  vae_checkpoint="${VAE_CHECKPOINT}" \
  "agent.ac_chunk=1" \
  "task.ac_dim=${Z_DIM}" \
  max_transitions=500

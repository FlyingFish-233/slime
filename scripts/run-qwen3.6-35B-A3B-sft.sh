#!/usr/bin/env bash
set -euo pipefail

# Single-node training: keep all runtime state inside the container and use all
# eight visible GPUs.
export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd /root/slime
source "${SCRIPT_DIR}/models/qwen3.6-35B-A3B.sh"

MODEL_DIR="/root/Qwen3.6-35B-A3B"
TORCH_DIST_DIR="/root/Qwen3.6-35B-A3B_torch_dist"
SLIME_CKPT_DIR="/root/Qwen3.6-35B-A3B_slime"
HF_OUTPUT_DIR="/root/Qwen3.6-35B-A3B_slime-hf"
DATA_FILE="/root/SFT-Trajectories/data/train.jsonl"

[[ -d "${MODEL_DIR}" ]] || { echo "Missing model directory: ${MODEL_DIR}" >&2; exit 1; }
[[ -f "${DATA_FILE}" ]] || { echo "Missing dataset file: ${DATA_FILE}" >&2; exit 1; }
[[ -d "${HF_OUTPUT_DIR}" ]] || { echo "Missing Hugging Face output directory: ${HF_OUTPUT_DIR}" >&2; exit 1; }

if ! find "${TORCH_DIST_DIR}" -type f -print -quit 2>/dev/null | grep -q .; then
  echo "No torch_dist checkpoint found; converting Qwen3.6-35B-A3B with eight GPUs..."
  PYTHONPATH=/root/Megatron-LM \
    torchrun --nproc_per_node=8 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --save "${TORCH_DIST_DIR}"
fi

# Clean up stale services from a previous run inside this container.
ray stop --force >/dev/null 2>&1 || true
pkill -9 sglang >/dev/null 2>&1 || true
sleep 3

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${TORCH_DIST_DIR}"
  --save "${SLIME_CKPT_DIR}"
  --save-interval 99999999
  --no-save-optim
)

# Resume only when an actual training checkpoint exists.
if find "${SLIME_CKPT_DIR}" -type f -print -quit 2>/dev/null | grep -q .; then
  CKPT_ARGS+=(--load "${SLIME_CKPT_DIR}")
fi

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${DATA_FILE}"
  --input-key messages
  --rollout-shuffle
  --num-epoch 3
  --rollout-batch-size 8
  --global-batch-size 8

  --loss-type sft_loss
  --loss-mask-type qwen3_5
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
  # --save-debug-rollout-data "/root/slime/sft_dumps/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
  # CP and EP reuse the same eight ranks in their respective process groups:
  # CP=8 handles this dataset's 85k-token maximum, while EP=8 places 32 of the
  # 256 experts on each GPU. TP=2 together with CP=8/EP=8 needs more than 8 GPUs.
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 8
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  # 8 * 12288 = 98304 tokens of total CP capacity per micro-batch.
  --max-tokens-per-gpu 12288
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-5
  --lr-decay-style cosine
  --min-lr 1e-6
  --lr-warmup-fraction 0.1
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98

  --use-distributed-optimizer
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

WANDB_ARGS=(
  --use-wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group qwen3.6-35B-A3B-sft
  --wandb-team "${WANDB_TEAM}"
  --wandb-host "${WANDB_BASE_URL}"
  --wandb-key "${WANDB_API_KEY}"
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash

   --moe-token-dispatcher-type alltoall
)

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}"

latest_iteration="$(tr -d "[:space:]" < "${SLIME_CKPT_DIR}/latest_checkpointed_iteration.txt")"
[[ "${latest_iteration}" =~ ^[0-9]+$ ]] || {
  echo "Invalid latest checkpoint iteration: ${latest_iteration}" >&2
  exit 1
}
printf -v checkpoint_dir "${SLIME_CKPT_DIR}/iter_%07d" "$((10#${latest_iteration}))"
[[ -d "${checkpoint_dir}" ]] || {
  echo "Missing latest checkpoint directory: ${checkpoint_dir}" >&2
  exit 1
}

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir "${checkpoint_dir}" \
  --output-dir "${HF_OUTPUT_DIR}" \
  --origin-hf-dir "${MODEL_DIR}" \
  --vocab-size 248320 \
  --add-missing-from-origin-hf \
  --force

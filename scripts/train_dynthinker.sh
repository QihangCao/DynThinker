#!/usr/bin/env bash
# DynThinker supervised fine-tuning launcher (torchrun + DeepSpeed ZeRO-2).
#
# Required environment variables:
#   MODEL_PATH       Qwen3-VL-Instruct checkpoint (local dir or Hub id)
#   VDPM_PATH        VDPM weights (model.pt)                        [not needed with USE_SDF=0]
#   DATASET_CONFIG   dataset registry (see configs/datasets.example.yaml)
#   DATASETS         comma-separated names from the registry, e.g. "spar_234k,llava_hound_64k"
#   DATASET_SIZE     total number of training samples after sampling (for the D-EMA decay)
# Optional:
#   OUTPUT_DIR (./output/dynthinker)  NPROC_PER_NODE (8)  NNODES (1)  NODE_RANK (0)
#   MASTER_ADDR (127.0.0.1)  MASTER_PORT (29500)  GLOBAL_BATCH (64)  LR (1e-5)
#   USE_SDF (1: VDPM + SDF; 0: D-EMA only, also the only mode for Qwen2.5-VL)
#   TOKEN_COMPRESSION (1, as in the research script): samples with > 8 frames use merge size x2 (4x fewer
#       visual tokens). Recorded in the checkpoint config, so it also applies at inference; 0 disables.
#   DEMA_DRIFT_MLP / DEMA_DRIFT_OTHER (0.125 / 0.125): drift budgets of the late-MLP / other groups;
#       a budget >= 1 disables averaging for that group.
#   DEMA_START (150)  DEMA_INTERVAL (10)  DEMA_MLP_THRESHOLD (4)  SAVE_STEPS (250)
#   ATTN_IMPLEMENTATION (flash_attention_2)  EXTRA_ARGS (appended verbatim)
#   DRY_RUN=1 prints the torchrun command instead of running it.
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH}"
: "${DATASET_CONFIG:?set DATASET_CONFIG}"
: "${DATASETS:?set DATASETS}"
: "${DATASET_SIZE:?set DATASET_SIZE (number of training samples)}"
USE_SDF=${USE_SDF:-1}
if [[ "$USE_SDF" == "1" ]]; then
    : "${VDPM_PATH:?set VDPM_PATH (or USE_SDF=0)}"
fi

OUTPUT_DIR=${OUTPUT_DIR:-./output/dynthinker}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
LR=${LR:-1e-5}
TOKEN_COMPRESSION=${TOKEN_COMPRESSION:-1}
DEMA_DRIFT_MLP=${DEMA_DRIFT_MLP:-0.125}
DEMA_DRIFT_OTHER=${DEMA_DRIFT_OTHER:-0.125}
DEMA_START=${DEMA_START:-150}
DEMA_INTERVAL=${DEMA_INTERVAL:-10}
DEMA_MLP_THRESHOLD=${DEMA_MLP_THRESHOLD:-4}
SAVE_STEPS=${SAVE_STEPS:-250}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-$SCRIPT_DIR/../configs/deepspeed_zero2.json}

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if (( GLOBAL_BATCH % WORLD_SIZE != 0 )); then
    echo "GLOBAL_BATCH=$GLOBAL_BATCH is not divisible by world size $WORLD_SIZE" >&2
    exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / WORLD_SIZE))

# D-EMA decay from the drift budget: rho = (1 - delta)^(1 / N_updates).
read -r MLP_DECAY MLP_SAVE OTHER_DECAY OTHER_SAVE < <(python - "$DATASET_SIZE" "$GLOBAL_BATCH" "$DEMA_START" \
    "$DEMA_INTERVAL" "$DEMA_DRIFT_MLP" "$DEMA_DRIFT_OTHER" <<'PY'
import sys
from dynthinker.tools.ema_decay import beta_from_drift, compute_ema_updates

size, batch, start, interval = (int(v) for v in sys.argv[1:5])
_, updates = compute_ema_updates(size, batch, start, interval)
out = []
for drift in (float(sys.argv[5]), float(sys.argv[6])):
    if drift >= 1.0:
        out += ["0.0", "False"]  # no averaging: keep the online weights of this group
    else:
        out += [f"{beta_from_drift(drift, updates):.6f}", "True"]
print(" ".join(out))
PY
)
echo "D-EMA: mlp_decay=$MLP_DECAY (save=$MLP_SAVE) other_decay=$OTHER_DECAY (save=$OTHER_SAVE)"

SDF_ARGS=(--use_geometry_encoder False --geo_cross_attn False)
if [[ "$USE_SDF" == "1" ]]; then
    SDF_ARGS=(
        --use_geometry_encoder True
        --geometry_encoder_type vdpm
        --geometry_encoder_path "$VDPM_PATH"
        --feature_fusion_method zero
        --cam_merger_type zero
        --geo_cross_attn True
        --geo_inject_version v1
        --geo_importance_gate True
        --vdpm_use_decoder_features True
        --vdpm_temporal_mode diag
        --vdpm_decoder_layer_offset -2
        --vdpm_compute_prediction False
        --vdpm_concat_encoder_decoder_features True
    )
fi
COMPRESSION_ARGS=()
if [[ "$TOKEN_COMPRESSION" == "1" ]]; then
    COMPRESSION_ARGS=(--depart_smi_token True --smi_image_num 8 --smi_downsample_rate 2)
fi

CMD=(
torchrun \
    --nnodes "$NNODES" --node-rank "$NODE_RANK" --nproc-per-node "$NPROC_PER_NODE" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    -m dynthinker.train.train \
    --model_name_or_path "$MODEL_PATH" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --tune_mm_llm True --tune_mm_vision False --tune_mm_mlp False \
    --dataset_config "$DATASET_CONFIG" \
    --dataset_use "$DATASETS" \
    --output_dir "$OUTPUT_DIR" \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" --mm_projector_lr 1e-5 --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_max_length 12800 \
    --max_pixels $((576 * 28 * 28)) --min_pixels $((16 * 28 * 28)) \
    --base_interval 2 --video_max_frames 8 --video_min_frames 4 \
    --video_max_frame_pixels $((1664 * 28 * 28)) --video_min_frame_pixels $((256 * 28 * 28)) \
    --num_train_epochs 1 --warmup_ratio 0.03 --lr_scheduler_type cosine --weight_decay 0.01 \
    --logging_steps 1 --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 2 \
    --gradient_checkpointing --dataloader_num_workers 4 \
    --group_by_modality_length True --seed 0 --report_to tensorboard \
    "${SDF_ARGS[@]}" \
    ${COMPRESSION_ARGS[@]+"${COMPRESSION_ARGS[@]}"} \
    --ema_enable True \
    --ema_start_step "$DEMA_START" --ema_update_interval "$DEMA_INTERVAL" \
    --ema_trainable_only True --ema_mlp_layer_threshold "$DEMA_MLP_THRESHOLD" \
    --ema_mlp_decay "$MLP_DECAY" --ema_other_decay "$OTHER_DECAY" \
    --ema_mlp_save_as_final "$MLP_SAVE" --ema_other_save_as_final "$OTHER_SAVE" \
    --ema_new_module_decay -1.0 --ema_new_module_save_as_final False \
    --ema_on_gpu False --ema_rank_zero_only True \
    --ema_checkpoint_save True --ema_checkpoint_strict True --ema_checkpoint_dtype float32
)
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})
CMD+=(${EXTRA[@]+"${EXTRA[@]}"})

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    echo
    exit 0
fi
mkdir -p "$OUTPUT_DIR"
exec "${CMD[@]}"

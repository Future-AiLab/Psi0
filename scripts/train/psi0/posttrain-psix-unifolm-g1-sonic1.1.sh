#!/bin/bash
# Usage: bash scripts/train/psi0/posttrain-psix-unifolm-g1.sh [exp] [timestamp]
#   Pass the same <exp> <timestamp> pair again to resume that run dir / W&B run.

set -uo pipefail

TORCHRUN_PID=
PYTHON_BIN=
CLEANUP_RUNNING=0

cleanup() {
    if [ "$CLEANUP_RUNNING" -eq 1 ]; then return; fi
    CLEANUP_RUNNING=1
    echo "Interrupted - stopping torchrun and worker processes..."
    trap - INT TERM
    if [ -n "$TORCHRUN_PID" ] && kill -0 "$TORCHRUN_PID" 2>/dev/null; then
        kill -TERM "$TORCHRUN_PID" 2>/dev/null || true
    fi
    if [ -n "$PYTHON_BIN" ]; then
        pkill -TERM -f "$PYTHON_BIN" 2>/dev/null || true
    fi
    if [ -n "$TORCHRUN_PID" ]; then
        wait "$TORCHRUN_PID" 2>/dev/null || true
    fi
    if [ -n "$TORCHRUN_PID" ] && kill -0 "$TORCHRUN_PID" 2>/dev/null; then
        kill -KILL "$TORCHRUN_PID" 2>/dev/null || true
    fi
    if [ -n "$PYTHON_BIN" ]; then
        pkill -KILL -f "$PYTHON_BIN" 2>/dev/null || true
    fi
    # A rank wedged in a CUDA/NCCL call can outlive both torchrun's SIGTERM sweep and
    # the pkill above, get reparented to init, and sit there holding ~64 GB of HBM.
    # The next launch then dies with "CUDA out of memory ... Process <old pid> has
    # 62.9 GiB in use" while looking like a model-too-big problem. So verify, and say
    # so loudly if anything survives.
    if [ -n "$PYTHON_BIN" ]; then
        pat="$PYTHON_BIN.*scripts/train.py"
        for _ in $(seq 1 15); do
            pgrep -f "$pat" >/dev/null 2>&1 || break
            sleep 1
        done
        if pgrep -f "$pat" >/dev/null 2>&1; then
            echo "cleanup: ranks alive after SIGKILL sweep, retrying" >&2
            pkill -KILL -f "$pat" 2>/dev/null || true
            sleep 3
        fi
        if pgrep -f "$pat" >/dev/null 2>&1; then
            echo "cleanup: WARNING orphaned ranks REMAIN and still hold GPU memory:" >&2
            pgrep -af "$pat" >&2
            echo "cleanup: kill them before relaunching, or the next run will OOM" >&2
        else
            echo "cleanup: all ranks gone"
        fi
    fi
}
trap cleanup INT TERM

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

export exp="${1:- }"
TS="${2:-}"   # %y%m%d%H%M run-dir suffix; empty -> fresh run

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"
PYTHON_BIN=$(readlink -f "$(command -v python3)")

# --- environment ------------------------------------------------------------
# scripts/train.py calls load_dotenv(), which does NOT override variables already
# exported in the shell/container -- so .env loses every time the container sets a
# value. The h100 container exports HF_HUB_OFFLINE=0 and HF_ENDPOINT=hf-mirror.com,
# so every from_pretrained() does a network HEAD against an unreachable mirror and
# hangs on retries, even though the weights (incl. openai/clip-vit-large-patch14)
# are already in $HF_HOME/hub. Force .env's intent here.
: "${HF_HOME:=/mnt/beegfs/shared/hfm/cache}"
export HF_HOME
export TORCH_HOME="${TORCH_HOME:-$HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[ "$HF_HUB_OFFLINE" = "0" ] && export HF_HUB_OFFLINE=1
[ -d "$HF_HOME/hub/models--openai--clip-vit-large-patch14" ] \
    || echo "WARN: CLIP not in $HF_HOME/hub - pooled text encoder will need network" >&2

# CUDA_LAUNCH_BLOCKING=true (set in .env and in this container) serialises every
# kernel launch. That is right for debugging and a real throughput loss over 100k
# steps, so it is off by default here. Prefix the call with CUDA_LAUNCH_BLOCKING=1
# to put it back.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
[ "$CUDA_LAUNCH_BLOCKING" = "true" ] && export CUDA_LAUNCH_BLOCKING=0

: "${OMP_NUM_THREADS:=32}"
export OMP_NUM_THREADS
echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"

NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"
echo "Experiment name: $exp"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$((29600 + ${SLURM_JOB_ID:-0} % 2000))}

# --- data preflight ----------------------------------------------------------
ROOT_DIR="${PSI_DATA_ROOT:-.data}"   # slurm_job.sh points this at the node-local stage
TRAIN_ID="unifolm_sonic_lerobot_train"
VAL_ID="${VAL_ID:-unifolm_sonic_lerobot_val}"
PACK="$ROOT_DIR/$TRAIN_ID"
STATS="$PACK/meta/stats_psi0.json"

for p in "$ROOT_DIR/$TRAIN_ID" "$ROOT_DIR/$VAL_ID"; do
    [ -d "$p" ]                      || { echo "FATAL: missing pack $p" >&2; exit 1; }
    [ -s "$p/meta/modality.json" ]   || { echo "FATAL: missing $p/meta/modality.json" >&2; exit 1; }
    [ -s "$p/meta/stats_psi0.json" ] || { echo "FATAL: missing $p/meta/stats_psi0.json" >&2; exit 1; }
done
# train and val MUST share one normalisation
if ! cmp -s "$STATS" "$ROOT_DIR/$VAL_ID/meta/stats_psi0.json"; then
    echo "FATAL: train/val stats_psi0.json differ - they must be the same global stats" >&2
    exit 1
fi

# VLM weights to start from. The ACTION HEADER is deliberately NOT initialised from
# a checkpoint -- PosttrainTrainer always builds it from scratch, so the header is
# learned fresh on this pack while the VLM is finetuned from the pretrained backbone.
VLM_CKPT="${VLM_CKPT:-cache/checkpoints/psi0/pre.fast.2605160748.ckpt.ego390k}"

args="
posttrain_sonic_psi0_config \
--seed=292285 \
--exp=$exp \
${TS:+--timestamp=$TS --train.resume_from_checkpoint=latest} \
--train.name=posttrain \
--train.num_workers=8 \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=2 \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=2000 \
--train.validation_steps=5000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=$ROOT_DIR \
--data.train_repo_ids=$TRAIN_ID \
--data.val_repo_ids=$VAL_ID \
--data.transform.repack.dataset-name=unifolm_sonic \
--data.transform.repack.image-keys observation.images.egocentric \
--data.transform.repack.instruction-key=task_description \
--data.transform.repack.state-keys observation.state \
--data.transform.repack.action-keys action.body_token_v1_1 action[:14] \
--data.transform.repack.action-chunk-size=30 \
--data.transform.repack.pad-action-dim=80 \
--data.transform.repack.pad-state-dim=45 \
--data.transform.field.stat-path=$STATS \
--data.transform.field.stat-state-keys observation.state \
--data.transform.field.stat-action-keys action.body_token_v1_1 action[:14] \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=80 \
--data.transform.field.pad-state-dim=45 \
--data.transform.model.img-aug \
--data.transform.model.resize.size 240 320 \
--data.transform.model.center_crop.size 240 320 \
--model.model_name_or_path=$VLM_CKPT \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=80 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=45 \
--model.dropout=0.0 \
--model.state-feature-dropout=0.0 \
--model.view_feature_dim=2048 \
--model.tune-vlm \
--model.lang-backbone-lr=1e-6 \
--model.vision-tower-lr=1e-5 \
--model.mm-projector-lr=1e-4 \
--model.gradient-checkpointing \
--model.no-use_film \
--model.qk-norm=rms_norm \
--model.combined-temb \
--model.num-blocks=12 \
--model.vlm-layer-indices 3 5 8 10 12 14 17 19 21 23 26 28 \
--model.state-drop-prob=0.5 \
--model.state-as-action-token \
--model.state-null-token \
--model.pooled-text-encoder=clip \
--model.pooled-text-encoder-path=openai/clip-vit-large-patch14 \
--model.pooled-projection-dim=768 \
--model.pooled-cache-path=clip_pooled_cache.pt \
--model.no-rtc \
--model.max-delay=8
"

cat <<EOF
Running:
torchrun \\
  --nnodes=$NNODES \\
  --nproc_per_node=$NPROC_PER_NODE \\
  --node_rank=$NODE_RANK \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  scripts/train.py \\
  ${args}
EOF

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    ${args} &

TORCHRUN_PID=$!
wait "$TORCHRUN_PID"

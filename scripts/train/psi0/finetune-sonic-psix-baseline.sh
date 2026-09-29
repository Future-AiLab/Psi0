#!/bin/bash
# psi0 (sonic, neck) finetune on an ALIGNED PsiX pack .data/finetune/<DATASET>_{train,val}/g1
# (flat 80-D [hand|neck|token] action, 45-D 0810-order state, observation.images.egocentric,
# as produced by scripts/data/align_to_g1_sonic_layout.py), warm-started from the sonic-v1.1
# POST-TRAINED model, with the VLM tuned.
#
# Pick the corpus with DATASET (default psix_steerability_0918):
#   DATASET=teleop5_round2_allinone_0918 bash scripts/train/psi0/finetune-sonic-psix-baseline.sh teleop5
# or point TRAIN_ID / VAL_ID at arbitrary repo ids under $ROOT_DIR when the split dirs do not
# follow the <DATASET>_{train,val}/g1 pattern.
# DATASET_NAME sets --data.transform.repack.dataset-name (the label carried on each sample);
# default: first 10 chars of the training dir's basename (the dir above the g1/ robot subdir).

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
    # A rank wedged in a CUDA/NCCL call can outlive both torchrun's SIGTERM sweep and the
    # pkill above, get reparented to init, and sit there holding ~64 GB of HBM. The next
    # launch then dies with "CUDA out of memory ... Process <old pid> has 62.9 GiB in use"
    # while looking like a model-too-big problem. So verify, and say so loudly.
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

# When run directly on the host (not via sbatch), resolve to project root.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

export exp="${1:-ff}"
TS="${2:-}"   # %y%m%d%H%M run-dir suffix; empty -> fresh run, config mints one

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
# The pack whose state column order the post-trained model expects.
POSTTRAIN_REF="${POSTTRAIN_REF:-.data/unifolm_sonic_lerobot_train}"
ROOT_DIR="${PSI_DATA_ROOT:-.data}"   # slurm_job.sh points this at the node-local stage
# Corpus: <DATASET>_{train,val}/g1 under finetune/, or explicit TRAIN_ID / VAL_ID repo ids.
DATASET="${DATASET:-psix_steerability_0918}"
TRAIN_ID="${TRAIN_ID:-finetune/${DATASET}_train/g1}"
VAL_ID="${VAL_ID:-finetune/${DATASET}_val/g1}"
PACK="$ROOT_DIR/$TRAIN_ID"
STATS="$PACK/meta/stats_psi0.json"
# Sample label (repack dataset-name): DATASET_NAME, else the first 10 chars of the training
# dir's basename, skipping the trailing g1/ robot subdir (e.g. psix_steerability_0918_train -> psix_steer).
_train_dir="${PACK%/}"
[ "$(basename "$_train_dir")" = "g1" ] && _train_dir="$(dirname "$_train_dir")"
DATASET_NAME="${DATASET_NAME:-$(basename "$_train_dir" | cut -c1-10)}"

for p in "$ROOT_DIR/$TRAIN_ID" "$ROOT_DIR/$VAL_ID"; do
    [ -d "$p" ]                 || { echo "FATAL: missing pack $p (an aligned PsiX pack: run scripts/data/align_to_g1_sonic_layout.py on the export, or set DATASET / TRAIN_ID / VAL_ID)" >&2; exit 1; }
    [ -s "$p/meta/modality.json" ] || { echo "FATAL: missing $p/meta/modality.json" >&2; exit 1; }
done

# stats_psi0.json == stats.json by construction (simple_to_sonic_lerobot.py:219-220).
if [ ! -s "$STATS" ]; then
    [ -s "$PACK/meta/stats.json" ] || { echo "FATAL: neither stats_psi0.json nor stats.json in $PACK/meta" >&2; exit 1; }
    cp "$PACK/meta/stats.json" "$STATS"
    echo "Created $STATS (copy of stats.json)"
fi
[ -s "$STATS" ] || { echo "FATAL: stats missing: $STATS" >&2; exit 1; }
echo "Corpus: train=$ROOT_DIR/$TRAIN_ID val=$ROOT_DIR/$VAL_ID stats=$STATS dataset_name=$DATASET_NAME"

# --- warm-start preflight ----------------------------------------------------
# Split the post-trained checkpoint into the VLM dir + action_header.safetensors that
# the two --model flags below want. ~11 GB, written once and reused; the exporter
# stages into <dir>.partial and renames, so a waiting node never sees a half file.
POSTTRAIN_RUN="${POSTTRAIN_RUN:-.runs/posttrain/sonic1.1.us.flow1000.cosine.lr1.0e-04.b256.gpus8.2609092158}"
CKPT_STEP="${CKPT_STEP:-40000}"
CKPT_DIR="$POSTTRAIN_RUN/checkpoints/ckpt_$CKPT_STEP"
INIT_DIR="$POSTTRAIN_RUN/posttrained/ckpt_$CKPT_STEP"

if [ ! -s "$INIT_DIR/model.safetensors" ] || [ ! -s "$INIT_DIR/action_header.safetensors" ]; then
    [ -s "$CKPT_DIR/model.safetensors" ] || { echo "FATAL: missing $CKPT_DIR/model.safetensors" >&2; exit 1; }
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "Exporting post-trained weights: $CKPT_DIR -> $INIT_DIR"
        python3 scripts/export_psi0_ckpt.py "$CKPT_DIR" "$INIT_DIR" \
            || { echo "FATAL: export_psi0_ckpt.py failed" >&2; exit 1; }
    else
        echo "Waiting for node 0 to export $INIT_DIR ..."
        for _ in $(seq 1 180); do
            [ -s "$INIT_DIR/model.safetensors" ] && [ -s "$INIT_DIR/action_header.safetensors" ] && break
            [ -s "$INIT_DIR/model.safetensors" ] && [ -s "$INIT_DIR/action_header.safetensors" ] && break
            sleep 10
        done
    fi
fi
for f in config.json model.safetensors action_header.safetensors; do
    [ -s "$INIT_DIR/$f" ] \
        || { echo "FATAL: $INIT_DIR/$f missing (rerun: python3 scripts/export_psi0_ckpt.py $CKPT_DIR $INIT_DIR)" >&2; exit 1; }
done
echo "Warm start from $INIT_DIR (VLM + action header, ckpt_$CKPT_STEP)"

# --- robustness knobs --------------------------------------------------------
STATE_DROP_PROB="${STATE_DROP_PROB:-0.1}"     # per-sample state drop -> learned null token; 0 disables
STATE_JITTER="${STATE_JITTER:-10}"            # frames; 0 disables the temporal state aug
STATE_JITTER_PROB="${STATE_JITTER_PROB:-0.5}"
STATE_NOISE_STD="${STATE_NOISE_STD:-0.05}"    # in normalized [-1,1] units; 0 disables
VIEW_AUG_MIN_SCALE="${VIEW_AUG_MIN_SCALE:-0.85}"
VIEW_AUG_PROB="${VIEW_AUG_PROB:-1.0}"
# "<task substring>=<weight> ..." (space separated); empty -> uniform sampling
TASK_SAMPLE_WEIGHTS="${TASK_SAMPLE_WEIGHTS-}"
task_weight_args=""
[ -n "$TASK_SAMPLE_WEIGHTS" ] && task_weight_args="--data.task-sample-weights $TASK_SAMPLE_WEIGHTS"
echo "State aug: drop=${STATE_DROP_PROB} (learned null token) jitter=+-${STATE_JITTER}f p=${STATE_JITTER_PROB} noise=${STATE_NOISE_STD}; view aug: min_scale=${VIEW_AUG_MIN_SCALE} p=${VIEW_AUG_PROB}; task weights: ${TASK_SAMPLE_WEIGHTS:-none}"

args="
finetune_sonic_psi0_config \
--seed=292285 \
--exp=$exp \
${TS:+--timestamp=$TS --train.resume_from_checkpoint=latest} \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=5000 \
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
--data.transform.repack.image-keys observation.images.egocentric \
--data.transform.repack.action-keys action[16:80] action[:14] action[14:16] \
--data.transform.repack.dataset-name=$DATASET_NAME \
--data.transform.repack.pad-action-dim=80 \
--data.transform.repack.pad-state-dim=45 \
--data.transform.repack.instruction-key=task_description \
--data.transform.repack.state-temporal-jitter=$STATE_JITTER \
--data.transform.repack.state-temporal-jitter-prob=$STATE_JITTER_PROB \
--data.transform.field.stat-path=$STATS \
--data.transform.field.state-noise-std=$STATE_NOISE_STD \
--data.transform.field.stat-action-keys action[16:80] action[:14] action[14:16] \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=80 \
--data.transform.field.pad-state-dim=45 \
--data.transform.model.img-aug \
--data.transform.model.view-aug \
--data.transform.model.view-aug-min-scale=$VIEW_AUG_MIN_SCALE \
--data.transform.model.view-aug-prob=$VIEW_AUG_PROB \
--data.transform.model.resize.size 270 480 \
--data.transform.model.center_crop.size 270 480 \
$task_weight_args \
--model.model_name_or_path=$INIT_DIR \
--model.pretrained-action-header-path=$INIT_DIR \
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
--model.state-drop-prob=$STATE_DROP_PROB \
--model.state-as-action-token \
--model.state-null-token \
--model.pooled-text-encoder=clip \
--model.pooled-text-encoder-path=openai/clip-vit-large-patch14 \
--model.pooled-projection-dim=768 \
--model.pooled-cache-path=clip_pooled_cache.pt \
--model.no-rtc \
--model.max-delay=8
"

# `args` is a double-quoted string, so its backslash-newlines have already collapsed into
# one long line; print it one word per line so the log stays readable and copy-pasteable.
cat <<EOF
Running:
torchrun \\
  --nnodes=$NNODES \\
  --nproc_per_node=$NPROC_PER_NODE \\
  --node_rank=$NODE_RANK \\
  --master_addr=$MASTER_ADDR \\
  --master_port=$MASTER_PORT \\
  scripts/train.py \\
EOF
printf '  %s \\\n' ${args} | sed '$ s/ \\$//'
# DRY_RUN=1: stop here after every preflight, with the resolved arg list on stdout as
# "DRY_RUN_ARGS: ..." (one shell-quoted line) so the config can be parsed offline.
if [ -n "${DRY_RUN:-}" ]; then
    printf 'DRY_RUN_ARGS:'; printf ' %q' ${args}; printf '\n'
    echo "DRY_RUN set - not launching torchrun"
    exit 0
fi

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

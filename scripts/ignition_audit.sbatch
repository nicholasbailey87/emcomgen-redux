#!/bin/bash
# Run `ignition_audit.py` on one rung per array task.
#
# Three tasks rather than one job auditing three configs in sequence: they are
# independent, the audit writes one CSV per config, and on `preemptgpu` a
# preempted task takes only its own rung down with it. There is no checkpoint
# and no resume -- a requeued task starts its rung again from step 1 and
# overwrites its CSV, which is correct, because half a trajectory measured
# either side of a preemption is not a trajectory.
#
# Submit from the repository root:
#
#     sbatch --array=0-2 scripts/ignition_audit.sbatch
#
# and, to change the step count or the rungs, either edit the CONFIGS array
# below or pass them through:
#
#     sbatch --array=0-2 --export=ALL,STEPS=800 scripts/ignition_audit.sbatch
#
# Results land in results/ignition_audit/<config stem>.csv, logs in
# experiments/ablation/logs/.

#SBATCH --job-name=ignition_audit
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -e

#===============================================================================
# Environment, matching scripts/slurm_request.sh
#===============================================================================

module purge
module load cuda/13.0.2/none-none/a-bjlot5m

PYTHON="$HOME/.pyenv/versions/myenv/bin/python"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

#===============================================================================
# The rungs, and why these three
#===============================================================================
#
#   07  SenderGRULM   + BilinearDiscriminator   -- ignites at ~epoch 10
#   09  SenderTransformerLM + BilinearDiscriminator -- never ignites
#   13  SenderTransformerLM + AttentionDiscriminator -- ignites at ~epoch 20
#
# 13 is the control: same speaker as 09, different listener, opposite outcome.
# So a per-group gradient norm across the three separates "the speaker cannot
# use the signal" from "the listener never supplies one".

CONFIGS=(
    "07_shapeworld_sender_contrast"
    "09_shapeworld_sender_transformer_lm"
    "13_shapeworld_attention_discriminator"
)

STEPS="${STEPS:-1800}"
SEED="${SEED:-0}"
OUT="${OUT:-$PROJECT_DIR/results/ignition_audit}"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG_STEM="${CONFIGS[$INDEX]}"
CONFIG_PATH="$PROJECT_DIR/experiments/ablation/configs/${CONFIG_STEM}.toml"

LOG_DIR="$PROJECT_DIR/experiments/ablation/logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/ignition_audit_${SLURM_ARRAY_JOB_ID}_${INDEX}.o" 2>&1

echo "=============================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array task:   $INDEX"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "Config:       $CONFIG_PATH"
echo "Steps:        $STEPS  (~$((STEPS / 156)) epochs at 156 steps/epoch)"
echo "Seed:         $SEED"
echo "Out:          $OUT"
echo "=============================================="

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found: $CONFIG_PATH"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
    || echo "nvidia-smi not available"
echo ""

$PYTHON "$PROJECT_DIR/scripts/ignition_audit.py" \
    --configs "$CONFIG_PATH" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --out "$OUT" || EXIT_CODE=$?

EXIT_CODE="${EXIT_CODE:-0}"

echo ""
echo "=============================================="
echo "Finished at:  $(date)"
echo "Exit code:    $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE

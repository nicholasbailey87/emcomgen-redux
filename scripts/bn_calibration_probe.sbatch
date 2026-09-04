#!/bin/bash
# Run `diagnostics/bn_calibration_probe.py` on one GPU.
#
# Submit from the repository root, naming a trained run:
#
#     sbatch --export=ALL,CHECKPOINT=$HOME/sharedscratch/output/silhouette_titration/05_silhouette_0.5_seed0/checkpoint_last.pt,CONFIG=experiments/silhouette_titration/configs/05_silhouette_0.5.toml \
#         scripts/bn_calibration_probe.sbatch
#
# and override the sweep through the environment:
#
#     sbatch --export=ALL,CHECKPOINT=...,CONFIG=...,N=0,1,2,4,8,16,SEEDS=5 scripts/bn_calibration_probe.sbatch
#
# The table lands in results/bn_calibration_probe/probe_<jobid>.txt, which is
# also the job's stdout -- there is nothing else to collect.
#
# Unlike the silhouette probe this one trains a real epoch before it sweeps (the
# statistics have to be polluted before they can be corrected), so the walltime
# is one ShapeWorld epoch plus `2 x len(N) x SEEDS` eval passes. `--requeue` on
# the preemptible partition with no checkpointing: a preempted job starts again
# and overwrites its own file, because half a table is not a table.

#SBATCH --job-name=bn_calibration_probe
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=08:00:00
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

if [ -z "${CHECKPOINT:-}" ] || [ -z "${CONFIG:-}" ]; then
    echo "CHECKPOINT and CONFIG must both be set, e.g."
    echo "  sbatch --export=ALL,CHECKPOINT=<run>/checkpoint_last.pt,CONFIG=<run's toml> $0"
    exit 2
fi

N="${N:-0,1,2,4,8,16,32,64,128}"
SEEDS="${SEEDS:-3}"
SEED="${SEED:-0}"
SILHOUETTE="${SILHOUETTE:-0.5}"

OUT_DIR="${OUT:-$PROJECT_DIR/results/bn_calibration_probe}"
mkdir -p "$OUT_DIR"
exec > "$OUT_DIR/probe_${SLURM_JOB_ID}.txt" 2>&1

echo "=============================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "Checkpoint:   $CHECKPOINT"
echo "Config:       $CONFIG"
echo "Sweep:        N = $N | $SEEDS seeds | base seed $SEED"
echo "Silhouette:   p_receiver = $SILHOUETTE"
echo "=============================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
    || echo "nvidia-smi not available"
echo ""

$PYTHON "$PROJECT_DIR/diagnostics/bn_calibration_probe.py" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --n "$N" \
    --seeds "$SEEDS" \
    --seed "$SEED" \
    --silhouette "$SILHOUETTE" || EXIT_CODE=$?

EXIT_CODE="${EXIT_CODE:-0}"

echo ""
echo "=============================================="
echo "Finished at:  $(date)"
echo "Exit code:    $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE

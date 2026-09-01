#!/bin/bash
# Run `diagnostics/silhouette_shape_probe.py` on one GPU.
#
# Submit from the repository root:
#
#     sbatch scripts/silhouette_shape_probe.sbatch
#
# and override any of the probe's settings through the environment:
#
#     sbatch --export=ALL,GAMES=1200,EPOCHS=25,SEEDS=10 scripts/silhouette_shape_probe.sbatch
#
# The table lands in results/silhouette_shape_probe/probe_<jobid>.txt, which is
# also the job's stdout -- there is nothing else to collect.
#
# `--requeue` on a preemptible partition, and no checkpointing: forty fresh Conv4
# fits take about a quarter of an hour, so a preempted job simply starts again
# and overwrites its own file. Half a table is not a table.

#SBATCH --job-name=silhouette_probe
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=01:00:00
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

DATA="${DATA:-$HOME/sharedscratch/data/emcomgen/data/shapeworld_40}"
SPLIT="${SPLIT:-train}"
GAMES="${GAMES:-600}"
EPOCHS="${EPOCHS:-15}"
SEED="${SEED:-0}"
SEEDS="${SEEDS:-5}"

OUT_DIR="${OUT:-$PROJECT_DIR/results/silhouette_shape_probe}"
mkdir -p "$OUT_DIR"
exec > "$OUT_DIR/probe_${SLURM_JOB_ID}.txt" 2>&1

echo "=============================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "Data:         $DATA ($SPLIT)"
echo "Games:        $GAMES  | epochs $EPOCHS | split seed $SEED | $SEEDS fits"
echo "=============================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
    || echo "nvidia-smi not available"
echo ""

$PYTHON "$PROJECT_DIR/diagnostics/silhouette_shape_probe.py" \
    --data "$DATA" \
    --split "$SPLIT" \
    --games "$GAMES" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --seeds "$SEEDS" || EXIT_CODE=$?

EXIT_CODE="${EXIT_CODE:-0}"

echo ""
echo "=============================================="
echo "Finished at:  $(date)"
echo "Exit code:    $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE

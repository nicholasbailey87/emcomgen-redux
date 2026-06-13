#!/bin/bash
set -e

#SBATCH --job-name=experiment                      # Overridden by run_experiment.sh
#SBATCH --partition=gpu-a100                        # GPU partition
#SBATCH --nodes=1                                  # Run on 1 node
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=6                          # Overridden by run_experiment.sh
#SBATCH --mem=24GB                                 # Default; overridden by run_experiment.sh
#SBATCH --time=24:00:00                            # Overridden by run_experiment.sh
#SBATCH --gres=gpu:a100:1                           # Default; overridden by run_experiment.sh

# Note: Output/error logs are configured dynamically below since they depend on EXPERIMENT

#===============================================================================
# Environment Setup
#===============================================================================

# Remove any unwanted modules
module purge

# Load required modules (check available versions with: module avail)
module load cuda/13.0.2/none-none/a-bjlot5m

# Python from virtualenv
PYTHON="$HOME/.pyenv/versions/myenv/bin/python"

#===============================================================================
# Path Configuration
#===============================================================================

# Project directory (where sbatch was called from)
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

# Experiment name (passed via --export from run_experiment.sh)

# Verify required environment variables
if [ -z "$EXPERIMENT" ]; then
    echo "Error: EXPERIMENT environment variable not set"
    exit 1
fi

# Experiment paths
EXPERIMENT_DIR="$PROJECT_DIR/experiments/$EXPERIMENT"
LOG_DIR="$EXPERIMENT_DIR/logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Redirect output to experiment log directory
exec > "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.o" 2>&1

#===============================================================================
# Job Mapping - Get config and seed for this array task
#===============================================================================

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=============================================="
echo "Experiment: $EXPERIMENT"
echo "=============================================="

# Use job_utils.py to get config file and seed for this array index
JOB_INFO=$($PYTHON "$PROJECT_DIR/scripts/job_utils.py" --experiment "$EXPERIMENT" --index "$SLURM_ARRAY_TASK_ID")
CONFIG_FILE=$(echo "$JOB_INFO" | cut -d' ' -f1)
SEED=$(echo "$JOB_INFO" | cut -d' ' -f2)

CONFIG_PATH="$EXPERIMENT_DIR/configs/$CONFIG_FILE"

echo "Config file: $CONFIG_FILE"
echo "Seed: $SEED"
echo "Config path: $CONFIG_PATH"
echo "=============================================="

# Verify config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

#===============================================================================
# GPU Information
#===============================================================================

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "nvidia-smi not available"
echo ""

#===============================================================================
# Run Training
#===============================================================================

echo "Starting training..."
echo ""

$PYTHON "$PROJECT_DIR/code/train.py" \
    --config "$CONFIG_PATH" \
    --seed "$SEED" \
    $EXTRA_ARGS

EXIT_CODE=$?

#===============================================================================
# Completion
#===============================================================================

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE

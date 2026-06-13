#!/bin/bash
# Submit an array of SLURM requests constituting an experiment

set -e  # Exit on error

# Parse flags
RERUN=false
GPU_TYPE=""
POSITIONAL=()
while [ $# -gt 0 ]; do
    case $1 in
        --rerun) RERUN=true; shift ;;
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        --gpu-type=*) GPU_TYPE="${1#*=}"; shift ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

# Check arguments
if [ "${#POSITIONAL[@]}" -lt 1 ]; then
    echo "Usage: $0 [--rerun] <experiment_name> [max_concurrent] [partition]"
    echo ""
    echo "Arguments:"
    echo "  experiment_name   Name of the experiment (folder in experiments/)"
    echo "  max_concurrent    Max SLURM array tasks running at once (default: 5)"
    echo "  partition         SLURM partition (default: gpu-a100)"
    echo "  --rerun           Ignore completed jobs and rerun all from scratch"
    echo "  --gpu-type <name> Request a specific GPU GRES type (e.g. a100, a100_80g, rtx_8000)"
    echo ""
    echo "Example:"
    echo "  $0 transformer"
    echo "  $0 transformer 10"
    echo "  $0 transformer 5 preemptgpu"
    echo "  $0 --rerun transformer"
    exit 1
fi

EXPERIMENT="${POSITIONAL[0]}"
MAX_CONCURRENT="${POSITIONAL[1]:-5}"
PARTITION="${POSITIONAL[2]:-gpu-a100}"

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Python from virtualenv
PYTHON="$HOME/.pyenv/versions/myenv/bin/python"

# Read data paths from root config.json
DATA_SOURCE=$($PYTHON -c "import json; print(json.load(open('$PROJECT_DIR/config.json'))['data_slow_storage'])")
DATA_DEST=$($PYTHON -c "import json; print(json.load(open('$PROJECT_DIR/config.json'))['data_fast_storage'])")
OUTPUT_ROOT=$($PYTHON -c "import json; print(json.load(open('$PROJECT_DIR/config.json'))['output_root'])")
DATA_SOURCE="${DATA_SOURCE/#\~/$HOME}"
DATA_DEST="${DATA_DEST/#\~/$HOME}"

echo "=============================================="
echo "Experiment Runner"
echo "=============================================="
echo "Experiment:   $EXPERIMENT"
echo "Data source:  $DATA_SOURCE"
echo "Data dest:    $DATA_DEST"
echo "Project dir:  $PROJECT_DIR"
echo "=============================================="

# Verify experiment exists
EXPERIMENT_DIR="$PROJECT_DIR/experiments/$EXPERIMENT"
if [ ! -d "$EXPERIMENT_DIR/configs" ]; then
    echo "Error: Experiment configs not found: $EXPERIMENT_DIR/configs"
    exit 1
fi

# Calculate total jobs using job_utils.py
echo ""
echo "Calculating total jobs..."
TOTAL_JOBS=$($PYTHON "$SCRIPT_DIR/job_utils.py" --experiment "$EXPERIMENT" --count)
echo "Total jobs: $TOTAL_JOBS"

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "Error: No jobs found for experiment $EXPERIMENT"
    exit 1
fi

# Verify data is ready
echo ""
if [ ! -d "$DATA_DEST" ]; then
    echo "Error: Data not found at $DATA_DEST"
    echo "Run ./scripts/prepare_data.sh first."
    exit 1
else
    echo "Data found at $DATA_DEST"
fi

# Create experiment directories
echo ""
echo "Creating experiment directories..."
mkdir -p "$EXPERIMENT_DIR/logs"

# Read SLURM settings from first config (TOML)
FIRST_CONFIG=$(ls "$EXPERIMENT_DIR/configs/"*.toml | sort | head -1)
SLURM_TIME=$($PYTHON -c "import toml; print(toml.load('$FIRST_CONFIG').get('slurm', {}).get('time', '24:00:00'))")
SLURM_CPUS=$($PYTHON -c "import toml; print(toml.load('$FIRST_CONFIG').get('slurm', {}).get('cpus_per_task', 6))")
SLURM_GPUS=$($PYTHON -c "import toml; print(toml.load('$FIRST_CONFIG').get('slurm', {}).get('gpus_per_task', 1))")
SLURM_MEM=$($PYTHON -c "import toml; print(str(toml.load('$FIRST_CONFIG').get('slurm', {}).get('mem_gb', 24)) + 'GB')")
echo "SLURM time limit: $SLURM_TIME"
echo "SLURM CPUs per task: $SLURM_CPUS"
echo "SLURM GPUs per job: $SLURM_GPUS"
echo "SLURM memory: $SLURM_MEM"

# Determine which jobs to submit
cd "$PROJECT_DIR"  # Ensure we're in project root for job_utils.py

if [ "$RERUN" = true ]; then
    echo ""
    echo "Rerun mode: submitting all $TOTAL_JOBS jobs from scratch"
    ARRAY_SPEC="0-$((TOTAL_JOBS-1))"
else
    echo ""
    echo "Checking for completed jobs..."
    INCOMPLETE=$($PYTHON "$SCRIPT_DIR/job_utils.py" --experiment "$EXPERIMENT" --incomplete --output-root "$OUTPUT_ROOT")

    if [ -z "$INCOMPLETE" ]; then
        echo "All $TOTAL_JOBS jobs already complete. Nothing to submit."
        exit 0
    fi

    INCOMPLETE_COUNT=$(echo "$INCOMPLETE" | tr ',' '\n' | wc -l | tr -d ' ')
    echo "$INCOMPLETE_COUNT of $TOTAL_JOBS jobs incomplete: $INCOMPLETE"
    ARRAY_SPEC="$INCOMPLETE"
fi

if [ -n "$GPU_TYPE" ]; then
    GRES="gpu:${GPU_TYPE}:${SLURM_GPUS}"
else
    GRES="gpu:${SLURM_GPUS}"
fi

# Submit SLURM array job
echo ""
echo "Submitting SLURM array job (gres=$GRES)..."

EXTRA_ARGS=""
if [ "$RERUN" = true ]; then
    EXTRA_ARGS="--no_resume"
fi

sbatch --array="${ARRAY_SPEC}"%${MAX_CONCURRENT} \
    --job-name="$EXPERIMENT" \
    --time="$SLURM_TIME" \
    --cpus-per-task="$SLURM_CPUS" \
    --mem="$SLURM_MEM" \
    --partition="$PARTITION" \
    --gres=${GRES} \
    --export=ALL,EXPERIMENT="$EXPERIMENT",EXTRA_ARGS="$EXTRA_ARGS" \
    "$SCRIPT_DIR/slurm_request.sh"

echo ""
echo "=============================================="
echo "Job submitted successfully!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f experiments/$EXPERIMENT/logs/${EXPERIMENT}_*.o"

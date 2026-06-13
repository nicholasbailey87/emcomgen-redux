#!/bin/bash
# Prepare datasets on fast storage before running experiments.
# Run this once before submitting any jobs.
#
# Stages each emcomgen dataset from slow storage to fast storage:
#   <data_slow_storage>/emcomgen/data/<name>  ->  <data_fast_storage>/emcomgen/data/<name>
# for name in cub, shapeworld, shapeworld_ref. Datasets absent from slow
# storage are skipped with a message. (CUB expects the preprocessed img.npz per
# class already present in slow storage; this script copies whatever is there.)
#
# Usage: ./scripts/prepare_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$HOME/.pyenv/versions/myenv/bin/python"

# Read paths from config.json
DATA_SOURCE=$($PYTHON -c "import json; print(json.load(open('$PROJECT_DIR/config.json'))['data_slow_storage'])")
DATA_DEST=$($PYTHON -c "import json; print(json.load(open('$PROJECT_DIR/config.json'))['data_fast_storage'])")

# Expand ~ in paths
DATA_SOURCE="${DATA_SOURCE/#\~/$HOME}"
DATA_DEST="${DATA_DEST/#\~/$HOME}"

# emcomgen datasets live under <storage>/emcomgen/data/<name>
SRC_ROOT="$DATA_SOURCE/emcomgen/data"
DEST_ROOT="$DATA_DEST/emcomgen/data"

echo "=============================================="
echo "Data Preparation"
echo "=============================================="
echo "Source (archive):   $SRC_ROOT"
echo "Destination (fast): $DEST_ROOT"
echo "=============================================="

mkdir -p "$DEST_ROOT"

DATASETS=("cub" "shapeworld" "shapeworld_ref")

for name in "${DATASETS[@]}"; do
    echo ""
    echo "--- $name ---"
    SRC="$SRC_ROOT/$name"
    DEST="$DEST_ROOT/$name"

    if [ -d "$DEST" ]; then
        echo "Already on fast storage, skipping."
    elif [ -d "$SRC" ]; then
        echo "Copying from archive..."
        TEMP_DEST="$DEST_ROOT/${name}.tmp.$$"
        cp -r "$SRC" "$TEMP_DEST"
        mv "$TEMP_DEST" "$DEST" 2>/dev/null || rm -rf "$TEMP_DEST"
        echo "Done."
    else
        echo "Not found in archive ($SRC), skipping."
    fi
done

echo ""
echo "=============================================="
echo "Data preparation complete."
echo "=============================================="

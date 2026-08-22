"""
Resolve the shared storage paths declared in the repo-root ``config.json``.

See docs/training.md.
"""
import json
from pathlib import Path

# code/paths.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_JSON = REPO_ROOT / "config.json"


def load_storage_config() -> dict:
    """Load and ``~``-expand the storage paths from the repo-root config.json."""
    with open(CONFIG_JSON, "r") as f:
        cfg = json.load(f)
    return {k: Path(v).expanduser() for k, v in cfg.items()}


def data_fast_storage() -> Path:
    """Fast-storage data root (where ``prepare_data.sh`` stages datasets)."""
    return load_storage_config()["data_fast_storage"]


def output_root() -> Path:
    """Root under which results are written, arranged by experiment."""
    return load_storage_config()["output_root"]

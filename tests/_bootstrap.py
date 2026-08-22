"""
Puts `code/` on `sys.path`. Imported by `conftest.py` for pytest runs, and by
each test module directly so `python tests/test_<name>.py` still works.
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
)

from parse_config import get_config  # noqa: E402


CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "experiments",
    "ablation",
    "configs",
)


def rung(config_file):
    return os.path.join(CONFIG_DIR, config_file)


def config_section(section, config_file=None, **overrides):
    """Omit `config_file` for `DEFAULT.toml` alone."""
    settings = dict(get_config(config_file)[section])
    settings.update(overrides)
    return settings

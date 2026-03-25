"""QP — Analysis of quasi-periodic Alfvén waves in Saturn's magnetosphere."""

from pathlib import Path
import os

__version__ = "0.1.0"

# Data root: set QP_DATA_ROOT env var, or default to DATA/ sibling of project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("QP_DATA_ROOT", _PROJECT_ROOT / "DATA"))
DATA_PRODUCTS = DATA_ROOT / "CASSINI-DATA" / "DataProducts"
OUTPUT_DIR = _PROJECT_ROOT / "Output"

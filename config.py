from pathlib import Path

# Root directory of the project (wherever config.py lives)
ROOT_DIR = Path(__file__).resolve().parent

# Data directories
DATA_RAW_DIR   = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR  = ROOT_DIR / "data" / "processed"

# Final demand column suffixes
FD_CODES = {"HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"}

# Row labels that are not production nodes
NON_NODE_ROWS = {"VA", "TLS", "OUT"}

# Base year for analysis
BASE_YEAR = 2018

# Substitution efficiency (alpha in dynamic operator)
ALPHA = 0.30

# Materiality threshold for damping calculation
TAU = 0.01

def raw_data_path(year: int) -> Path:
    """Returns the path to the raw CSV for a given year."""
    return DATA_RAW_DIR / f"{year}_SML.csv"


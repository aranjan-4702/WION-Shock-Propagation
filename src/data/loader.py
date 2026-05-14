import pandas as pd
from pathlib import Path
from config import FD_CODES, NON_NODE_ROWS, raw_data_path


def load_icio(year: int = 2018) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load raw OECD ICIO table and return Z, F, X.

    Parameters
    ----------
    year : int
        The ICIO table year to load. Default 2018.

    Returns
    -------
    Z : pd.DataFrame  (n x n)
        Inter-industry flow matrix. Rows and columns are production nodes.
        Z_ij = value of inputs sector j purchases from sector i.

    F : pd.DataFrame  (n x 6)
        Final demand matrix. Rows are production nodes.
        Columns are FD categories: HFCE, NPISH, GGFC, GFCF, INVNT, DPABR.

    X : pd.Series  (n,)
        Total output vector. X_j = sum of row j across Z and F.
        This is computed, NOT taken from the OUT column in the CSV.
    """

    path = raw_data_path(year)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")

    # ── 1. Read raw table ─────────────────────────────────────────────────────
    raw = pd.read_csv(path, index_col=0)
    # Index is row labels (e.g. AGO_A01, VA, TLS, OUT)
    # Columns are column labels (same structure + FD columns)

    # ── 2. Identify production node rows ──────────────────────────────────────
    # A row is a production node if:
    #   - it is NOT in NON_NODE_ROWS (not VA, TLS, OUT)
    #   - it does NOT end with any FD suffix (not a final demand row)
    def is_production_node(label: str) -> bool:
        if label in NON_NODE_ROWS:
            return False
        if any(label.endswith(fd) for fd in FD_CODES):
            return False
        if label.endswith('_T'):
            return False
        return True

    prod_rows = [idx for idx in raw.index if is_production_node(idx)]

    # ── 3. Identify production node columns ───────────────────────────────────
    # Same logic applied to columns
    prod_cols = [col for col in raw.columns if is_production_node(col)]

    # ── 4. Identify final demand columns ─────────────────────────────────────
    # A column is a final demand column if it ends with a known FD suffix
    fd_cols = [col for col in raw.columns
               if any(col.endswith(fd) for col in [col] for fd in FD_CODES)]

    # cleaner rewrite of fd_cols
    fd_cols = [col for col in raw.columns
               if any(col.endswith(fd) for fd in FD_CODES)]

    # ── 5. Extract Z, F ───────────────────────────────────────────────────────
    Z = raw.loc[prod_rows, prod_cols].astype(float)
    F = raw.loc[prod_rows, fd_cols].astype(float)

    # ── 6. Compute X as row sum of Z and F ───────────────────────────────────
    # X_j = total output of sector j = what it sells to all industries + all FD
    X = Z.sum(axis=1) + F.sum(axis=1)
    X.name = "total_output"

    return Z, F, X
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy import linalg
from src.data.loader import load_icio


def build_matrices(year: int = 2018) -> tuple[
    pd.DataFrame,  # Z
    pd.DataFrame,  # F
    pd.Series,     # X
    pd.DataFrame,  # A
    pd.DataFrame,  # B
]:
    """
    Build all core economic matrices from raw ICIO data.

    Steps:
        1. Load Z, F, X via loader
        2. Drop zero-output sectors (X_j = 0)
        3. Compute A = Z / X  (technical coefficients)
        4. Compute B = (I - A)^{-1}  (Leontief inverse)

    Parameters
    ----------
    year : int
        ICIO table year to use.

    Returns
    -------
    Z : pd.DataFrame  (n x n)   inter-industry flows, zero-output nodes removed
    F : pd.DataFrame  (n x fd)  final demand, zero-output nodes removed
    X : pd.Series     (n,)      total output, zero-output nodes removed
    A : pd.DataFrame  (n x n)   technical coefficient matrix
    B : pd.DataFrame  (n x n)   Leontief inverse
    """

    # ── 1. Load ───────────────────────────────────────────────────────────────
    Z, F, X = load_icio(year)

    # ── 2. Remove zero-output sectors ────────────────────────────────────────
    # If X_j = 0, sector j produced nothing. A_ij = Z_ij / X_j is undefined.
    # These are genuine data gaps in ICIO, not modelling errors.
    active = X[X > 0].index

    Z = Z.loc[active, active]
    F = F.loc[active]
    X = X.loc[active]

    n = len(active)
    print(f"Active nodes after zero-output removal: {n}")

    # ── 3. Build A matrix ─────────────────────────────────────────────────────
    # A_ij = Z_ij / X_j
    # Each column j is divided by X_j (total output of sector j)
    # Column sum of A_j < 1 always, because X_j also includes value added
    A = Z.div(X, axis=1)   # axis=1 = divide each column by corresponding X_j

    # Sanity check: no A_ij should exceed 1
    assert A.values.max() <= 1.0 + 1e-9, "A_ij > 1 detected — check X computation"
    assert A.values.min() >= 0.0 - 1e-9, "Negative A_ij detected"

    # ── 4. Build Leontief inverse B = (I - A)^{-1} ───────────────────────────
    I = np.eye(n)
    B_values = linalg.inv(I - A.values)

    B = pd.DataFrame(B_values, index=active, columns=active)

    # Sanity check: diagonal of B must be >= 1 (each sector multiplies itself)
    #assert B.values.diagonal().min() >= 1.0 - 1e-9, "B diagonal < 1 detected"

    return Z, F, X, A, B
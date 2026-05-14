"""
src/dynamics/compute_damping.py
================================
Assembles the damping matrix D from V_j and phi_j.

Formula
-------
d_j = (1 - V_j) * phi_j
D   = diag(d_1, d_2, ..., d_n)

Economic meaning
----------------
d_j is sector j's absorption capacity — the fraction of incoming
shock that is absorbed rather than transmitted onward.

Two independent components:
    (1 - V_j) : structural substitutability
                High when j's supplier base is distributed globally.
                Low when j's suppliers are concentrated/irreplaceable.

    phi_j     : institutional capacity
                Can j execute supplier substitution if disrupted?
                High = strong institutions enable resilience (high LPI)
                Low  = weak institutions limit resilience (low LPI)

Multiplicative form justification:
    Absorption requires both substitutability and institutional capacity.
    If either is weak, absorption remains limited.

Damped dynamic model:
    x_{t+1} = A(I - D)x_t

    (I - D) scales column j by transmission factor (1 - d_j).
    High d_j means strong absorption in supplier j, so less shock is
    passed through that supplier column.
"""

import numpy as np
import pandas as pd


def compute_damping_vector(
    V         : pd.Series,
    phi_node  : pd.Series,
) -> pd.Series:
    """
    Compute d_j = (1 - V_j) * phi_j for each sector.

    Parameters
    ----------
    V        : pd.Series (n,)  upstream vulnerability [0,1]
    phi_node : pd.Series (n,)  institutional capacity [0,1]

    Returns
    -------
    d : pd.Series (n,)
        Damping coefficient per sector. Range [0, 1].
        High d_j = high absorption = lower shock transmission.
        Low d_j  = low absorption = higher shock transmission.
    """
    # Align indices — both should have same node labels
    V_aligned   = V.reindex(phi_node.index).fillna(0)
    phi_aligned = phi_node.reindex(V.index).fillna(0)

    # Core formula: absorption = structural substitutability × institutional capacity
    d = (1 - V_aligned) * phi_aligned
    d.name = 'damping'

    # Validate
    assert d.isna().sum() == 0, "NaN in d_j"
    assert d.min() >= 0,        "Negative d_j"
    assert d.max() <= 1 + 1e-9, "d_j > 1"

    print(f"Damping vector d_j computed:")
    print(f"  Mean   : {d.mean():.4f}")
    print(f"  Median : {d.median():.4f}")
    print(f"  Min    : {d.min():.4f}  ({d.idxmin()})")
    print(f"  Max    : {d.max():.4f}  ({d.idxmax()})")
    print(f"  Sectors with d_j > 0.4 : {(d > 0.4).sum()}")
    print(f"  Sectors with d_j < 0.2 : {(d < 0.2).sum()}")

    return d


def build_damping_matrix(d: pd.Series) -> np.ndarray:
    """
    Build diagonal damping matrix D = diag(d_1, ..., d_n).

    Parameters
    ----------
    d : pd.Series (n,)  damping coefficients

    Returns
    -------
    D : np.ndarray (n x n)
        Diagonal matrix. Off-diagonal entries are zero.
        D[j,j] = d_j for all j.
    """
    D = np.diag(d.values)

    assert D.shape == (len(d), len(d)), "D shape mismatch"
    assert np.allclose(np.diag(D), d.values), "Diagonal mismatch"

    print(f"Damping matrix D: {D.shape[0]} x {D.shape[1]}")
    print(f"  Diagonal mean  : {np.diag(D).mean():.4f}")
    print(f"  Diagonal range : [{np.diag(D).min():.4f}, {np.diag(D).max():.4f}]")

    return D


def compute_damping(
    V        : pd.Series,
    phi_node : pd.Series,
) -> tuple[pd.Series, np.ndarray]:
    """
    Master function: compute d_j and D from V_j and phi_j.

    Parameters
    ----------
    V        : pd.Series (n,)  upstream vulnerability
    phi_node : pd.Series (n,)  institutional capacity

    Returns
    -------
    d : pd.Series (n,)    damping coefficients
    D : np.ndarray (n×n)  diagonal damping matrix
    """
    print("=" * 60)
    print("Building Damping Matrix D")
    print("d_j = (1 - V_j) * phi_j")
    print("=" * 60)

    d = compute_damping_vector(V, phi_node)
    D = build_damping_matrix(d)

    print("\nD ready for damped simulation:")
    print("  x_{t+1} = A(I - D)x_t")
    print("=" * 60)

    return d, D

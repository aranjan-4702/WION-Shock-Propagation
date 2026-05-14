"""
src/dynamics/compute_vj.py
===========================
Computes V_j: Upstream Systemic Exposure for each sector j.

Economic meaning
----------------
V_j measures how much sector j's input basket is concentrated
on globally irreplaceable suppliers. High V_j = fragile.

Formula
-------
V_j = sum_i [ w_ij * PR_supply_i ]

where:
    w_ij        = share of sector j's inputs coming from sector i
                  = A_ij / sum_k(A_kj)  [input_dist: size-bias removed]

    PR_supply_i = PageRank of sector i on the SUPPLY network (input_dist^T)
                  Uses input_dist weights NOT raw A weights.
                  This ensures structural position drives scores, not size.

Why input_dist as PageRank weights (not raw A)?
------------------------------------------------
Raw A_ij is dominated by economic size. USA_M (real estate) has large
A values simply because it is a massive sector in dollar terms, not
because it is a structural supply bottleneck.

input_dist_ij = A_ij / col_sum_j removes size bias completely.
Every sector's outgoing weights sum to 1 regardless of economic scale.
PageRank then reflects pure structural dependency, not dollar flows.

Result: RUS_C19, ROW_B06, CHN_C26 score high because many sectors
genuinely depend on them for a large SHARE of their inputs —
not merely because they are large in absolute value.
"""

import numpy as np
import pandas as pd
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Block 1: Build input distribution matrix (w_ij)
# ─────────────────────────────────────────────────────────────────────────────

def compute_input_dist(A: pd.DataFrame) -> pd.DataFrame:
    """
    Compute input distribution matrix from A.

    input_dist_ij = A_ij / sum_k(A_kj)
                  = fraction of j's intermediate input purchases from i.

    Why column sum?
        Column j of A = all inputs purchased by j.
        Dividing by col_sum_j gives each supplier's SHARE of j's inputs.
        Each column sums to 1 — pure dependency share, no size effect.

    Parameters
    ----------
    A : pd.DataFrame (n x n)

    Returns
    -------
    input_dist : pd.DataFrame (n x n)
        Column j sums to 1. input_dist_ij = fraction of j's inputs from i.
    """
    col_sums = A.sum(axis=0)
    input_dist = A.div(col_sums, axis=1).fillna(0)

    col_sums_check = input_dist.sum(axis=0)
    non_unit = ((col_sums_check - 1).abs() > 1e-6) & (col_sums_check > 0)
    if non_unit.sum() > 0:
        print(f"  WARNING: {non_unit.sum()} columns don't sum to 1")

    print(f"  Input distribution: {input_dist.shape}")
    print(f"  Zero-input sectors: {(col_sums == 0).sum()}")

    return input_dist


# ─────────────────────────────────────────────────────────────────────────────
# Block 2: Supply-side PageRank on input_dist^T
# ─────────────────────────────────────────────────────────────────────────────

def compute_supply_pagerank(
    input_dist: pd.DataFrame,
    alpha: float = 0.85,
) -> pd.Series:
    """
    Compute PageRank on the SUPPLY network using input_dist as edge weights.

    Three key decisions:

    1. WHY TRANSPOSE?
       input_dist edge i→j = "i supplies to j".
       We want to rank suppliers = nodes most depended upon.
       Transposing: j→i = "j depends on i as supplier".
       PageRank on input_dist^T ranks nodes by how much others depend on them.

    2. WHY input_dist weights NOT raw A?
       Raw A is dominated by economic size → USA sectors inflate artificially.
       input_dist weights are pure dependency shares → structural position only.
       A sector supplying 40% of another's inputs scores the same whether
       it is a $1B or $1T sector — only the share matters.

    3. WHY alpha=0.85?
       Standard PageRank damping. 85% follow supply link, 15% random jump.
       Prevents score concentration in a single hub. Standard in literature.

    Parameters
    ----------
    input_dist : pd.DataFrame (n x n)
        Size-normalized input distribution. Each column sums to 1.
    alpha : float
        PageRank damping factor.

    Returns
    -------
    pr_supply : pd.Series (n,)  normalized to [0,1]
        Higher = more structurally critical as a global supplier.
    """
    # Transpose: edge direction reverses to "depends on" relationship
    input_dist_T = input_dist.T

    # Build directed graph with input_dist weights
    G = nx.from_pandas_adjacency(
        input_dist_T,
        create_using=nx.DiGraph()
    )

    # Weighted PageRank on supply network
    pr_dict = nx.pagerank(
        G,
        alpha=alpha,
        weight='weight',
        max_iter=1000,
        tol=1e-6
    )

    # Convert and align
    pr_supply = pd.Series(pr_dict, name='pr_supply')
    pr_supply = pr_supply.reindex(input_dist.index).fillna(0)

    # Normalize to [0,1] for interpretability
    pr_max = pr_supply.max()
    if pr_max > 0:
        pr_supply = pr_supply / pr_max
    pr_supply.name = 'pr_supply_normalized'

    print(f"  Top 10 most critical global suppliers:")
    print(pr_supply.nlargest(10).to_string())

    return pr_supply


# ─────────────────────────────────────────────────────────────────────────────
# Block 3: Compute V_j
# ─────────────────────────────────────────────────────────────────────────────

def compute_upstream_vulnerability(
    input_dist: pd.DataFrame,
    pr_supply: pd.Series,
) -> pd.Series:
    """
    Compute V_j = sum_i [ input_dist_ij * PR_supply_i ]

    In matrix form: V = input_dist.T @ pr_supply

    Step by step for sector j:
        1. Column j of input_dist = supplier shares (who supplies j and how much)
        2. PR_supply_i = how structurally critical is each supplier i
        3. Multiply: dependency share × criticality
        4. Sum = V_j = exposure-weighted average supplier criticality

    Why better than HHI:
        HHI = sum(w_ij^2) — measures concentration only (how many suppliers).
        V_j = sum(w_ij * PR_i) — measures systemic exposure (how critical).
        Sector with 10 equal suppliers but all are global hubs:
            HHI → low (looks resilient)
            V_j → high (correctly fragile)

    Parameters
    ----------
    input_dist : pd.DataFrame (n x n)
    pr_supply  : pd.Series (n,)  normalized to [0,1]

    Returns
    -------
    V : pd.Series (n,)  range [0,1]
    """
    pr_aligned = pr_supply.reindex(input_dist.index).fillna(0).values

    # Matrix multiply: (n×n).T @ (n,) → (n,)
    V_vals = input_dist.T.values @ pr_aligned
    V = pd.Series(V_vals, index=input_dist.columns, name='upstream_vulnerability')
    V = V.clip(0, 1)

    assert V.isna().sum() == 0, "NaN in V_j"
    assert V.min() >= 0,        "Negative V_j"
    assert V.max() <= 1 + 1e-9, "V_j > 1"

    print(f"\n  V_j statistics:")
    print(f"  Mean   : {V.mean():.4f}")
    print(f"  Median : {V.median():.4f}")
    print(f"  Min    : {V.min():.4f}  ({V.idxmin()})")
    print(f"  Max    : {V.max():.4f}  ({V.idxmax()})")
    print(f"\n  Top 10 most vulnerable sectors:")
    print(V.nlargest(10).to_string())
    print(f"\n  Top 10 least vulnerable sectors:")
    print(V.nsmallest(10).to_string())

    return V


# ─────────────────────────────────────────────────────────────────────────────
# Block 4: Master function
# ─────────────────────────────────────────────────────────────────────────────

def compute_Vj(
    A: pd.DataFrame,
    alpha: float = 0.85,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Master function: compute V_j from A matrix.

    Pipeline:
        A  →  input_dist (size-normalized)
           →  PR^supply on input_dist^T (structural criticality)
           →  V_j = Σ w_ij · PR^supply_i

    Parameters
    ----------
    A     : pd.DataFrame (n x n)  technical coefficient matrix
    alpha : float                  PageRank damping factor

    Returns
    -------
    V          : pd.Series (n,)      upstream vulnerability
    pr_supply  : pd.Series (n,)      supply PageRank scores
    input_dist : pd.DataFrame (n×n)  input distribution (reused for phi_j)
    """
    print("=" * 60)
    print("Computing V_j: Upstream Systemic Exposure")
    print("PageRank weights: input_dist (size-bias removed)")
    print("=" * 60)

    print("\n[1/3] Building input distribution matrix...")
    input_dist = compute_input_dist(A)

    print("\n[2/3] Computing supply PageRank on input_dist^T...")
    pr_supply = compute_supply_pagerank(input_dist, alpha=alpha)

    print("\n[3/3] Computing V_j = Σ w_ij · PR_supply_i ...")
    V = compute_upstream_vulnerability(input_dist, pr_supply)

    print("\n" + "=" * 60)
    print("V_j computation complete.")
    print("=" * 60)

    return V, pr_supply, input_dist

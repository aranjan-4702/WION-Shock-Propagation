"""
src/dynamics/compute_vj.py
===========================
Computes V_j: Upstream Systemic Exposure for each sector j.

Method: Option C — Global Supply Concentration (IRR-based)
----------------------------------------------------------
No PageRank. Instead we directly measure how irreplaceable
each product type is by computing its global supply concentration
across countries using a Herfindahl-Hirschman Index (HHI).

Economic meaning
----------------
V_j = input-share-weighted average global supply concentration
      of sector j's input basket.

Formula
-------
Step 1: w_ij      = A_ij / sum_k(A_kj)
        [fraction of j's intermediate inputs from supplier i]

Step 2: IRR_s     = sum_c [ (X_{c,s} / X_{total,s})^2 ]
        [global HHI of product s across countries c]
        [how concentrated is global production of product s?]

Step 3: IRR_i     = IRR_{sector(i)}
        [map product-level IRR to each node i]

Step 4: V_j       = sum_i [ w_ij * IRR_i ]
        [input-share-weighted average IRR of j's suppliers]

Why this solves the PageRank dominance problem
----------------------------------------------
PageRank rewards BREADTH — sectors with many buyers score high.
USA_M (real estate) has buyers in every sector → dominates PageRank.

IRR rewards CONCENTRATION — sectors where few countries dominate
global supply score high. Real estate exists in every country
(low IRR). Petroleum, semiconductors, rare metals are concentrated
in few countries (high IRR).

Finance and IT stay in the model — they just get weighted by their
actual global supply concentration. If semiconductor production is
concentrated in CHN/KOR/TWN, C26 gets high IRR and correctly
drives V_j up for sectors that depend on it.

Defense statement
-----------------
"V_j is the input-share-weighted global supply concentration of
sector j's input basket. For each product j purchases, we measure
how concentrated global production of that product is across
countries using a Herfindahl index. Products supplied by few
countries (petroleum, semiconductors) receive high concentration
scores. Products supplied by virtually all countries (real estate,
basic financial services) receive low scores. V_j aggregates
these across j's entire input basket — capturing true
irreplaceability rather than mere network centrality."
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Block 1: Input distribution matrix (unchanged from before)
# ─────────────────────────────────────────────────────────────────────────────

def compute_input_dist(A: pd.DataFrame) -> pd.DataFrame:
    """
    Compute input distribution matrix.

    w_ij = A_ij / sum_k(A_kj)
         = fraction of j's intermediate inputs from supplier i.

    Each column j sums to 1 — pure dependency share, no size effect.

    Why column sum?
        Column j of A = all inputs purchased by j.
        Dividing by col_sum_j gives each supplier's SHARE of j's inputs.
        Size-neutral — a $1B supplier providing 40% of j's inputs
        is treated identically to a $1T supplier providing 40%.

    Parameters
    ----------
    A : pd.DataFrame (n x n)  technical coefficient matrix

    Returns
    -------
    input_dist : pd.DataFrame (n x n)
        Column j sums to 1. w_ij = fraction of j's inputs from i.
    """
    col_sums = A.sum(axis=0)
    input_dist = A.div(col_sums, axis=1).fillna(0)

    # Validate
    col_check = input_dist.sum(axis=0)
    non_unit  = ((col_check - 1).abs() > 1e-6) & (col_check > 0)
    if non_unit.sum() > 0:
        print(f"  WARNING: {non_unit.sum()} columns don't sum to 1")

    print(f"  Input distribution : {input_dist.shape}")
    print(f"  Zero-input sectors : {(col_sums == 0).sum()}")

    return input_dist


# ─────────────────────────────────────────────────────────────────────────────
# Block 2: Compute IRR — Global Supply Concentration per sector code
# ─────────────────────────────────────────────────────────────────────────────

def compute_irr(
    X: pd.Series,
    nodes: pd.Index,
) -> pd.Series:
    """
    Compute IRR_s: global supply concentration for each sector code s.

    Formula:
        IRR_s = sum_c [ (X_{c,s} / X_{total,s})^2 ]

    This is the standard Herfindahl-Hirschman Index (HHI) applied
    at the GLOBAL PRODUCT LEVEL across countries.

    Key distinction from node-level HHI (what professor objected to):
        Node-level HHI: how concentrated are j's INPUT PURCHASES?
            → measures j's buying behavior
            → professor said not explainable enough
        Product-level IRR: how concentrated is GLOBAL SUPPLY of product s?
            → measures market structure of each product globally
            → directly measures irreplaceability
            → well established in industrial organization literature

    Interpretation:
        IRR_s = 1.0  → one country supplies everything (perfectly irreplaceable)
        IRR_s = 1/n  → n countries each supply equal share (perfectly substitutable)
        IRR_s ~ 0.15+ → highly concentrated (petroleum, semiconductors)
        IRR_s ~ 0.02  → widely distributed (real estate, basic services)

    Why this handles services correctly:
        Real estate (M): 81 countries all produce it, roughly proportional
                         to GDP → IRR_M is low → contributes little to V_j ✓
        Finance (K):     Similar story → low IRR_K ✓
        Petroleum (C19): RUS, USA, CHN dominate → high IRR_C19 ✓
        Semiconductors:  TWN, KOR, CHN dominate → high IRR_C26 ✓
        Oil extraction:  SAU, RUS, ROW dominate → high IRR_B06 ✓

    Parameters
    ----------
    X : pd.Series (n,)
        Total output per node. Index = node labels (e.g. 'RUS_C19').
    nodes : pd.Index
        All node labels (same as X.index).

    Returns
    -------
    IRR_node : pd.Series (n,)
        IRR value mapped to each NODE (not sector code).
        All nodes with the same sector code share the same IRR value.
        Range [1/n_countries, 1].
    """

    # Step 1: Parse node labels into country and sector code
    # 'RUS_C19' → country='RUS', sector_code='C19'
    # 'ROW_B06' → country='ROW', sector_code='B06'
    # 'CHN_C302T309' → country='CHN', sector_code='C302T309'
    df = pd.DataFrame({
        'node'        : nodes,
        'country'     : [n.split('_')[0] for n in nodes],
        'sector_code' : ['_'.join(n.split('_')[1:]) for n in nodes],
        'X'           : X.reindex(nodes).fillna(0).values,
    }, index=nodes)

    # Step 2: For each sector code, compute global total output
    # X_total_s = sum of X across all countries for sector code s
    sector_totals = df.groupby('sector_code')['X'].sum()
    df['X_total_s'] = df['sector_code'].map(sector_totals)

    # Step 3: Compute each node's share of global supply of its sector
    # share_{c,s} = X_{c,s} / X_total_s
    # Guard against division by zero (sector with zero global output)
    df['share'] = np.where(
        df['X_total_s'] > 0,
        df['X'] / df['X_total_s'],
        0.0
    )

    # Step 4: IRR_s = sum_c (share_{c,s})^2  — Herfindahl across countries
    # Group by sector code, sum squared shares
    df['share_sq'] = df['share'] ** 2
    irr_by_sector = df.groupby('sector_code')['share_sq'].sum()

    # Step 5: Map IRR back to each NODE
    # Every node with sector_code s gets IRR_s
    IRR_node = df['sector_code'].map(irr_by_sector)
    IRR_node.index = nodes
    IRR_node.name  = 'IRR'

    # Validate
    assert IRR_node.isna().sum() == 0, "NaN in IRR"
    assert IRR_node.min() >= 0,        "Negative IRR"
    assert IRR_node.max() <= 1 + 1e-9, "IRR > 1"

    # Report top and bottom sector codes (not nodes)
    print(f"\n  Global supply concentration IRR by sector code:")
    print(f"  Top 10 most concentrated (irreplaceable) products:")
    print(irr_by_sector.sort_values(ascending=False).head(10).to_string())
    print(f"\n  Bottom 10 least concentrated (most substitutable) products:")
    print(irr_by_sector.sort_values(ascending=True).head(10).to_string())
    print(f"\n  IRR range: [{IRR_node.min():.4f}, {IRR_node.max():.4f}]")
    print(f"  IRR mean : {IRR_node.mean():.4f}")

    return IRR_node, irr_by_sector


# ─────────────────────────────────────────────────────────────────────────────
# Block 3: Compute V_j
# ─────────────────────────────────────────────────────────────────────────────

def compute_upstream_vulnerability(
    input_dist : pd.DataFrame,
    IRR_node   : pd.Series,
) -> pd.Series:
    """
    Compute V_j = sum_i [ w_ij * IRR_i ]

    In matrix form: V = input_dist.T @ IRR_node

    Step by step for sector j:
        1. Column j of input_dist → supplier shares (who supplies j, how much)
        2. IRR_i → how concentrated is global supply of supplier i's product?
        3. Multiply: dependency share × irreplaceability
        4. Sum over all suppliers → V_j

    Concrete example:
        DEU_C24A buys:
            40% from RUS_C19 (IRR_C19 = 0.08) → contributes 0.032
            20% from USA_M   (IRR_M   = 0.03) → contributes 0.006
            15% from ROW_B06 (IRR_B06 = 0.12) → contributes 0.018
            ...
        V_j = 0.032 + 0.006 + 0.018 + ... = meaningful vulnerability score

        USA_M contributes little because real estate is globally distributed.
        RUS_C19 and ROW_B06 contribute more because energy is concentrated.

    Why this is better than PageRank for our purpose:
        PageRank: "how many buyers does i have?" → breadth measure
        IRR:      "how few countries supply i's product?" → concentration measure
        We want concentration (irreplaceability), not breadth.

    Parameters
    ----------
    input_dist : pd.DataFrame (n x n)  column j sums to 1
    IRR_node   : pd.Series (n,)        IRR value per node

    Returns
    -------
    V : pd.Series (n,)  range [0, 1]
        High V_j = input basket dominated by concentrated-supply products
        Low V_j  = input basket spread across globally distributed products
    """

    # Align IRR to input_dist index
    IRR_aligned = IRR_node.reindex(input_dist.index).fillna(0).values

    # Matrix multiply: input_dist.T (n×n) @ IRR (n,) → V (n,)
    # Row j of input_dist.T = column j of input_dist = supplier shares of j
    # Dot with IRR = weighted average IRR of j's suppliers
    V_vals = input_dist.T.values @ IRR_aligned

    V = pd.Series(V_vals, index=input_dist.columns, name='upstream_vulnerability')

    # Normalize to [0,1]
    # V is already in [0,1] since input_dist cols sum to 1 and IRR in [0,1]
    # but normalize by observed max to use full range
    v_max = V.max()
    if v_max > 0:
        V = V / v_max
    V = V.clip(0, 1)

    # Validate
    assert V.isna().sum() == 0,  "NaN in V_j"
    assert V.min()  >= 0,        "Negative V_j"
    assert V.max()  <= 1 + 1e-9, "V_j > 1"

    print(f"\n  Upstream Vulnerability V_j (IRR-weighted):")
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
    A     : pd.DataFrame,
    X     : pd.Series,
    alpha : float = 0.85,   # kept for API compatibility, not used in Option C
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Master function: compute V_j using Option C (Global Supply Concentration).

    Pipeline:
        A, X
         → input_dist   (size-normalized supplier shares)
         → IRR_node     (global supply concentration per product)
         → V_j          (input-share-weighted average IRR)

    Note: alpha parameter kept for API compatibility with old code
          but is not used — Option C does not use PageRank.

    Parameters
    ----------
    A     : pd.DataFrame (n x n)  technical coefficient matrix
    X     : pd.Series (n,)        total output per node
    alpha : float                 unused (kept for compatibility)

    Returns
    -------
    V            : pd.Series (n,)      upstream vulnerability [0,1]
    IRR_node     : pd.Series (n,)      IRR per node (for EDA/plotting)
    input_dist   : pd.DataFrame (n×n)  input distribution (reused for phi_j)
    """

    print("=" * 60)
    print("Computing V_j: Upstream Systemic Exposure")
    print("Method: Option C — Global Supply Concentration (IRR)")
    print("=" * 60)

    print("\n[1/3] Building input distribution matrix...")
    input_dist = compute_input_dist(A)

    print("\n[2/3] Computing IRR: global supply concentration per sector...")
    IRR_node, irr_by_sector = compute_irr(X, A.index)

    print("\n[3/3] Computing V_j = Σ w_ij · IRR_i ...")
    V = compute_upstream_vulnerability(input_dist, IRR_node)

    print("\n" + "=" * 60)
    print("V_j computation complete.")
    print("=" * 60)

    # Return IRR_node as second return value
    # (replaces pr_supply in old API — same position, different meaning)
    return V, IRR_node, input_dist
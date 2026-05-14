"""
src/dynamics/compute_vj.py
===========================
Computes V_j: Upstream Systemic Exposure for each sector j.

Method: Option C — Global Export Concentration (IRR-based)
----------------------------------------------------------
V_j measures the structural difficulty of replacing sector j's
input suppliers — a supply-side resilience metric.

Formula
-------
Step 1: w_ij    = A_ij / sum_k(A_kj)
        [fraction of j's intermediate inputs from supplier i]

Step 2: IRR_s   = sum_c [ (Xexp_{c,s} / Xexp_{total,s})^2 ]
        [global HHI of product s using EXPORT flows only]
        [Xexp_{c,s} = what country c sells internationally in sector s]

Step 3: IRR_i   = IRR_{sector(i)}
        [map product-level IRR to each node i]

Step 4: V_j     = sum_i [ w_ij * IRR_i ]
        [input-share-weighted average IRR of j's suppliers]

Why export flows, not total output?
------------------------------------
Total output X includes production consumed domestically —
which is never available to foreign buyers.

When a shock hits supplier i, sector j searches the GLOBAL
TRADED MARKET for alternatives. The relevant concentration
is therefore how much of internationally traded supply of
product s is controlled by each country — not total production.

Example: USA produces large amounts of petroleum (C19) but
consumes most domestically. Using total output understates
how concentrated the EXPORT market for petroleum is.
Using export flows correctly captures that RUS, SAU, ROW
dominate the internationally available petroleum supply.

Paper note
----------
"IRR is computed using export flows rather than total output,
as domestically consumed production is not available to foreign
buyers and therefore does not relieve global supply constraints
when a cross-border shock occurs."

Defense statement for V_j
--------------------------
"V_j is the input-share-weighted global export concentration
of sector j's input basket. For each product j purchases, we
measure how concentrated internationally traded supply of that
product is across countries using a Herfindahl index on export
flows. Products whose global exports are dominated by few
countries (petroleum exports, semiconductors, basic metals)
receive high concentration scores — reflecting true
irreplaceability in global markets. V_j aggregates these
across j's entire input basket weighted by input dependency
share, capturing the structural difficulty of supplier
substitution."
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Block 1: Input distribution matrix (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_input_dist(A: pd.DataFrame) -> pd.DataFrame:
    """
    Compute input distribution matrix.

    w_ij = A_ij / sum_k(A_kj)
         = fraction of j's intermediate inputs from supplier i.

    Each column j sums to 1 — pure dependency share, no size effect.

    Parameters
    ----------
    A : pd.DataFrame (n x n)  technical coefficient matrix

    Returns
    -------
    input_dist : pd.DataFrame (n x n)
        Column j sums to 1. w_ij = fraction of j's inputs from i.
    """
    col_sums   = A.sum(axis=0)
    input_dist = A.div(col_sums, axis=1).fillna(0)

    col_check = input_dist.sum(axis=0)
    non_unit  = ((col_check - 1).abs() > 1e-6) & (col_check > 0)
    if non_unit.sum() > 0:
        print(f"  WARNING: {non_unit.sum()} columns don't sum to 1")

    print(f"  Input distribution : {input_dist.shape}")
    print(f"  Zero-input sectors : {(col_sums == 0).sum()}")

    return input_dist


# ─────────────────────────────────────────────────────────────────────────────
# Block 2: IRR using EXPORT flows (key change from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def compute_irr(
    Z     : pd.DataFrame,
    nodes : pd.Index,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute IRR_s: global EXPORT concentration for each sector code s.

    Formula:
        Xexp_{c,s}      = sum_j Z_{i,j}  where country(j) != country(i)
                        = total exports of node i = (c, s) to all foreign buyers

        Xexp_total_s    = sum_c Xexp_{c,s}
                        = total international trade volume of product s

        IRR_s           = sum_c [ (Xexp_{c,s} / Xexp_total_s)^2 ]
                        = Herfindahl index of global EXPORT market for product s

    Why export flows not total output:
        Total output X includes domestic consumption —
        production that was never available to foreign buyers.
        When sector j loses a foreign supplier, it searches
        the internationally traded market for alternatives.
        IRR should reflect concentration of THAT market, not
        total production which includes non-tradeable domestic supply.

        Consequence:
        - C19 (petroleum refining): many countries refine domestically
          but RUS, SAU, ROW dominate EXPORTS → high export IRR ✓
        - M (real estate): purely domestic, near-zero exports
          → Xexp_M ≈ 0 → these nodes contribute nothing to IRR
          → correctly treated as non-global-suppliers ✓
        - C26 (semiconductors): CHN, KOR, TWN dominate exports → high IRR ✓

    Handling non-exporters:
        Nodes with zero exports (Xexp_i = 0) are purely domestic suppliers.
        They contribute zero to global export market concentration.
        They are assigned IRR = IRR_s of their sector code,
        which will be low if the sector is dominated by domestic producers.

    Parameters
    ----------
    Z     : pd.DataFrame (n x n)  inter-industry flow matrix (raw flows)
    nodes : pd.Index               all node labels

    Returns
    -------
    IRR_node      : pd.Series (n,)   IRR mapped to each node
    irr_by_sector : pd.Series        IRR per sector code (for EDA/plotting)
    """

    # Step 1: Parse node labels
    df = pd.DataFrame({
        'node'        : nodes,
        'country'     : [n.split('_')[0] for n in nodes],
        'sector_code' : ['_'.join(n.split('_')[1:]) for n in nodes],
    }, index=nodes)

    # Step 2: Compute export flows for each node
    # Xexp_i = sum of Z_ij where country(j) != country(i)
    # i.e. sum of all flows from node i to nodes in OTHER countries

    # For each node i, we need sum of Z[i, j] where country(j) != country(i)
    # Strategy: compute total row sum of Z, subtract domestic flows

    # Total row sum (all flows from i)
    Z_rowsum = Z.sum(axis=1)  # shape (n,)

    # Domestic flows from i = sum of Z[i, j] where country(j) = country(i)
    # Build a mask: True where supplier country = buyer country
    # For each row i, sum only columns j where country(j) = country(i)

    print(f"  Computing export flows (cross-border only)...")
    print(f"  This may take a moment for {len(nodes)} nodes...")

    # Efficient approach: group columns by country, sum per supplier
    # Add country info to Z columns
    col_countries = pd.Series(
        [c.split('_')[0] for c in Z.columns],
        index=Z.columns
    )

    # For each node i, domestic flow = sum of Z[i, j] where country(j) = country(i)
    domestic_flows = pd.Series(0.0, index=nodes)

    for country, group_rows in Z.T.groupby(col_countries):
    # group_rows is a submatrix where all COLUMNS are from this country
    # (because we transposed — rows are now columns)
        group_cols = group_rows.T  # transpose back → rows=suppliers, cols=this country's buyers
        country_nodes = df[df['country'] == country].index
        if len(country_nodes) > 0 and len(group_cols.columns) > 0:
            domestic_flows.loc[country_nodes] = (
                group_cols.loc[country_nodes].sum(axis=1).values
            )

    # Export flows = total row sum - domestic flows
    export_flows = Z_rowsum - domestic_flows
    export_flows = export_flows.clip(lower=0)  # guard against float negatives

    df['Xexp'] = export_flows.reindex(nodes).fillna(0).values

    n_exporters   = (df['Xexp'] > 0).sum()
    n_nonexporter = (df['Xexp'] == 0).sum()
    print(f"  Nodes with positive exports : {n_exporters}")
    print(f"  Non-exporting nodes         : {n_nonexporter}")
    print(f"  Total export volume         : {df['Xexp'].sum():.2f}")

    # Step 3: Compute IRR_s = HHI of export market shares per sector code
    sector_export_totals = df.groupby('sector_code')['Xexp'].sum()
    df['Xexp_total_s']   = df['sector_code'].map(sector_export_totals)

    # Export market share of each node within its sector
    df['export_share'] = np.where(
        df['Xexp_total_s'] > 0,
        df['Xexp'] / df['Xexp_total_s'],
        0.0
    )

    # IRR_s = sum of squared export shares across countries
    df['share_sq']    = df['export_share'] ** 2
    irr_by_sector     = df.groupby('sector_code')['share_sq'].sum()

    # Step 4: Map IRR back to each node
    IRR_node       = df['sector_code'].map(irr_by_sector)
    IRR_node.index = nodes
    IRR_node.name  = 'IRR_export'

    # Validate
    assert IRR_node.isna().sum() == 0, "NaN in IRR"
    assert IRR_node.min()  >= 0,       "Negative IRR"
    assert IRR_node.max()  <= 1 + 1e-9,"IRR > 1"

    print(f"\n  Global EXPORT concentration IRR by sector code:")
    print(f"  Top 10 most concentrated (irreplaceable export markets):")
    print(irr_by_sector.sort_values(ascending=False).head(10).to_string())
    print(f"\n  Bottom 10 least concentrated (substitutable export markets):")
    print(irr_by_sector.sort_values(ascending=True).head(10).to_string())
    print(f"\n  IRR range : [{IRR_node.min():.4f}, {IRR_node.max():.4f}]")
    print(f"  IRR mean  : {IRR_node.mean():.4f}")

    return IRR_node, irr_by_sector


# ─────────────────────────────────────────────────────────────────────────────
# Block 3: Compute V_j (unchanged formula, new IRR input)
# ─────────────────────────────────────────────────────────────────────────────

def compute_upstream_vulnerability(
    input_dist : pd.DataFrame,
    IRR_node   : pd.Series,
) -> pd.Series:
    """
    Compute V_j = sum_i [ w_ij * IRR_i ]

    In matrix form: V = input_dist.T @ IRR_node

    For sector j:
        1. Column j of input_dist → supplier shares (w_ij)
        2. IRR_i → how concentrated is EXPORT market of supplier i's product?
        3. Multiply: dependency share × export market concentration
        4. Sum → V_j = structural difficulty of replacing j's suppliers

    Parameters
    ----------
    input_dist : pd.DataFrame (n x n)  column j sums to 1
    IRR_node   : pd.Series (n,)        export IRR per node

    Returns
    -------
    V : pd.Series (n,)  normalized to [0,1]
        High V_j = suppliers dominate global export markets = hard to replace
        Low V_j  = suppliers compete in distributed export markets = substitutable
    """
    IRR_aligned = IRR_node.reindex(input_dist.index).fillna(0).values

    # Matrix multiply: input_dist.T (n×n) @ IRR (n,) → V (n,)
    V_vals = input_dist.T.values @ IRR_aligned
    V      = pd.Series(V_vals, index=input_dist.columns,
                       name='upstream_vulnerability')

    # Normalize to [0,1] using observed max
    v_max = V.max()
    if v_max > 0:
        V = V / v_max
    V = V.clip(0, 1)

    assert V.isna().sum() == 0,  "NaN in V_j"
    assert V.min()  >= 0,        "Negative V_j"
    assert V.max()  <= 1 + 1e-9, "V_j > 1"

    print(f"\n  Upstream Vulnerability V_j (export IRR-weighted):")
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
    X     : pd.Series,           # kept for API compatibility, not used
    Z     : pd.DataFrame = None, # raw flow matrix — required for export IRR
    alpha : float = 0.85,        # kept for API compatibility, not used
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Master function: compute V_j using export-based IRR.

    Pipeline:
        A, Z
         → input_dist   (size-normalized supplier shares from A)
         → IRR_node     (global export concentration per product from Z)
         → V_j          (input-share-weighted average export IRR)

    Parameters
    ----------
    A     : pd.DataFrame (n x n)  technical coefficient matrix
    X     : pd.Series (n,)        total output (kept for compatibility)
    Z     : pd.DataFrame (n x n)  raw inter-industry flow matrix (required)
    alpha : float                  unused

    Returns
    -------
    V          : pd.Series (n,)      upstream vulnerability [0,1]
    IRR_node   : pd.Series (n,)      export IRR per node (for EDA)
    input_dist : pd.DataFrame (n×n)  input distribution (reused for phi_j)
    """
    if Z is None:
        raise ValueError(
            "Z (raw flow matrix) is required for export-based IRR. "
            "Pass Z from build_matrices(): V, IRR, dist = compute_Vj(A, X, Z=Z)"
        )

    print("=" * 60)
    print("Computing V_j: Upstream Systemic Exposure")
    print("Method: Option C — Global Export Concentration (IRR)")
    print("=" * 60)

    print("\n[1/3] Building input distribution matrix...")
    input_dist = compute_input_dist(A)

    print("\n[2/3] Computing IRR: global export concentration per sector...")
    IRR_node, irr_by_sector = compute_irr(Z, A.index)

    print("\n[3/3] Computing V_j = Σ w_ij · IRR_i ...")
    V = compute_upstream_vulnerability(input_dist, IRR_node)

    print("\n" + "=" * 60)
    print("V_j computation complete.")
    print("=" * 60)

    return V, IRR_node, input_dist
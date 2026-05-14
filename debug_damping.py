"""
Diagnostic script for damping behavior.

Model under test:
    d_j = (1 - V_j) * phi_j
    x_{t+1} = A(I-D)x_t

Important metric:
    Compare shock-induced losses within each regime against that regime's own
    no-shock trajectory (apples-to-apples).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from src.network.builder import build_matrices
from src.dynamics.compute_vj import compute_Vj
from src.dynamics.compute_phi import compute_phi
from src.dynamics.compute_damping import compute_damping


def simulate_shock_losses(A_vals: np.ndarray, D: np.ndarray, x0: np.ndarray, shock_pos: int, steps: int = 5):
    """Return baseline and damped shock-induced losses for t=0..steps."""
    n = len(x0)
    I = np.eye(n)
    A_damp = A_vals @ (I - D)

    x0_shocked = x0.copy()
    x0_shocked[shock_pos] = 0.0

    traj_base = np.zeros((steps + 1, n))
    traj_damp = np.zeros((steps + 1, n))
    traj_base_ref = np.zeros((steps + 1, n))
    traj_damp_ref = np.zeros((steps + 1, n))

    traj_base[0] = x0_shocked
    traj_damp[0] = x0_shocked
    traj_base_ref[0] = x0
    traj_damp_ref[0] = x0

    for t in range(steps):
        traj_base[t + 1] = A_vals @ traj_base[t]
        traj_damp[t + 1] = A_damp @ traj_damp[t]
        traj_base_ref[t + 1] = A_vals @ traj_base_ref[t]
        traj_damp_ref[t + 1] = A_damp @ traj_damp_ref[t]

    loss_base = np.array([(traj_base_ref[t] - traj_base[t]).sum() for t in range(steps + 1)])
    loss_damp = np.array([(traj_damp_ref[t] - traj_damp[t]).sum() for t in range(steps + 1)])
    return loss_base, loss_damp


def run_node_test(node_label: str, node_name: str, A: pd.DataFrame, V: pd.Series, phi_node: pd.Series, D: np.ndarray, x0: np.ndarray, steps: int = 5):
    """Run one node shock test and return a summary row."""
    V_indexed = V.reindex(A.index).fillna(0)
    phi_indexed = phi_node.reindex(A.index).fillna(0)
    d_indexed = pd.Series(np.diag(D), index=A.index)

    pos = A.index.get_loc(node_name)
    loss_base, loss_damp = simulate_shock_losses(A.values, D, x0, pos, steps=steps)

    print(f"\nNode [{node_label}]: {node_name} (index {pos})")
    print(f"  V_j: {V_indexed[node_name]:.4f}")
    print(f"  phi_j: {phi_indexed[node_name]:.4f}")
    print(f"  d_j = (1-V_j)*phi_j: {d_indexed[node_name]:.4f}")
    print(f"  1-d_j (transmission): {1.0 - d_indexed[node_name]:.4f}")

    one_step_base = loss_base[1]
    one_step_damp = loss_damp[1]
    one_step_improve = (one_step_base - one_step_damp) / one_step_base * 100 if one_step_base != 0 else 0.0

    print("  One-step shock-induced loss:")
    print(f"    Baseline: {one_step_base:,.0f}")
    print(f"    Damped:   {one_step_damp:,.0f}")
    print(f"    Improve:  {one_step_improve:.1f}%")

    print(f"  {'Iter':<6} {'Baseline':<18} {'Damped':<18} {'Ratio':<10} {'Benefit':<10}")
    print("  " + "-" * 64)
    for t in range(steps + 1):
        ratio = loss_damp[t] / loss_base[t] if loss_base[t] > 0 else 1.0
        benefit = (1.0 - ratio) * 100.0 if ratio < 1.0 else -((ratio - 1.0) * 100.0)
        print(f"  {t:<6} {loss_base[t]:<18,.0f} {loss_damp[t]:<18,.0f} {ratio:<10.4f} {benefit:>8.1f}%")

    never_worse = all(ld <= lb for ld, lb in zip(loss_damp[1:], loss_base[1:]))
    print("  [OK] Damped is never worse for t>=1" if never_worse else "  [WARN] Some steps show damped > baseline")

    return {
        "case": node_label,
        "node": node_name,
        "d_j": float(d_indexed[node_name]),
        "one_step_improve_pct": float(one_step_improve),
        "five_step_ratio": float(loss_damp[steps] / loss_base[steps]) if loss_base[steps] > 0 else 1.0,
        "never_worse": bool(never_worse),
    }


def main():
    print("\n" + "=" * 70)
    print("BUILDING DAMPING MATRIX")
    print("=" * 70)

    Z, F, X, A, B = build_matrices(2018)
    V, IRR_node, input_dist = compute_Vj(A, X, Z=Z)
    phi_node, phi_country = compute_phi(
        lpi_path="data/raw/lpi.xlsx",
        nodes=A.index,
        year=2018,
    )
    d, D = compute_damping(V, phi_node)

    D_diag = np.diag(D)
    I_minus_D_diag = 1.0 - D_diag

    print("\n" + "=" * 70)
    print("D MATRIX ANALYSIS")
    print("=" * 70)
    print("d_j = (1 - V_j) * phi_j (ABSORPTION weight)")
    print("\nD diagonal (d_j) statistics:")
    print(f"  Min:    {D_diag.min():.6f}")
    print(f"  Max:    {D_diag.max():.6f}")
    print(f"  Mean:   {D_diag.mean():.6f}")
    print(f"  Median: {np.median(D_diag):.6f}")
    print(f"  Std:    {D_diag.std():.6f}")

    print("\n(1-d_j) transmission factors:")
    print(f"  Min:    {I_minus_D_diag.min():.6f}")
    print(f"  Max:    {I_minus_D_diag.max():.6f}")
    print(f"  Mean:   {I_minus_D_diag.mean():.6f}")

    print("\nDistribution of d_j:")
    print(f"  # with d > 0.2: {(D_diag > 0.2).sum()}")
    print(f"  # with d > 0.3: {(D_diag > 0.3).sum()}")
    print(f"  # with d > 0.4: {(D_diag > 0.4).sum()}")
    print(f"  # with d = 0:   {(D_diag == 0).sum()}")

    print("\n" + "=" * 70)
    print("NETWORK STABILITY (SPECTRAL RADIUS)")
    print("=" * 70)
    spec_rad_base = np.max(np.abs(np.linalg.eigvals(A.values)))
    spec_rad_damp = np.max(np.abs(np.linalg.eigvals(A.values @ (np.eye(len(D)) - D))))
    print(f"Baseline A:        {spec_rad_base:.6f}")
    print(f"A(I-D):            {spec_rad_damp:.6f}")
    print(f"Ratio (damp/base): {spec_rad_damp / spec_rad_base:.6f}")
    print("[OK] Damped dynamic is more stable" if spec_rad_damp < spec_rad_base else "[WARN] Damped dynamic is less stable")

    print("\n" + "=" * 70)
    print("SHOCK PROPAGATION TESTS (APPLES-TO-APPLES)")
    print("=" * 70)

    V_indexed = V.reindex(A.index).fillna(0)
    d_indexed = pd.Series(D_diag, index=A.index)
    x0 = X.values.astype(float)

    test_nodes = [
        ("max_V", V_indexed.idxmax()),
        ("max_d", d_indexed.idxmax()),
    ]
    if "RUS_C19" in A.index:
        test_nodes.append(("RUS_C19", "RUS_C19"))

    summary_rows = []
    for label, node in test_nodes:
        row = run_node_test(label, node, A, V, phi_node, D, x0, steps=5)
        summary_rows.append(row)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    if summary_df["never_worse"].all():
        print("\n[OK] Damped is not worse on tested nodes under apples-to-apples loss.")
    else:
        print("\n[WARN] At least one tested node still shows worse damped loss.")


if __name__ == "__main__":
    main()

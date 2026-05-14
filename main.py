import numpy as np
import pandas as pd
from src.network.builder import build_matrices

# ── Load all matrices ─────────────────────────────────────────────────────────
Z, F, X, A, B = build_matrices(2018)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("=" * 50)
print("MATRIX SHAPES")
print("=" * 50)
print(f"Z : {Z.shape}")
print(f"F : {F.shape}")
print(f"X : {X.shape}")
print(f"A : {A.shape}")
print(f"B : {B.shape}")

print()
print("=" * 50)
print("A MATRIX STATISTICS")
print("=" * 50)
print(f"A max             : {A.values.max():.4f}")
print(f"A min (non-zero)  : {A.values[A.values > 0].min():.6f}")
print(f"A col sums mean   : {A.sum(axis=0).mean():.4f}")
print(f"A col sums max    : {A.sum(axis=0).max():.4f}")
print(f"A col sums min    : {A.sum(axis=0).min():.4f}")
print(f"Sparsity          : {(A.values == 0).sum() / A.size:.4f}")

print()
print("=" * 50)
print("B MATRIX STATISTICS")
print("=" * 50)
print(f"B diagonal min    : {B.values.diagonal().min():.4f}")
print(f"B diagonal mean   : {B.values.diagonal().mean():.4f}")
print(f"B diagonal max    : {B.values.diagonal().max():.4f}")
print(f"Negative in B     : {(B.values < 0).any()}")



print()
print("=" * 50)
print("INVESTIGATING ANOMALIES")
print("=" * 50)

# Column sums of A
col_sums = A.sum(axis=0)

# Sectors where column sum > 1
over_unity = col_sums[col_sums > 1.0]
print(f"\nSectors with col sum > 1: {len(over_unity)}")
print(over_unity.sort_values(ascending=False).head(10))

# Sectors where column sum = 0
zero_cols = col_sums[col_sums == 0.0]
print(f"\nSectors with col sum = 0: {len(zero_cols)}")
print(zero_cols.head(10))

# For over-unity sectors: compare Z row sum vs X
print("\nFor top over-unity sector:")
top = over_unity.idxmax()
print(f"  Sector       : {top}")
print(f"  Z col sum    : {Z[top].sum():.2f}")
print(f"  F row sum    : {F.loc[top].sum():.2f}")
print(f"  X value      : {X[top]:.2f}")
print(f"  Z row sum    : {Z.loc[top].sum():.2f}")
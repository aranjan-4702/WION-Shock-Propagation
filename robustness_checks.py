"""
Robustness checks for the damped shock model.

This script runs two defenses for the thesis results:
1. Damping-scale sensitivity sweep
2. Bootstrap-style perturbations of the technical coefficient matrix A

Outputs are written to outputs/ and outputs/figures/.

The model convention is the current shock-only damping formulation:
    x_{t+1} = baseline_{t+1} - A(I-D) loss_t
where loss_t = baseline_t - x_t.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.network.builder import build_matrices
from src.dynamics.compute_vj import compute_Vj
from src.dynamics.compute_phi import compute_phi
from src.dynamics.compute_damping import compute_damping
from src.simulation.simulator import ShockSimulator


sns.set_theme(style="whitegrid", context="talk")


def ensure_output_dirs() -> tuple[Path, Path]:
    output_dir = ROOT_DIR / "outputs"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, figure_dir


def build_base_model(year: int, n_iter: int):
    Z, F, X, A, B = build_matrices(year)
    V, IRR_node, input_dist = compute_Vj(A, X, Z=Z)
    phi_node, phi_country = compute_phi(
        lpi_path=str(ROOT_DIR / "data" / "raw" / "lpi.xlsx"),
        nodes=A.index,
        year=year,
    )
    d, D = compute_damping(V, phi_node)
    sim = ShockSimulator(A=A, X=X, D=D, n_iter=n_iter)
    return {
        "Z": Z,
        "F": F,
        "X": X,
        "A": A,
        "B": B,
        "V": V,
        "IRR_node": IRR_node,
        "input_dist": input_dist,
        "phi_node": phi_node,
        "phi_country": phi_country,
        "d": d,
        "D": D,
        "sim": sim,
    }


def country_from_index(index: pd.Index) -> pd.Series:
    return pd.Series([node.split("_")[0] for node in index], index=index, name="country")


def aggregate_by_country(series: pd.Series) -> pd.Series:
    countries = country_from_index(series.index)
    return series.groupby(countries).mean().sort_values(ascending=False)


def select_reference_shocks(A: pd.DataFrame, V: pd.Series, d: pd.Series, phi_node: pd.Series) -> list[tuple[str, str, str]]:
    candidates = [
        ("max_V", V.idxmax(), "single_node"),
        ("max_d", d.idxmax(), "single_node"),
        ("max_phi", phi_node.idxmax(), "single_node"),
    ]
    if "RUS_C19" in A.index:
        candidates.append(("RUS_C19", "RUS_C19", "single_node"))
    if "USA_C19" in A.index:
        candidates.append(("USA_C19", "USA_C19", "single_node"))

    seen = set()
    unique = []
    for label, shock, mode in candidates:
        key = (shock, mode)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, shock, mode))
    return unique


def summarize_result(result: dict) -> dict:
    rl_base = result["RL_final_base"].copy().drop(index=result["shock_nodes"], errors="ignore")
    rl_damp = result["RL_final_damp"].copy().drop(index=result["shock_nodes"], errors="ignore")
    delta = result["delta_RL"].copy().drop(index=result["shock_nodes"], errors="ignore")

    country_base = aggregate_by_country(rl_base)
    country_damp = aggregate_by_country(rl_damp)
    country_delta = country_base.reindex(country_damp.index) - country_damp

    top_country = country_delta.idxmax()
    top_country_value = float(country_delta.max())

    return {
        "label": result["label"],
        "mode": result["mode"],
        "shock_nodes": len(result["shock_nodes"]),
        "mean_rl_base": float(rl_base.mean()),
        "mean_rl_damp": float(rl_damp.mean()),
        "mean_delta_rl": float(delta.mean()),
        "median_delta_rl": float(delta.median()),
        "max_delta_rl": float(delta.max()),
        "top_country_delta": top_country,
        "top_country_delta_value": top_country_value,
        "country_delta_mean": float(country_delta.mean()),
    }


def run_scenarios(sim: ShockSimulator, shocks: Iterable[tuple[str, str, str]]) -> tuple[dict, pd.DataFrame]:
    all_results: dict[str, dict] = {}
    rows = []
    for label, shock, mode in shocks:
        result = sim.run(shock=shock, mode=mode, label=label)
        all_results[label] = result
        rows.append(summarize_result(result))
    summary = pd.DataFrame(rows)
    return all_results, summary


def scaled_damping_matrix(D: np.ndarray, scale: float) -> np.ndarray:
    diag = np.clip(np.diag(D) * scale, 0.0, 0.999)
    return np.diag(diag)


def perturb_technical_matrix(A: pd.DataFrame, rng: np.random.Generator, sigma: float) -> pd.DataFrame:
    noise = rng.normal(loc=0.0, scale=sigma, size=A.shape)
    perturbed = A.values * np.exp(noise)
    perturbed = np.clip(perturbed, 0.0, None)

    col_sums = perturbed.sum(axis=0)
    target_sums = np.minimum(A.sum(axis=0).values, 0.999)
    scale = np.divide(target_sums, col_sums, out=np.ones_like(target_sums), where=col_sums > 0)
    perturbed = perturbed * scale

    return pd.DataFrame(perturbed, index=A.index, columns=A.columns)


def build_bootstrap_model(base: dict, A_boot: pd.DataFrame):
    X = base["X"]
    Z_boot = A_boot.mul(X, axis=1)
    V_boot, IRR_node_boot, input_dist_boot = compute_Vj(A_boot, X, Z=Z_boot)
    phi_node_boot, phi_country_boot = compute_phi(
        lpi_path=str(ROOT_DIR / "data" / "raw" / "lpi.xlsx"),
        nodes=A_boot.index,
        year=2018,
    )
    d_boot, D_boot = compute_damping(V_boot, phi_node_boot)
    sim_boot = ShockSimulator(A=A_boot, X=X, D=D_boot, n_iter=base["sim"].n_iter)
    return {
        "Z": Z_boot,
        "A": A_boot,
        "V": V_boot,
        "IRR_node": IRR_node_boot,
        "input_dist": input_dist_boot,
        "phi_node": phi_node_boot,
        "phi_country": phi_country_boot,
        "d": d_boot,
        "D": D_boot,
        "sim": sim_boot,
    }


def run_scale_sweep(base: dict, shocks: list[tuple[str, str, str]], scales: list[float], output_dir: Path, figure_dir: Path) -> pd.DataFrame:
    rows = []
    for scale in scales:
        D_scaled = scaled_damping_matrix(base["D"], scale)
        sim_scaled = ShockSimulator(A=base["A"], X=base["X"], D=D_scaled, n_iter=base["sim"].n_iter)
        for label, shock, mode in shocks:
            result = sim_scaled.run(shock=shock, mode=mode, label=label)
            rows.append({
                "scale": scale,
                **summarize_result(result),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "robustness_scale_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, grp in summary.groupby("label"):
        grp = grp.sort_values("scale")
        ax.plot(grp["scale"], grp["mean_delta_rl"], marker="o", linewidth=2, label=label)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Sensitivity of delta_RL to damping scale")
    ax.set_xlabel("Damping scale")
    ax.set_ylabel("Mean delta_RL")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "robustness_scale_sweep.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return summary


def run_bootstrap(base: dict, shocks: list[tuple[str, str, str]], n_boot: int, sigma: float, seed: int, output_dir: Path, figure_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    draw_rows = []
    summary_rows = []

    for boot_id in range(n_boot):
        A_boot = perturb_technical_matrix(base["A"], rng=rng, sigma=sigma)
        boot_model = build_bootstrap_model(base, A_boot)
        sim_boot = boot_model["sim"]

        for label, shock, mode in shocks:
            result = sim_boot.run(shock=shock, mode=mode, label=label)
            base_summary = summarize_result(result)
            draw_rows.append({
                "boot_id": boot_id,
                "label": label,
                "mean_delta_rl": base_summary["mean_delta_rl"],
                "median_delta_rl": base_summary["median_delta_rl"],
                "top_country_delta_value": base_summary["top_country_delta_value"],
                "country_delta_mean": base_summary["country_delta_mean"],
            })

    draws = pd.DataFrame(draw_rows)
    for label, grp in draws.groupby("label"):
        summary_rows.append({
            "label": label,
            "n_boot": int(len(grp)),
            "mean_delta_rl": float(grp["mean_delta_rl"].mean()),
            "std_delta_rl": float(grp["mean_delta_rl"].std(ddof=1)),
            "p05_delta_rl": float(grp["mean_delta_rl"].quantile(0.05)),
            "p95_delta_rl": float(grp["mean_delta_rl"].quantile(0.95)),
            "mean_top_country_delta": float(grp["top_country_delta_value"].mean()),
            "mean_country_delta": float(grp["country_delta_mean"].mean()),
        })

    summary = pd.DataFrame(summary_rows)
    draws.to_csv(output_dir / "robustness_bootstrap_draws.csv", index=False)
    summary.to_csv(output_dir / "robustness_bootstrap_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=draws, x="label", y="mean_delta_rl", ax=ax, color="#9ecae1")
    ax.set_title(f"Bootstrap distribution of mean delta_RL (sigma={sigma:.3f})")
    ax.set_xlabel("Shock scenario")
    ax.set_ylabel("Mean delta_RL")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_dir / "robustness_bootstrap_boxplot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return draws, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness checks for the damped shock model.")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--scales", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--n-bootstrap", type=int, default=25)
    parser.add_argument("--bootstrap-sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    output_dir, figure_dir = ensure_output_dirs()
    base = build_base_model(year=args.year, n_iter=args.n_iter)
    shocks = select_reference_shocks(base["A"], base["V"], base["d"], base["phi_node"])

    print("=" * 72)
    print("Running baseline robustness scenarios")
    print("=" * 72)
    all_results, scenario_summary = run_scenarios(base["sim"], shocks)
    scenario_summary.to_csv(output_dir / "robustness_scenario_summary.csv", index=False)
    print(scenario_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("Running damping scale sweep")
    print("=" * 72)
    scale_summary = run_scale_sweep(base, shocks, args.scales, output_dir, figure_dir)
    print(scale_summary.head().to_string(index=False))

    if not args.skip_bootstrap:
        print("\n" + "=" * 72)
        print("Running bootstrap perturbations")
        print("=" * 72)
        draws, bootstrap_summary = run_bootstrap(
            base=base,
            shocks=shocks,
            n_boot=args.n_bootstrap,
            sigma=args.bootstrap_sigma,
            seed=args.seed,
            output_dir=output_dir,
            figure_dir=figure_dir,
        )
        print(bootstrap_summary.to_string(index=False))
    else:
        draws = pd.DataFrame()
        bootstrap_summary = pd.DataFrame()

    manifest = pd.DataFrame([
        {"file": "robustness_scenario_summary.csv", "rows": len(scenario_summary)},
        {"file": "robustness_scale_sweep.csv", "rows": len(scale_summary)},
        {"file": "robustness_bootstrap_draws.csv", "rows": len(draws)},
        {"file": "robustness_bootstrap_summary.csv", "rows": len(bootstrap_summary)},
    ])
    manifest.to_csv(output_dir / "robustness_manifest.csv", index=False)
    print("\nSaved outputs to:")
    print(f"  {output_dir}")
    if not args.skip_plots:
        print(f"  {figure_dir}")


if __name__ == "__main__":
    main()

"""
src/simulation/simulator.py  [v2 — with damped model]
=======================================================
Shock propagation simulator supporting both:
    Baseline : x_{t+1} = A x_t
    Damped   : x_{t+1} = x_{t+1}^{base} - A(I - D) loss_t

The damped model introduces node-heterogeneous damping.
Each supplier sector j attenuates propagated loss by (1 - d_j),
so damping acts on shock transmission rather than total output.

Running both modes for the same shock gives:
    RL_baseline : worst-case frictionless propagation
    RL_damped   : propagation with structural + institutional friction
    RL_damped_common : damped propagation normalized by baseline no-shock
    delta_RL    : RL_baseline - RL_damped = damping effect of D
    delta_RL_common : RL_baseline - RL_damped_common (common denominator)

delta_RL is your main thesis result — it shows how much
your damping model reduces losses, and where in the network
the reduction is largest.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union


# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse shock nodes
# ─────────────────────────────────────────────────────────────────────────────

def _parse_shock_nodes(
    shock     : Union[str, list],
    mode      : str,
    all_nodes : pd.Index,
) -> list:
    """Return list of node labels to zero out."""
    if mode == "single_node":
        assert isinstance(shock, str), "single_node expects a string"
        assert shock in all_nodes, f"Node '{shock}' not found"
        return [shock]
    elif mode == "single_country":
        assert isinstance(shock, str), "single_country expects ISO3 string"
        nodes = [n for n in all_nodes if n.split('_')[0] == shock]
        assert nodes, f"No nodes found for country '{shock}'"
        return nodes
    elif mode == "multi_node":
        if isinstance(shock, str):
            shock = [shock]
        missing = [n for n in shock if n not in all_nodes]
        assert not missing, f"Nodes not found: {missing}"
        return shock
    else:
        raise ValueError(f"Unknown mode '{mode}'")


# ─────────────────────────────────────────────────────────────────────────────
# Main simulator class
# ─────────────────────────────────────────────────────────────────────────────

class ShockSimulator:
    """
    Iterative shock propagation supporting baseline and damped models.

    Parameters
    ----------
    A      : pd.DataFrame (n x n)  technical coefficient matrix
    X      : pd.Series (n,)        total output vector (baseline t=0)
    D      : np.ndarray (n x n)    diagonal damping matrix (optional)
             If None, only baseline mode is available.
    n_iter : int                   number of time steps (default 10)
    phi    : float                 shock intensity (1.0 = complete loss)
    """

    def __init__(
        self,
        A      : pd.DataFrame,
        X      : pd.Series,
        D      : np.ndarray = None,
        f      : np.ndarray = None,
        n_iter : int   = 10,
        phi    : float = 1.0,
    ):
        self.A      = A
        self.A_vals = A.values
        self.X0     = X.copy()
        self.nodes  = A.index
        self.n      = len(self.nodes)
        self.n_iter = n_iter
        self.phi    = phi

        # Damping matrix
        self.D = D
        if D is not None:
            # Pre-compute A(I - D) for efficiency
            # This scales each COLUMN j of A by (1-d_j)
            # Sector j with high d_j has fewer input needs (can substitute)
            I = np.eye(self.n)
            self.IminusD_A = self.A_vals @ (I - D)
            print(f"Damped model ready: A(I-D) computed")
            print(f"  Mean diagonal of D : {np.diag(D).mean():.4f}")
        else:
            self.IminusD_A = None

        # Legacy field kept for API compatibility; unused in current dynamic.
        self.f = f if f is not None else np.zeros(self.n)

    # ── Core iteration ────────────────────────────────────────────────────────

    def _iterate_baseline(
        self,
        x0          : np.ndarray,
        shocked_idx : list,
    ) -> np.ndarray:
        """
        Run baseline: x_{t+1} = A x_t
        Shocked nodes zeroed every step (persistent failure).
        """
        traj    = np.zeros((self.n_iter + 1, self.n))
        traj[0] = x0.copy()
        for t in range(self.n_iter):
            x_next = self.A_vals @ traj[t]
            x_next[shocked_idx] = 0.0
            traj[t + 1] = x_next
        return traj

    def _iterate_damped(
        self,
        x0          : np.ndarray,
        shocked_idx : list,
        baseline_ref: np.ndarray = None,
    ) -> np.ndarray:
        """
        Run damped with shock-only attenuation.
        Shocked nodes are zeroed every step.

        Let loss_t = baseline_t - x_t. Then:
            x_{t+1} = baseline_{t+1} - A(I-D)loss_t

        This means damping is applied to the propagated shock (loss),
        not to the full production level.
        """
        assert self.IminusD_A is not None, \
            "Damped model requires D. Pass D to ShockSimulator()."

        if baseline_ref is None:
            baseline_ref = np.zeros((self.n_iter + 1, self.n))
            baseline_ref[0] = x0.copy()
            for t in range(self.n_iter):
                baseline_ref[t + 1] = self.A_vals @ baseline_ref[t]

        traj    = np.zeros((self.n_iter + 1, self.n))
        traj[0] = x0.copy()

        for t in range(self.n_iter):
            # Propagate only the loss component, then subtract from no-shock path.
            loss_t = np.maximum(baseline_ref[t] - traj[t], 0.0)
            propagated_loss_next = self.IminusD_A @ loss_t
            x_next = baseline_ref[t + 1] - propagated_loss_next

            # Keep states numerically stable and economically meaningful.
            x_next = np.clip(x_next, 0.0, baseline_ref[t + 1])
            x_next[shocked_idx] = 0.0
            traj[t + 1] = x_next

        return traj

    # ── Relative Loss ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_rl(
        baseline : np.ndarray,
        shocked  : np.ndarray,
        nodes    : pd.Index,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Compute RL = (baseline - shocked) / baseline at each time step.
        Returns RL DataFrame (n_iter+1 x n) and final-step RL Series.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            RL_vals = np.where(
                baseline > 0,
                (baseline - shocked) / baseline,
                0.0
            )
        RL       = pd.DataFrame(RL_vals, columns=nodes)
        RL_final = RL.iloc[-1]
        return RL, RL_final

    # ── Main run method ───────────────────────────────────────────────────────

    def run(
        self,
        shock : Union[str, list],
        mode  : str  = "single_node",
        label : str  = None,
    ) -> dict:
        """
        Run baseline AND damped simulation for a given shock.
        Returns both results plus delta_RL = RL_baseline - RL_damped.

        Parameters
        ----------
        shock : str or list   node(s) to shock
        mode  : str           'single_node', 'single_country', 'multi_node'
        label : str           human-readable label for plots

        Returns
        -------
        dict with keys:
            shock_nodes   : list of zeroed nodes
            label         : str
            mode          : str
            baseline      : np.ndarray (n_iter+1, n)
            shocked_base  : np.ndarray (n_iter+1, n)  baseline shocked
            shocked_damp  : np.ndarray (n_iter+1, n)  damped shocked (if D)
            baseline_damp : np.ndarray (n_iter+1, n)  damped unshocked (if D)
            RL_base       : pd.DataFrame (n_iter+1, n)
            RL_damp       : pd.DataFrame (n_iter+1, n)  (if D)
            RL_damp_common: pd.DataFrame (n_iter+1, n)  (if D)
            RL_final_base : pd.Series (n,)
            RL_final_damp : pd.Series (n,)  (if D)
            RL_final_damp_common : pd.Series (n,)  (if D)
            delta_RL      : pd.Series (n,)  base - damp (if D)
            delta_RL_common : pd.Series (n,) base - damp_common (if D)
        """
        shock_nodes = _parse_shock_nodes(shock, mode, self.nodes)
        shocked_idx = [self.nodes.get_loc(n) for n in shock_nodes]

        if label is None:
            label = shock if isinstance(shock, str) else f"{len(shock_nodes)} nodes"

        print(f"\n{'='*60}")
        print(f"Shock   : {label}")
        print(f"Mode    : {mode} | Nodes: {len(shock_nodes)}")
        print(f"{'='*60}")

        x0 = self.X0.values.astype(float)

        # ── Unshocked baseline (no shock at all) ──────────────────────────────
        baseline = np.zeros((self.n_iter + 1, self.n))
        baseline[0] = x0.copy()
        for t in range(self.n_iter):
            baseline[t + 1] = self.A_vals @ baseline[t]

        # ── Baseline shocked ──────────────────────────────────────────────────
        x0_shocked = x0.copy()
        x0_shocked[shocked_idx] = 0.0
        shocked_base = self._iterate_baseline(x0_shocked, shocked_idx)

        RL_base, RL_final_base = self._compute_rl(
            baseline, shocked_base, self.nodes
        )

        print(f"Baseline — nodes RL>10%: {(RL_final_base > 0.1).sum()} | "
              f"Max RL: {RL_final_base.max():.4f} ({RL_final_base.idxmax()})")

        result = {
            'shock_nodes'   : shock_nodes,
            'label'         : label,
            'mode'          : mode,
            'baseline'      : baseline,
            'shocked_base'  : shocked_base,
            'RL_base'       : RL_base,
            'RL_final_base' : RL_final_base,
            # damped fields default to None
            'baseline_damp' : None,
            'shocked_damp'  : None,
            'RL_damp'       : None,
            'RL_damp_common': None,
            'RL_final_damp' : None,
            'RL_final_damp_common' : None,
            'delta_RL'      : None,
            'delta_RL_common': None,
        }

        # ── Damped shocked (if D provided) ────────────────────────────────────
        if self.D is not None:
            # Use a like-for-like damped no-shock reference. Comparing
            # damped shocked to baseline (A-only) overstates damped RL.
            baseline_damp = self._iterate_damped(
                x0.copy(), [], baseline_ref=baseline
            )
            shocked_damp = self._iterate_damped(
                x0_shocked, shocked_idx, baseline_ref=baseline
            )

            RL_damp, RL_final_damp = self._compute_rl(
                baseline_damp, shocked_damp, self.nodes
            )

            # Common-reference damped RL: compare damped shocked path to the
            # same no-shock baseline used by RL_base.
            RL_damp_common, RL_final_damp_common = self._compute_rl(
                baseline, shocked_damp, self.nodes
            )

            delta_RL = RL_final_base - RL_final_damp
            delta_RL.name = 'delta_RL'

            delta_RL_common = RL_final_base - RL_final_damp_common
            delta_RL_common.name = 'delta_RL_common'

            print(f"Damped  — nodes RL>10%: {(RL_final_damp > 0.1).sum()} | "
                  f"Max RL: {RL_final_damp.max():.4f} ({RL_final_damp.idxmax()})")
            print(f"Delta   — mean absorption: {delta_RL.mean():.4f} | "
                  f"Max absorption: {delta_RL.max():.4f} ({delta_RL.idxmax()})")
            print(f"Damped* — common-ref RL>10%: {(RL_final_damp_common > 0.1).sum()} | "
                f"Max RL: {RL_final_damp_common.max():.4f} ({RL_final_damp_common.idxmax()})")
            print(f"Delta*  — common-ref mean absorption: {delta_RL_common.mean():.4f} | "
                f"Max absorption: {delta_RL_common.max():.4f} ({delta_RL_common.idxmax()})")

            result.update({
                'baseline_damp' : baseline_damp,
                'shocked_damp'  : shocked_damp,
                'RL_damp'       : RL_damp,
                'RL_damp_common': RL_damp_common,
                'RL_final_damp' : RL_final_damp,
                'RL_final_damp_common' : RL_final_damp_common,
                'delta_RL'      : delta_RL,
                'delta_RL_common': delta_RL_common,
            })

        return result

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_comparison(
        self,
        results         : dict,
        top_n           : int  = 15,
        exclude_shocked : bool = True,
        figsize         : tuple = (16, 12),
        save_path       : str  = None,
    ):
        """
        Four-panel comparison plot:
        Top-left  : Baseline RL time series
        Top-right : Damped RL time series
        Bottom-left : Final-state RL bar (baseline vs damped)
        Bottom-right: delta_RL bar (absorption effect)
        """
        label       = results['label']
        shock_nodes = results['shock_nodes']
        RL_base     = results['RL_base'].copy()
        RL_damp     = results['RL_damp']
        RL_damp_common = results.get('RL_damp_common')
        delta_RL    = results['delta_RL']
        delta_RL_common = results.get('delta_RL_common')

        if exclude_shocked:
            RL_base = RL_base.drop(columns=shock_nodes, errors='ignore')
            if RL_damp is not None:
                RL_damp = RL_damp.drop(columns=shock_nodes, errors='ignore')
            if RL_damp_common is not None:
                RL_damp_common = RL_damp_common.drop(columns=shock_nodes, errors='ignore')

        top_nodes = RL_base.iloc[-1].nlargest(top_n).index

        has_damp = RL_damp is not None
        n_rows   = 2 if has_damp else 1
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, 2)

        colors = plt.cm.tab20.colors

        # Panel 1: Baseline time series
        ax = axes[0, 0]
        for i, node in enumerate(top_nodes):
            ax.plot(range(self.n_iter + 1),
                    RL_base[node],
                    color=colors[i % 20], linewidth=1.5, label=node)
        ax.set_title(f'Baseline RL over time\n{label}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Loss (RL)')
        ax.legend(fontsize=6, loc='upper left',
                  bbox_to_anchor=(1, 1), ncol=1)
        ax.grid(True, linestyle='--', alpha=0.4)

        # Panel 2: Damped time series
        if has_damp:
            ax = axes[0, 1]
            for i, node in enumerate(top_nodes):
                if node in RL_damp.columns:
                    ax.plot(range(self.n_iter + 1),
                            RL_damp[node],
                            color=colors[i % 20], linewidth=1.5,
                            label=node)
                if RL_damp_common is not None and node in RL_damp_common.columns:
                    ax.plot(range(self.n_iter + 1),
                            RL_damp_common[node],
                            color=colors[i % 20], linewidth=1.0,
                            linestyle='--')
            ax.set_title(f'Damped RL over time\n{label}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Relative Loss (RL)')
            ax.text(
                0.01, 0.01,
                'solid: damped ref | dashed: common ref',
                transform=ax.transAxes, fontsize=7,
                va='bottom', ha='left',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'}
            )
            ax.legend(fontsize=6, loc='upper left',
                      bbox_to_anchor=(1, 1), ncol=1)
            ax.grid(True, linestyle='--', alpha=0.4)

            # Panel 3: Side-by-side RL comparison at t_end
            ax = axes[1, 0]
            rl_b  = results['RL_final_base'].drop(
                index=shock_nodes, errors='ignore').nlargest(20)
            rl_d  = results['RL_final_damp'].drop(
                index=shock_nodes, errors='ignore').reindex(rl_b.index)
            rl_dc = results.get('RL_final_damp_common')
            if rl_dc is not None:
                rl_dc = rl_dc.drop(
                    index=shock_nodes, errors='ignore').reindex(rl_b.index)
            x     = np.arange(len(rl_b))
            has_common = rl_dc is not None
            width = 0.27 if has_common else 0.35
            if has_common:
                ax.bar(x - width, rl_b.values, width,
                       label='Baseline', color='#e74c3c', alpha=0.8)
                ax.bar(x, rl_d.values, width,
                       label='Damped (own ref)', color='#2ecc71', alpha=0.8)
                ax.bar(x + width, rl_dc.values, width,
                       label='Damped (common ref)', color='#1f77b4', alpha=0.8)
            else:
                ax.bar(x - width/2, rl_b.values, width,
                       label='Baseline', color='#e74c3c', alpha=0.8)
                ax.bar(x + width/2, rl_d.values, width,
                       label='Damped', color='#2ecc71', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(rl_b.index, rotation=45,
                               ha='right', fontsize=7)
            ax.set_ylabel('Final RL')
            ax.set_title('Baseline vs Damped RL at t_end\n(Top 20 most affected)',
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.4)

            # Panel 4: delta_RL (absorption effect)
            ax = axes[1, 1]
            delta = delta_RL.drop(
                index=shock_nodes, errors='ignore').nlargest(20)
            delta_common = None
            if delta_RL_common is not None:
                delta_common = delta_RL_common.drop(
                    index=shock_nodes, errors='ignore').reindex(delta.index)
            colors_d = plt.cm.YlGn(delta.values / delta.max())
            ax.barh(range(len(delta)), delta.values,
                    color=colors_d, edgecolor='white')
            if delta_common is not None:
                ax.scatter(
                    delta_common.values,
                    range(len(delta_common)),
                    color='#1f77b4', s=18,
                    label='Common-ref delta'
                )
            ax.set_yticks(range(len(delta)))
            ax.set_yticklabels(delta.index, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel('delta_RL = RL_base - RL_damp')
            ax.set_title('Absorption Effect (delta_RL)\nWhich nodes benefit most from damping?',
                         fontsize=10, fontweight='bold')
            if delta_common is not None:
                ax.legend(fontsize=8, loc='lower right')
            ax.grid(axis='x', linestyle='--', alpha=0.4)

        plt.suptitle(
            f'Baseline vs Damped Shock Propagation\nShock: {label}',
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
        return fig

    def plot_rl_heatmap(
        self,
        results         : dict,
        mode            : str  = 'base',
        top_n_nodes     : int  = 30,
        exclude_shocked : bool = True,
        figsize         : tuple = (16, 10),
        save_path       : str  = None,
    ):
        """
        Heatmap of final-state RL: countries x sectors.
        mode: 'base', 'damp', 'damp_common', 'delta', or 'delta_common'
        """
        if mode == 'base':
            RL_final = results['RL_final_base'].copy()
            title_suffix = 'Baseline'
            cmap = 'YlOrRd'
        elif mode == 'damp':
            RL_final = results['RL_final_damp'].copy()
            title_suffix = 'Damped'
            cmap = 'YlOrRd'
        elif mode == 'damp_common':
            RL_final = results['RL_final_damp_common'].copy()
            title_suffix = 'Damped (Common Reference)'
            cmap = 'YlOrRd'
        elif mode == 'delta':
            RL_final = results['delta_RL'].copy()
            title_suffix = 'Absorption Effect (delta_RL)'
            cmap = 'YlGn'
        elif mode == 'delta_common':
            RL_final = results['delta_RL_common'].copy()
            title_suffix = 'Absorption Effect (delta_RL_common)'
            cmap = 'YlGn'
        else:
            raise ValueError(
                "mode must be 'base', 'damp', 'damp_common', 'delta', or 'delta_common'"
            )

        if exclude_shocked:
            RL_final = RL_final.drop(
                index=results['shock_nodes'], errors='ignore')

        idx       = RL_final.index
        countries = idx.map(lambda x: x.split('_')[0])
        sectors   = idx.map(lambda x: '_'.join(x.split('_')[1:]))

        df = pd.DataFrame({
            'country': countries,
            'sector' : sectors,
            'RL'     : RL_final.values,
        })

        pivot = df.pivot_table(
            index='country', columns='sector',
            values='RL', aggfunc='mean'
        ).fillna(0)

        top_sectors = pivot.mean(axis=0).nlargest(top_n_nodes).index
        pivot       = pivot[top_sectors]
        active      = pivot.index[pivot.max(axis=1) > 0.005]
        pivot       = pivot.loc[active]
        pivot       = pivot.loc[
            pivot.sum(axis=1).sort_values(ascending=False).index
        ]

        # Skip plotting if pivot is empty after filtering
        if pivot.empty:
            print(f'  ⚠️  No data to plot for {title_suffix} heatmap (all values below threshold)')
            return None

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot, ax=ax, cmap=cmap,
            vmin=0, vmax=min(1.0, pivot.values.max()),
            linewidths=0.3, linecolor='#e0e0e0',
            cbar_kws={'label': f'RL ({title_suffix})', 'shrink': 0.6},
        )
        ax.set_title(
            f'Final-State RL Heatmap — {title_suffix}\n'
            f'Shock: {results["label"]}',
            fontsize=12, fontweight='bold', pad=10
        )
        ax.set_xlabel('Sector')
        ax.set_ylabel('Country')
        ax.tick_params(axis='x', labelsize=7, rotation=90)
        ax.tick_params(axis='y', labelsize=7, rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_country_bar(
        self,
        results         : dict,
        top_n           : int  = 20,
        exclude_shocked : bool = True,
        figsize         : tuple = (14, 6),
        save_path       : str  = None,
    ):
        """
        Side-by-side country bar: baseline vs damped mean RL per country.
        """
        rl_b = results['RL_final_base'].copy()
        rl_d = results['RL_final_damp'].copy() \
               if results['RL_final_damp'] is not None else None
        rl_dc = results.get('RL_final_damp_common')
        rl_dc = rl_dc.copy() if rl_dc is not None else None

        if exclude_shocked:
            shock_nodes = results['shock_nodes']
            rl_b = rl_b.drop(index=shock_nodes, errors='ignore')
            if rl_d is not None:
                rl_d = rl_d.drop(index=shock_nodes, errors='ignore')
            if rl_dc is not None:
                rl_dc = rl_dc.drop(index=shock_nodes, errors='ignore')

        # Aggregate by country
        countries_b = rl_b.index.map(lambda x: x.split('_')[0])
        country_rl_b = rl_b.groupby(countries_b).mean() \
                           .sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        x     = np.arange(len(country_rl_b))
        has_common = rl_dc is not None
        width = 0.27 if has_common else 0.35

        if rl_d is not None and has_common:
            ax.bar(x - width,
                   country_rl_b.values, width,
                   label='Baseline', color='#e74c3c', alpha=0.85)
        else:
            ax.bar(x - width/2 if rl_d is not None else x,
                   country_rl_b.values, width if rl_d is not None else 0.6,
                   label='Baseline', color='#e74c3c', alpha=0.85)

        if rl_d is not None:
            countries_d  = rl_d.index.map(lambda x: x.split('_')[0])
            country_rl_d = rl_d.groupby(countries_d).mean() \
                               .reindex(country_rl_b.index).fillna(0)
            if has_common:
                ax.bar(x, country_rl_d.values, width,
                       label='Damped (own ref)', color='#2ecc71', alpha=0.85)
            else:
                ax.bar(x + width/2, country_rl_d.values, width,
                       label='Damped', color='#2ecc71', alpha=0.85)

        if has_common:
            countries_dc = rl_dc.index.map(lambda x: x.split('_')[0])
            country_rl_dc = rl_dc.groupby(countries_dc).mean() \
                                 .reindex(country_rl_b.index).fillna(0)
            ax.bar(x + width, country_rl_dc.values, width,
                   label='Damped (common ref)', color='#1f77b4', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(country_rl_b.index,
                           rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean RL across sectors')
        ax.set_title(
            f'Top {top_n} Most Affected Countries\n'
            f'Baseline vs Damped | Shock: {results["label"]}',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def summary_table(self, results_list: list) -> pd.DataFrame:
        """
        Cross-scenario summary table.
        Pass a list of results dicts from sim.run().
        """
        rows = []
        for r in results_list:
            rl_b = r['RL_final_base'].drop(
                index=r['shock_nodes'], errors='ignore')
            row = {
                'Scenario'     : r['label'],
                'Shocked nodes': len(r['shock_nodes']),
                'Base RL>10%'  : (rl_b > 0.1).sum(),
                'Base Max RL'  : f"{rl_b.max():.4f}",
                'Base Mean RL' : f"{rl_b.mean():.4f}",
            }
            if r['RL_final_damp'] is not None:
                rl_d   = r['RL_final_damp'].drop(
                    index=r['shock_nodes'], errors='ignore')
                delta  = r['delta_RL'].drop(
                    index=r['shock_nodes'], errors='ignore')
                rl_dc = r.get('RL_final_damp_common')
                if rl_dc is not None:
                    rl_dc = rl_dc.drop(index=r['shock_nodes'], errors='ignore')
                delta_common = r.get('delta_RL_common')
                if delta_common is not None:
                    delta_common = delta_common.drop(index=r['shock_nodes'], errors='ignore')

                keep_idx = [
                    self.nodes.get_loc(n)
                    for n in self.nodes if n not in r['shock_nodes']
                ]

                base_abs_t = float(np.sum(
                    r['baseline'][-1, keep_idx] - r['shocked_base'][-1, keep_idx]
                ))
                damp_abs_t = float(np.sum(
                    r['baseline_damp'][-1, keep_idx] - r['shocked_damp'][-1, keep_idx]
                ))
                base_abs_cum = float(np.sum(
                    r['baseline'][:, keep_idx] - r['shocked_base'][:, keep_idx]
                ))
                damp_abs_cum = float(np.sum(
                    r['baseline_damp'][:, keep_idx] - r['shocked_damp'][:, keep_idx]
                ))

                red_t = base_abs_t - damp_abs_t
                red_cum = base_abs_cum - damp_abs_cum
                red_t_pct = 100 * red_t / base_abs_t if base_abs_t > 0 else np.nan
                red_cum_pct = 100 * red_cum / base_abs_cum if base_abs_cum > 0 else np.nan

                row.update({
                    'Damp RL>10%'   : (rl_d > 0.1).sum(),
                    'Damp Max RL'   : f"{rl_d.max():.4f}",
                    'Damp Mean RL'  : f"{rl_d.mean():.4f}",
                    'Mean Absorption': f"{delta.mean():.4f}",
                    'Max Absorption' : f"{delta.max():.4f}",
                    'Base AbsLoss t_end'     : f"{base_abs_t:,.0f}",
                    'Damp AbsLoss t_end'     : f"{damp_abs_t:,.0f}",
                    'AbsLoss Reduction t_end': f"{red_t:,.0f}",
                    'AbsLoss Red% t_end'     : f"{red_t_pct:.2f}",
                    'Base AbsLoss cumulative'     : f"{base_abs_cum:,.0f}",
                    'Damp AbsLoss cumulative'     : f"{damp_abs_cum:,.0f}",
                    'AbsLoss Reduction cumulative': f"{red_cum:,.0f}",
                    'AbsLoss Red% cumulative'     : f"{red_cum_pct:.2f}",
                })

                if rl_dc is not None and delta_common is not None:
                    row.update({
                        'Damp RL>10% (CommonRef)'   : (rl_dc > 0.1).sum(),
                        'Damp Max RL (CommonRef)'   : f"{rl_dc.max():.4f}",
                        'Damp Mean RL (CommonRef)'  : f"{rl_dc.mean():.4f}",
                        'Mean Absorption (CommonRef)': f"{delta_common.mean():.4f}",
                        'Max Absorption (CommonRef)' : f"{delta_common.max():.4f}",
                    })
            rows.append(row)
        return pd.DataFrame(rows)

"""
visualize.py - Interactive visualization for Hopfield network experiments.

Provides real-time views of:
1. Weight matrix heatmap
2. Stored patterns grid
3. Energy landscape cross-sections
4. Stability profile (learned vs spurious)
5. Basin of attraction sizes
6. Learning progress over time
7. Pseudorehearsal generation process

Uses matplotlib for static plots and optional Plotly/Dash for
interactive web-based exploration.

Author: Simon McCallum (2026)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from typing import List, Optional, Tuple
from hopfield import HopfieldNetwork, HopfieldConfig, ExperimentRunner


class HopfieldVisualizer:
    """Static matplotlib-based visualization."""

    def __init__(self, net: HopfieldNetwork, figsize: Tuple[int, int] = (16, 10)):
        self.net = net
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    # ================================================================
    # Weight matrix visualization
    # ================================================================

    def plot_weight_matrix(self, ax: Optional[plt.Axes] = None,
                           title: str = "Weight Matrix"):
        """Heatmap of the weight matrix with symmetric colorbar."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        W = self.net.weights.cpu().numpy()
        vmax = max(abs(W.min()), abs(W.max()))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(W, cmap='RdBu_r', norm=norm, aspect='equal')
        ax.set_title(title)
        ax.set_xlabel('From unit')
        ax.set_ylabel('To unit')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return ax

    # ================================================================
    # Pattern visualization
    # ================================================================

    def plot_patterns(self, patterns: torch.Tensor,
                      labels: Optional[List[str]] = None,
                      title: str = "Stored Patterns",
                      ax: Optional[plt.Axes] = None,
                      reshape: Optional[Tuple[int, int]] = None):
        """
        Display patterns as a grid of binary images.

        If reshape is given (rows, cols), each pattern is displayed
        as a 2D image. Otherwise displayed as a 1D bar.
        """
        n_patterns = patterns.shape[0]
        N = patterns.shape[1]

        if reshape:
            rows, cols_per = reshape
            n_cols = min(8, n_patterns)
            n_rows = (n_patterns + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 1.5, n_rows * 1.5))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            fig.suptitle(title)

            for i in range(n_patterns):
                r, c = i // n_cols, i % n_cols
                pat = patterns[i].cpu().numpy().reshape(rows, cols_per)
                axes[r, c].imshow(pat, cmap='binary', vmin=-1, vmax=1,
                                  aspect='equal')
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                if labels:
                    axes[r, c].set_title(labels[i], fontsize=8)

            # Hide empty subplots
            for i in range(n_patterns, n_rows * n_cols):
                r, c = i // n_cols, i % n_cols
                axes[r, c].axis('off')
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, max(3, n_patterns * 0.3)))

            data = patterns.cpu().numpy()
            ax.imshow(data, cmap='binary', vmin=-1, vmax=1, aspect='auto')
            ax.set_title(title)
            ax.set_xlabel('Unit')
            ax.set_ylabel('Pattern')
            if labels:
                ax.set_yticks(range(n_patterns))
                ax.set_yticklabels(labels)

        return ax

    # ================================================================
    # Stability profile (key thesis visualization)
    # ================================================================

    def plot_stability_profile(self, patterns: torch.Tensor,
                                pattern_type: str = 'learned',
                                ax: Optional[plt.Axes] = None):
        """
        Plot the stability profile for each pattern.

        Shows the sorted signed stability values for each unit:
        - Positive = unit is correctly aligned with its input
        - Negative = unit wants to flip

        This is the key diagnostic from the thesis for understanding
        how robust each stored memory is.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(patterns.shape[0]):
            stats = self.net.stability_stats(patterns[i])
            inputs = torch.mv(self.net.weights, patterns[i])
            desired = self.net.threshold_activation(inputs)

            signed = torch.where(
                desired == patterns[i],
                inputs.abs(),
                -inputs.abs()
            ).cpu().numpy()
            signed.sort()

            color = 'blue' if stats['stable'] else 'red'
            alpha = 0.3 if patterns.shape[0] > 10 else 0.7
            ax.plot(signed, color=color, alpha=alpha, linewidth=0.8)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Unit (sorted by stability)')
        ax.set_ylabel('Signed stability')
        ax.set_title(f'Stability Profile ({pattern_type})')
        ax.legend(['Stable', 'Unstable'], loc='upper left')
        return ax

    # ================================================================
    # Energy ratio discrimination plot
    # ================================================================

    def plot_energy_ratio_discrimination(self, learned: torch.Tensor,
                                          spurious: Optional[List[torch.Tensor]] = None,
                                          ax: Optional[plt.Axes] = None):
        """
        Plot the energy ratio measure for learned vs spurious patterns.

        This is the thesis's key contribution: showing that the ratio
        measure can discriminate learned from spurious attractors.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        # Learned pattern ratios
        learned_ratios = []
        learned_means = []
        for i in range(learned.shape[0]):
            stats = self.net.stability_stats(learned[i])
            learned_ratios.append(stats['ratio'])
            learned_means.append(stats['mean_stable'])

        ax.scatter(learned_means, learned_ratios,
                   c='blue', label='Learned', alpha=0.7, s=50, marker='o')

        # Spurious pattern ratios
        if spurious:
            spurious_ratios = []
            spurious_means = []
            for sp in spurious:
                stats = self.net.stability_stats(sp)
                spurious_ratios.append(stats['ratio'])
                spurious_means.append(stats['mean_stable'])

            ax.scatter(spurious_means, spurious_ratios,
                       c='red', label='Spurious', alpha=0.5, s=30, marker='x')

        ax.set_xlabel('Mean Stability')
        ax.set_ylabel('Stability Ratio (low/high)')
        ax.set_title('Energy Ratio Discrimination: Learned vs Spurious')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)
        return ax

    # ================================================================
    # Learning progress
    # ================================================================

    def plot_experiment_results(self, results: List[dict],
                                 title: str = "Sequential Learning Results"):
        """Plot the main experiment results over pattern additions."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        n_patterns = [r['num_patterns'] for r in results]

        # 1. Fraction of patterns stable
        axes[0, 0].plot(n_patterns, [r['fraction_stable'] for r in results],
                        'b-o', markersize=4)
        axes[0, 0].set_ylabel('Fraction Stable')
        axes[0, 0].set_title('Pattern Stability')
        axes[0, 0].set_ylim(-0.05, 1.05)
        axes[0, 0].axhline(y=1.0, color='green', linestyle=':', alpha=0.5)

        # 2. Number of spurious patterns
        axes[0, 1].plot(n_patterns, [r['num_spurious'] for r in results],
                        'r-s', markersize=4)
        axes[0, 1].set_ylabel('Spurious Attractors')
        axes[0, 1].set_title('Spurious Pattern Count')

        # 3. Basin sizes (if available)
        if 'basin_sizes' in results[0]:
            mean_basins = [np.mean(r['basin_sizes']) if r['basin_sizes']
                          else 0 for r in results]
            axes[1, 0].plot(n_patterns, mean_basins, 'g-^', markersize=4)
            axes[1, 0].set_ylabel('Mean Basin Size')
            axes[1, 0].set_title('Basin of Attraction')
        else:
            # Per-pattern stability detail
            for r in results:
                stats = r['pattern_stats']
                ratios = [s['ratio'] for s in stats]
                axes[1, 0].scatter([r['num_patterns']] * len(ratios),
                                    ratios, c='blue', alpha=0.2, s=10)
            axes[1, 0].set_ylabel('Stability Ratio')
            axes[1, 0].set_title('Per-Pattern Stability Ratio')

        # 4. Individual pattern stability heatmap
        max_n = max(n_patterns)
        stability_matrix = np.zeros((len(results), max_n))
        stability_matrix[:] = np.nan
        for idx, r in enumerate(results):
            for s in r['pattern_stats']:
                stability_matrix[idx, s['pattern_idx']] = 1 if s['stable'] else 0

        im = axes[1, 1].imshow(stability_matrix.T, cmap='RdYlGn',
                                aspect='auto', vmin=0, vmax=1,
                                interpolation='nearest')
        axes[1, 1].set_xlabel('Learning Step')
        axes[1, 1].set_ylabel('Pattern Index')
        axes[1, 1].set_title('Pattern Stability Over Time')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        for ax in axes.flat:
            ax.set_xlabel('Number of Patterns')

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        return fig

    # ================================================================
    # Dashboard: all-in-one view
    # ================================================================

    def dashboard(self, patterns: torch.Tensor,
                   spurious: Optional[List[torch.Tensor]] = None,
                   title: str = "Hopfield Network Dashboard"):
        """Full dashboard showing network state."""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Weight matrix
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_weight_matrix(ax=ax1, title="Weights")

        # Stored patterns
        ax2 = fig.add_subplot(gs[0, 1:])
        self.plot_patterns(patterns, ax=ax2, title="Stored Patterns")

        # Stability profile - learned
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_stability_profile(patterns, 'learned', ax=ax3)

        # Energy ratio discrimination
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_energy_ratio_discrimination(patterns, spurious, ax=ax4)

        # Weight distribution
        ax5 = fig.add_subplot(gs[1, 2])
        W = self.net.weights.cpu().numpy().flatten()
        W = W[W != 0]  # Exclude zeros (diagonal)
        ax5.hist(W, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax5.set_title('Weight Distribution')
        ax5.set_xlabel('Weight Value')
        ax5.set_ylabel('Count')

        # Pattern stability summary
        ax6 = fig.add_subplot(gs[2, 0])
        stabilities = self.net.batch_stability_check(patterns)
        colors = ['green' if s else 'red' for s in stabilities]
        ax6.bar(range(len(stabilities)), [1] * len(stabilities), color=colors)
        ax6.set_title(f'Pattern Stability ({stabilities.sum()}/{len(stabilities)})')
        ax6.set_xlabel('Pattern Index')
        ax6.set_yticks([])

        # Energy values
        ax7 = fig.add_subplot(gs[2, 1])
        energies = []
        for i in range(patterns.shape[0]):
            energies.append(self.net.energy(patterns[i]))
        ax7.bar(range(len(energies)), energies, color='steelblue')
        ax7.set_title('Pattern Energy')
        ax7.set_xlabel('Pattern Index')
        ax7.set_ylabel('E = -0.5 s^T W s')

        # Ratio values
        ax8 = fig.add_subplot(gs[2, 2])
        ratios = []
        for i in range(patterns.shape[0]):
            stats = self.net.stability_stats(patterns[i])
            ratios.append(stats['ratio'])
        ax8.bar(range(len(ratios)), ratios,
                color=['green' if r > 0 else 'red' for r in ratios])
        ax8.set_title('Stability Ratio')
        ax8.set_xlabel('Pattern Index')
        ax8.set_ylabel('Low/High Ratio')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        return fig


# ================================================================
# Interactive Plotly/Dash dashboard (optional)
# ================================================================

def create_interactive_dashboard(net: HopfieldNetwork,
                                  patterns: torch.Tensor,
                                  port: int = 8050):
    """
    Launch an interactive web dashboard using Plotly Dash.

    Features:
    - Real-time weight matrix with hover values
    - Clickable patterns to see relaxation trajectory
    - Slider for noise level to see soft-threshold effect
    - Animation of learning process
    """
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("Install dash and plotly for interactive visualization:")
        print("  pip install dash plotly")
        return None

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Hopfield Network Explorer"),

        html.Div([
            html.Div([
                html.H3("Weight Matrix"),
                dcc.Graph(id='weight-heatmap'),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.H3("Stability Profile"),
                dcc.Graph(id='stability-profile'),
            ], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        html.Div([
            html.H3("Noise Level (Gaussian soft-threshold)"),
            dcc.Slider(id='noise-slider', min=0, max=2, step=0.05,
                       value=0, marks={i/10: f'{i/10:.1f}' for i in range(0, 21, 5)}),
            html.P(id='noise-description'),
        ]),

        html.Div([
            html.H3("Pattern Selection"),
            dcc.Dropdown(
                id='pattern-selector',
                options=[{'label': f'Pattern {i}', 'value': i}
                         for i in range(patterns.shape[0])],
                value=0
            ),
            dcc.Graph(id='relaxation-trajectory'),
        ]),
    ])

    @app.callback(
        Output('weight-heatmap', 'figure'),
        Input('noise-slider', 'value')
    )
    def update_weights(noise_val):
        W = net.weights.cpu().numpy()
        fig = go.Figure(data=go.Heatmap(z=W, colorscale='RdBu_r',
                                         zmid=0))
        fig.update_layout(title='Weight Matrix', height=400)
        return fig

    @app.callback(
        Output('stability-profile', 'figure'),
        Input('pattern-selector', 'value')
    )
    def update_stability(pat_idx):
        pat = patterns[pat_idx]
        inputs = torch.mv(net.weights, pat)
        desired = net.threshold_activation(inputs)
        signed = torch.where(desired == pat, inputs.abs(), -inputs.abs())
        signed_sorted = signed.sort().values.cpu().numpy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signed_sorted, mode='lines',
                                  name='Stability'))
        fig.add_hline(y=0, line_dash='dash', line_color='red')
        fig.update_layout(title=f'Stability Profile - Pattern {pat_idx}',
                          xaxis_title='Unit (sorted)',
                          yaxis_title='Signed Stability',
                          height=400)
        return fig

    @app.callback(
        Output('noise-description', 'children'),
        Input('noise-slider', 'value')
    )
    def update_noise_desc(noise_val):
        if noise_val == 0:
            return "No noise: hard threshold activation"
        elif noise_val < 0.5:
            return f"Light noise (σ={noise_val:.2f}): slight smoothing of activation"
        elif noise_val < 1.0:
            return f"Moderate noise (σ={noise_val:.2f}): sigmoid-like activation"
        else:
            return f"Heavy noise (σ={noise_val:.2f}): nearly linear activation"

    print(f"Starting dashboard at http://localhost:{port}")
    app.run(port=port, debug=False)
    return app


# ================================================================
# Quick-start demonstration
# ================================================================

if __name__ == '__main__':
    print("Hopfield Network Visualization Demo")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create network
    config = HopfieldConfig(
        N=100,
        learning_type=LearningType.PSEUDO_DELTA,
        learning_rate=0.1,
        training_epochs=500,
        initial_training_epochs=1000,
        max_pseudo_patterns=64,
        max_samples_for_pseudo=500,
        num_check=100,
        max_learnt_patterns=30,
        initial_num_patterns=10,
        step=2,
        device=device,
    )

    net = HopfieldNetwork(config)
    viz = HopfieldVisualizer(net)

    # Generate random patterns
    patterns = HopfieldNetwork.generate_random_patterns(
        30, 100, prob_on=0.5, device=device)

    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run_sequential_learning(patterns, verbose=True)

    # Visualize
    fig = viz.plot_experiment_results(results, "Pseudorehearsal Demo (N=100)")
    plt.savefig('experiment_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to experiment_results.png")

    # Dashboard
    viz.dashboard(patterns, title="Final Network State")
    plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
    print("Dashboard saved to dashboard.png")

    plt.show()

"""
demo.py - Step-by-step demonstration of Hopfield network learning
and pseudorehearsal for catastrophic forgetting.

Run this to watch the network learn, forget, and recover memories.

Usage:
    python demo.py                    # Full demo with plots
    python demo.py --no-plot          # Text-only output
    python demo.py --mode hebbian     # Hebbian only (shows forgetting)
    python demo.py --mode delta       # Delta only (shows catastrophic forgetting)
    python demo.py --mode pseudo      # Pseudorehearsal (shows recovery)
    python demo.py --mode compare     # Side-by-side comparison of all methods
    python demo.py --N 200            # Larger network
    python demo.py --interactive      # Launch web dashboard
"""

import argparse
import sys
import torch
import numpy as np

from hopfield import (HopfieldNetwork, HopfieldConfig, ExperimentRunner,
                      LearningType, EWCHopfield)


def print_pattern(pattern, width=50):
    """Print a binary pattern as a text bar."""
    n = len(pattern)
    chars = ['#' if v > 0 else '.' for v in pattern]
    if n > width:
        step = n / width
        chars = [chars[int(i * step)] for i in range(width)]
    return ''.join(chars)


def print_stability_bar(net, patterns, label=""):
    """Print a visual stability bar for all patterns."""
    stable = net.batch_stability_check(patterns)
    n = len(stable)
    bar = ''.join(['G' if s else 'x' for s in stable])
    count = stable.sum().item()
    print(f"  {label:20s} [{bar}] {count}/{n} stable ({100*count/n:.0f}%)")
    return count


def demo_catastrophic_forgetting(config, patterns, show_plots=True):
    """
    Demonstrate catastrophic forgetting with delta learning.

    Learn patterns in two batches. After learning batch 2,
    batch 1 patterns are destroyed.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Catastrophic Forgetting (Delta Learning)")
    print("=" * 60)
    print(f"Network: N={config.N}, {patterns.shape[0]} patterns")
    print()

    net = HopfieldNetwork(config)
    n_batch1 = patterns.shape[0] // 2
    batch1 = patterns[:n_batch1]
    batch2 = patterns[n_batch1:]

    # Learn batch 1
    print("Step 1: Learning first batch of patterns...")
    net.delta_learn(batch1)
    print_stability_bar(net, batch1, "After batch 1")
    print()

    # Show each pattern
    print("  Stored patterns (batch 1):")
    for i in range(min(5, n_batch1)):
        stable = net.is_stable(batch1[i])
        status = "STABLE" if stable else "UNSTABLE"
        print(f"    P{i}: {print_pattern(batch1[i].cpu())} [{status}]")
    if n_batch1 > 5:
        print(f"    ... and {n_batch1 - 5} more")
    print()

    # Now learn batch 2 WITHOUT pseudorehearsal
    print("Step 2: Learning second batch (NO rehearsal)...")
    config_no_pseudo = HopfieldConfig(
        N=config.N, learning_type=LearningType.DELTA,
        learning_rate=config.learning_rate,
        training_epochs=config.training_epochs,
        initial_training_epochs=config.training_epochs,
        error_criteria=config.error_criteria,
        device=config.device)
    net2 = HopfieldNetwork(config_no_pseudo)
    net2.weights = net.weights.clone()

    # Learn batch 2 on top of existing weights
    net2.delta_learn(batch2)

    print_stability_bar(net2, batch1, "Batch 1 (old)")
    print_stability_bar(net2, batch2, "Batch 2 (new)")
    all_pats = torch.cat([batch1, batch2])
    print_stability_bar(net2, all_pats, "All patterns")
    print()

    old_stable = net2.batch_stability_check(batch1).sum().item()
    new_stable = net2.batch_stability_check(batch2).sum().item()
    print(f"  >> CATASTROPHIC FORGETTING: {n_batch1 - old_stable}/{n_batch1} "
          f"old patterns destroyed!")
    print()

    return net2


def demo_pseudorehearsal(config, patterns, show_plots=True):
    """
    Demonstrate pseudorehearsal preventing catastrophic forgetting.

    Same setup as above, but with pseudo-pattern generation between batches.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Pseudorehearsal (Catastrophic Forgetting Prevention)")
    print("=" * 60)
    print(f"Network: N={config.N}, {patterns.shape[0]} patterns")
    print()

    net = HopfieldNetwork(config)
    n_batch1 = patterns.shape[0] // 2
    batch1 = patterns[:n_batch1]
    batch2 = patterns[n_batch1:]

    # Learn batch 1
    print("Step 1: Learning first batch of patterns...")
    net.delta_learn(batch1)
    s1 = print_stability_bar(net, batch1, "After batch 1")
    print()

    # Generate pseudo-patterns
    print("Step 2: Generating pseudo-patterns from network...")
    print("  (Random probes -> relax -> filter by energy ratio)")
    pseudo = net.generate_pseudo_patterns(batch1, verbose=True)
    print(f"  Pseudo-patterns found: {pseudo.shape[0]}")

    # Show some pseudo-patterns
    print("\n  Sample pseudo-patterns:")
    for i in range(min(3, pseudo.shape[0])):
        stats = net.stability_stats(pseudo[i])
        print(f"    PP{i}: {print_pattern(pseudo[i].cpu())} "
              f"[ratio={stats['ratio']:.3f}]")
    print()

    # Learn batch 2 WITH pseudorehearsal
    print("Step 3: Learning second batch WITH pseudo-rehearsal...")
    net.delta_learn(
        torch.cat([batch1, batch2]),
        start_idx=n_batch1,
        pseudo_patterns=pseudo)

    print_stability_bar(net, batch1, "Batch 1 (old)")
    print_stability_bar(net, batch2, "Batch 2 (new)")
    all_pats = torch.cat([batch1, batch2])
    total = print_stability_bar(net, all_pats, "All patterns")
    print()

    old_stable = net.batch_stability_check(batch1).sum().item()
    print(f"  >> PSEUDOREHEARSAL: {old_stable}/{n_batch1} old patterns PRESERVED!")
    print()

    return net


def demo_sequential_learning(config, patterns, show_plots=True):
    """
    Full sequential learning experiment matching the thesis.

    Add patterns one at a time, measuring stability at each step.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Sequential Learning Experiment")
    print("=" * 60)
    print(f"Network: N={config.N}, learning {patterns.shape[0]} patterns "
          f"incrementally")
    print(f"Method: {config.learning_type.name}")
    print()

    runner = ExperimentRunner(config)
    results = runner.run_sequential_learning(patterns, verbose=True)

    # Print summary table
    print("\n" + "-" * 60)
    print(f"{'Patterns':>10s} {'Stable':>10s} {'%':>8s} {'Spurious':>10s}")
    print("-" * 60)
    for r in results:
        print(f"{r['num_patterns']:>10d} "
              f"{r['num_stable']:>10d} "
              f"{r['fraction_stable']*100:>7.1f}% "
              f"{r['num_spurious']:>10d}")
    print("-" * 60)

    if show_plots:
        try:
            from visualize import HopfieldVisualizer
            viz = HopfieldVisualizer(runner.net)
            fig = viz.plot_experiment_results(
                results,
                f"Sequential Learning ({config.learning_type.name}, N={config.N})")
            fig.savefig(f'results_{config.learning_type.name.lower()}.png',
                        dpi=150, bbox_inches='tight')
            print(f"\nPlot saved: results_{config.learning_type.name.lower()}.png")
        except Exception as e:
            print(f"\nPlot error: {e}")

    return results


def demo_comparison(config, patterns, show_plots=True):
    """
    Side-by-side comparison of all learning methods.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Method Comparison")
    print("=" * 60)
    print(f"Network: N={config.N}, {patterns.shape[0]} patterns")
    print()

    methods = [
        ("Hebbian", LearningType.HEBBIAN),
        ("Delta", LearningType.DELTA),
        ("PseudoDelta", LearningType.PSEUDO_DELTA),
        ("Unlearning", LearningType.UNLEARNING),
    ]

    all_results = {}

    for name, ltype in methods:
        print(f"\n--- {name} ---")
        cfg = HopfieldConfig(
            N=config.N,
            learning_type=ltype,
            learning_rate=config.learning_rate,
            training_epochs=config.training_epochs,
            initial_training_epochs=config.initial_training_epochs,
            max_pseudo_patterns=config.max_pseudo_patterns,
            max_samples_for_pseudo=config.max_samples_for_pseudo,
            pseudo_rehearsal_cutoff=config.pseudo_rehearsal_cutoff,
            num_check=config.num_check,
            max_learnt_patterns=config.max_learnt_patterns,
            initial_num_patterns=config.initial_num_patterns,
            step=config.step,
            device=config.device,
        )

        runner = ExperimentRunner(cfg)
        results = runner.run_sequential_learning(patterns, verbose=False)
        all_results[name] = results

        final = results[-1]
        print(f"  Final: {final['num_stable']}/{final['num_patterns']} stable "
              f"({final['fraction_stable']*100:.0f}%), "
              f"{final['num_spurious']} spurious")

    # Comparison plot
    if show_plots:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            for name, results in all_results.items():
                n_pats = [r['num_patterns'] for r in results]
                stability = [r['fraction_stable'] for r in results]
                spurious = [r['num_spurious'] for r in results]

                ax1.plot(n_pats, stability, '-o', label=name, markersize=3)
                ax2.plot(n_pats, spurious, '-s', label=name, markersize=3)

            ax1.set_xlabel('Number of Patterns')
            ax1.set_ylabel('Fraction Stable')
            ax1.set_title('Pattern Stability')
            ax1.legend()
            ax1.set_ylim(-0.05, 1.05)

            ax2.set_xlabel('Number of Patterns')
            ax2.set_ylabel('Spurious Attractors')
            ax2.set_title('Spurious Pattern Count')
            ax2.legend()

            fig.suptitle(f'Method Comparison (N={config.N})', fontsize=14)
            fig.tight_layout()
            fig.savefig('comparison.png', dpi=150, bbox_inches='tight')
            print(f"\nComparison plot saved: comparison.png")
            plt.show()
        except Exception as e:
            print(f"\nPlot error: {e}")

    return all_results


def demo_dashboard(config, patterns):
    """Launch the network dashboard."""
    print("\n" + "=" * 60)
    print("DEMO 5: Network Dashboard")
    print("=" * 60)

    net = HopfieldNetwork(config)
    n_initial = min(15, patterns.shape[0])

    print(f"Learning {n_initial} patterns with pseudorehearsal...")
    net.delta_learn(patterns[:n_initial])

    # Sample to find spurious patterns
    sample = net.sample_network(patterns[:n_initial], num_samples=200)
    spurious = sample['spurious_patterns']
    print(f"Found {len(spurious)} spurious attractors")

    from visualize import HopfieldVisualizer
    viz = HopfieldVisualizer(net)
    fig = viz.dashboard(patterns[:n_initial], spurious=spurious,
                         title=f"Hopfield Network (N={config.N}, "
                               f"{n_initial} patterns)")
    fig.savefig('dashboard.png', dpi=150, bbox_inches='tight')
    print("Dashboard saved: dashboard.png")

    import matplotlib.pyplot as plt
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Hopfield Network Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                     # Full demo
  python demo.py --mode compare      # Compare all methods
  python demo.py --mode pseudo       # Pseudorehearsal demo
  python demo.py --N 200 --patterns 40  # Larger experiment
  python demo.py --no-plot           # Text only
        """)

    parser.add_argument('--mode', choices=['hebbian', 'delta', 'pseudo',
                                            'compare', 'forgetting',
                                            'dashboard', 'all'],
                        default='all', help='Demo mode')
    parser.add_argument('--N', type=int, default=100, help='Network size')
    parser.add_argument('--patterns', type=int, default=20,
                        help='Number of patterns')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plots')
    parser.add_argument('--device', default=None,
                        help='Device (cuda/cpu, auto-detected)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--interactive', action='store_true',
                        help='Launch interactive web dashboard')

    args = parser.parse_args()

    # Setup
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    print("Hopfield Network - Catastrophic Forgetting Demos")
    print("=" * 50)
    print(f"Device: {device}", end="")
    if device == 'cuda':
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"Network: N={args.N}, Patterns: {args.patterns}")
    print()

    # Config - tuned for interactive speed (~30s per demo)
    config = HopfieldConfig(
        N=args.N,
        learning_type=LearningType.PSEUDO_DELTA,
        learning_rate=0.1,
        training_epochs=200,
        initial_training_epochs=500,
        error_criteria=0.001,
        max_pseudo_patterns=64,
        max_samples_for_pseudo=500,
        pseudo_rehearsal_cutoff=-1.0,
        num_check=50,
        max_learnt_patterns=args.patterns,
        initial_num_patterns=max(5, args.patterns // 4),
        step=2,
        device=device,
    )

    # Generate patterns
    patterns = HopfieldNetwork.generate_random_patterns(
        args.patterns, args.N, prob_on=0.5, device=device)

    show_plots = not args.no_plot

    # Run selected demo
    if args.mode in ('forgetting', 'all'):
        demo_catastrophic_forgetting(config, patterns, show_plots)

    if args.mode in ('pseudo', 'all'):
        demo_pseudorehearsal(config, patterns, show_plots)

    if args.mode == 'hebbian':
        config.learning_type = LearningType.HEBBIAN
        demo_sequential_learning(config, patterns, show_plots)

    if args.mode == 'delta':
        config.learning_type = LearningType.DELTA
        demo_sequential_learning(config, patterns, show_plots)

    if args.mode == 'pseudo' and args.mode != 'all':
        config.learning_type = LearningType.PSEUDO_DELTA
        demo_sequential_learning(config, patterns, show_plots)

    if args.mode in ('compare', 'all'):
        demo_comparison(config, patterns, show_plots)

    if args.mode == 'dashboard' or args.interactive:
        demo_dashboard(config, patterns)

    if args.interactive:
        from visualize import create_interactive_dashboard
        net = HopfieldNetwork(config)
        net.delta_learn(patterns[:15])
        create_interactive_dashboard(net, patterns[:15])

    print("\nDone!")


if __name__ == '__main__':
    main()

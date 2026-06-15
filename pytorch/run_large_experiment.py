"""
run_large_experiment.py - Run the large N=500 and N=1000 pseudorehearsal
experiments matching the thesis parameters from delta-RealR-10-1.train.

Thesis parameters:
  learningType: PSEUDODELTA
  realRehearsal: true (10%)
  learningConst: 1.0
  trainingEpochs: 10000
  errorCriteria: 0.001
  errorTail: 0.99
  noiseOnInput: yes
  gaussianNoiseType: RELATIVE
  gaussianRelativeRange: 0.1
  initialNumberPatts: 10
  step: 1
  maxLearntPatts: 1.5*N
"""

import torch
import time
import sys
import json

from hopfield import (HopfieldNetwork, HopfieldConfig, ExperimentRunner,
                      LearningType, RelaxationType)

def run_experiment(N, seed=42, save_results=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    max_learnt = int(1.5 * N)

    print(f"\n{'='*70}")
    print(f"LARGE EXPERIMENT: N={N}, PseudoDelta + Real Rehearsal (10%)")
    print(f"{'='*70}")
    print(f"Device: {device}", end="")
    if device == 'cuda':
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"Max patterns: {max_learnt} (1.5*N)")
    print(f"Parameters matching thesis: delta-RealR-10-1.train")
    print()

    config = HopfieldConfig(
        N=N,
        learning_type=LearningType.PSEUDO_DELTA,
        learning_rate=1.0,
        training_epochs=10000,
        initial_training_epochs=10000,
        error_criteria=0.001,
        error_tail=0.99,
        noise_on_input=True,
        gaussian_noise_type='relative',
        gaussian_relative_range=0.1,
        max_pseudo_patterns=256,
        max_samples_for_pseudo=2000,
        pseudo_rehearsal_cutoff=-1.0,
        real_rehearsal=True,
        real_rehearsal_percent=0.1,
        num_check=50,
        max_learnt_patterns=max_learnt,
        initial_num_patterns=10,
        step=1,
        relaxation=RelaxationType.ASYNCHRONOUS,
        max_cycles=10 * N,
        device=device,
    )

    # Generate random patterns
    patterns = HopfieldNetwork.generate_random_patterns(
        max_learnt, N, prob_on=0.5, device=device)

    print(f"Generated {patterns.shape[0]} random patterns ({patterns.shape})")
    print(f"Starting sequential learning...\n")

    runner = ExperimentRunner(config)
    start_time = time.time()

    results = runner.run_sequential_learning(patterns, verbose=True)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")

    # Summary table
    print(f"\n{'Patterns':>10s} {'Stable':>10s} {'%':>8s} {'Spurious':>10s}")
    print("-" * 45)
    for r in results:
        print(f"{r['num_patterns']:>10d} "
              f"{r['num_stable']:>10d} "
              f"{r['fraction_stable']*100:>7.1f}% "
              f"{r['num_spurious']:>10d}")
    print("-" * 45)

    if save_results:
        # Save results (strip non-serializable data)
        save_data = []
        for r in results:
            save_data.append({
                'num_patterns': r['num_patterns'],
                'num_stable': r['num_stable'],
                'fraction_stable': r['fraction_stable'],
                'num_spurious': r['num_spurious'],
                'cycling_count': r['cycling_count'],
            })

        filename = f'results_N{N}_pseudodelta_realR.json'
        with open(filename, 'w') as f:
            json.dump({
                'N': N,
                'config': 'delta-RealR-10-1',
                'elapsed_seconds': elapsed,
                'seed': seed,
                'results': save_data,
            }, f, indent=2)
        print(f"\nResults saved to {filename}")

    return results


if __name__ == '__main__':
    # Default: N=500. Pass N on command line to override.
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run_experiment(N)

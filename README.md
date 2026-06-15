# Hopfield Network: Catastrophic Forgetting & Pseudorehearsal

Research code for studying catastrophic forgetting in Hopfield networks, with pseudorehearsal as a biologically-plausible solution. Originally a thesis project (C code, ~2003), now modernised with a PyTorch reimplementation and CUDA acceleration.

## Quick Start (PyTorch - recommended)

### Prerequisites

```bash
# Python 3.10+ with PyTorch (CUDA optional but recommended)
pip install torch numpy matplotlib
```

### Run the demo

```bash
cd pytorch

# Watch catastrophic forgetting happen, then see pseudorehearsal fix it
python demo.py

# Text-only (no matplotlib needed)
python demo.py --no-plot
```

This runs through 4 demos:

1. **Catastrophic Forgetting** - Learn 6 patterns, then 6 more. All original patterns destroyed.
2. **Pseudorehearsal** - Same setup, but generate pseudo-patterns between batches. Old memories preserved.
3. **Sequential Learning** - Add patterns one at a time, tracking stability at each step.
4. **Method Comparison** - Side-by-side: Hebbian vs Delta vs Pseudorehearsal vs Unlearning.

### Demo modes

```bash
# Just the forgetting demo (fast, dramatic)
python demo.py --mode forgetting

# Just pseudorehearsal (shows the fix)
python demo.py --mode pseudo

# Compare all 4 methods side by side
python demo.py --mode compare

# Full dashboard with weight matrix, stability profiles, energy ratios
python demo.py --mode dashboard

# Bigger network (GPU recommended)
python demo.py --N 200 --patterns 40

# Reproduce with same random seed
python demo.py --seed 42
```

### What to watch for

When you run `python demo.py`, look for:

```
DEMO 1: Catastrophic Forgetting (Delta Learning)
  After batch 1        [GGGGGG] 6/6 stable (100%)    <-- all learned
  Batch 1 (old)        [xxxxxx] 0/6 stable (0%)      <-- ALL DESTROYED
  Batch 2 (new)        [GGGGGG] 6/6 stable (100%)    <-- new ones fine
  >> CATASTROPHIC FORGETTING: 6/6 old patterns destroyed!

DEMO 2: Pseudorehearsal
  After batch 1        [GGGGGG] 6/6 stable (100%)    <-- all learned
  Generating pseudo-patterns from network...
  Pseudo-patterns found: 16                            <-- network probed
  Batch 1 (old)        [GxGxGG] 4/6 stable (67%)     <-- mostly preserved!
  Batch 2 (new)        [GGGGxG] 5/6 stable (83%)     <-- new ones too
  >> PSEUDOREHEARSAL: 4/6 old patterns PRESERVED!
```

`G` = stable, `x` = unstable (destroyed). The key result: without rehearsal, **all** old patterns are destroyed. With pseudorehearsal, most survive -- and the network never stored the originals, only probed its own attractors.

### Visualization

```bash
# Generate dashboard PNG showing network internals
python demo.py --mode dashboard

# Interactive web dashboard (requires: pip install dash plotly)
python demo.py --interactive
```

The dashboard shows:
- **Weight matrix** heatmap (symmetric for Hopfield nets)
- **Stored patterns** as binary rows
- **Stability profile** - sorted signed stability per unit (thesis diagnostic)
- **Energy ratio** scatter - learned (blue) vs spurious (red) attractors
- **Weight distribution** histogram
- **Per-pattern stability** over sequential learning

### Run a custom experiment

```python
from hopfield import HopfieldNetwork, HopfieldConfig, LearningType

config = HopfieldConfig(
    N=100,
    learning_type=LearningType.PSEUDO_DELTA,
    learning_rate=0.1,
    training_epochs=500,
    max_pseudo_patterns=64,
    device='cuda',  # or 'cpu'
)

net = HopfieldNetwork(config)

# Generate random bipolar patterns
patterns = HopfieldNetwork.generate_random_patterns(20, 100, device='cuda')

# Learn first 10
net.delta_learn(patterns[:10])

# Generate pseudo-patterns (probe network, relax, filter)
pseudo = net.generate_pseudo_patterns(patterns[:10], verbose=True)

# Learn next 5 with pseudorehearsal
net.delta_learn(patterns[:15], start_idx=10, pseudo_patterns=pseudo)

# Check what survived
stable = net.batch_stability_check(patterns[:15])
print(f"Stable: {stable.sum()}/{len(stable)}")

# Examine a pattern's stability
stats = net.stability_stats(patterns[0])
print(f"Pattern 0: ratio={stats['ratio']:.3f}, "
      f"least_stable={stats['least_stable']:.3f}")
```

### Modern continual learning comparisons

The PyTorch code includes EWC and PackNet for comparison:

```python
from hopfield import EWCHopfield, PackNetHopfield, HopfieldConfig

# Elastic Weight Consolidation
net = EWCHopfield(HopfieldConfig(N=100, device='cuda'))
net.delta_learn(patterns[:10])
net.compute_fisher(patterns[:10])  # estimate weight importance
net.delta_learn_ewc(patterns[:15], start_idx=10)  # penalise changes to important weights

# PackNet (prune and freeze)
net = PackNetHopfield(HopfieldConfig(N=100, device='cuda'))
net.delta_learn(patterns[:10])
net.prune_and_freeze(keep_fraction=0.5)  # freeze important weights
net.delta_learn(patterns[:15], start_idx=10)  # only free weights updated
```

## Original C Code

The original thesis code compiles and runs on modern Windows.

### Build

Open `ThesisCode/annescode.sln` in Visual Studio 2022, or:

```bash
cd ThesisCode
# MSBuild (the vcxproj has been updated to v143 toolset)
MSBuild annescode.vcxproj /p:Configuration=Debug /p:Platform=Win32
```

### Run

```bash
cd ThesisCode

# Default parameters
Debug/annescode.exe

# Specific experiment
Debug/annescode.exe Data/Net100 Data/delta-RealR-10-1

# With output redirect
Debug/annescode.exe Data/Net100 Data/delta-RealR-10-1 output.txt
```

Parameter files: `.net` files configure the network, `.train` files configure the experiment. The program reads `default.net` and `default.train` first, then overlays the specified files.

### Key parameter files

| File | Description |
|------|-------------|
| `Data/Net64.net` | 64-unit network (thesis NetA) |
| `Data/Net100.net` | 100-unit network (thesis NetB) |
| `Data/delta-RealR-10-1.train` | Delta learning with real rehearsal |
| `default.train` | Pseudorehearsal configuration |

### CUDA acceleration

CUDA kernels are provided for the C code in `ThesisCode/cuda/`:

```bash
cd ThesisCode/cuda
nvcc -arch=sm_86 -c gpu_hopfield.cu -o gpu_hopfield.obj -lcublas
```

These replace the computational hotspots: matrix-vector multiply (cuBLAS), Hebbian/delta weight updates, and batch relaxation.

## Key Concept: Gaussian Noise as Soft Threshold

The thesis uses Gaussian noise on threshold units:

```
output = sign(input + noise),  noise ~ N(0, σ)
```

This creates a **soft sigmoid activation**: `E[output] = 2Φ(input/σ) - 1`, where Φ is the normal CDF. The noise standard deviation σ acts as an inverse temperature - high σ gives a smooth sigmoid, low σ approaches the hard threshold. This enables gradient-like learning on binary Hopfield units.

## Project Structure

```
thesiscode/
├── pytorch/              # Modern PyTorch reimplementation
│   ├── hopfield.py       # Core: HopfieldNetwork, EWC, PackNet
│   ├── visualize.py      # Matplotlib + Plotly visualization
│   ├── demo.py           # Step-by-step demos
│   └── requirements.txt
├── ThesisCode/           # Original C code
│   ├── annescode.c       # Main experiment loop
│   ├── net.c             # Network operations (calcInputs, relax)
│   ├── learning.c        # Hebbian & delta learning
│   ├── pseudolearning.c  # Pseudorehearsal algorithm
│   ├── sampling.c        # Basin of attraction sampling
│   ├── cuda/             # CUDA kernels
│   └── Data/             # Parameter files and network configs
├── Thesis/               # LaTeX thesis source
│   ├── Master.tex
│   └── thesis source/    # Chapter files
├── docs/
│   ├── analysis.tex      # Technical analysis report
│   └── references.bib    # Bibliography
└── README.md
```

## References

- McCallum, S. (2003). *Pseudorehearsal in Hopfield Networks*. PhD Thesis, University of Otago.
- Atkinson, C. et al. (2018). *Pseudo-Rehearsal: Achieving Deep Reinforcement Learning without Catastrophic Forgetting*. arXiv:1812.02464.

"""
hopfield.py - Modern PyTorch implementation of Hopfield networks
with pseudorehearsal for catastrophic forgetting research.

Reimplements the thesis experiments with:
- GPU acceleration via PyTorch tensors
- Gaussian noise as soft-threshold activation (see thesis insight)
- Modern continual learning comparisons
- Interactive visualization support

Author: Simon McCallum (2026 modernisation)
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto


class LearningType(Enum):
    HEBBIAN = auto()
    DELTA = auto()
    PSEUDO_DELTA = auto()
    UNLEARNING = auto()


class RelaxationType(Enum):
    SYNCHRONOUS = auto()
    ASYNCHRONOUS = auto()


@dataclass
class HopfieldConfig:
    """Configuration mirroring the thesis parameter files."""
    N: int = 100                        # Number of units
    ON: float = 1.0                     # ON activation value
    OFF: float = -1.0                   # OFF activation value

    # Learning
    learning_type: LearningType = LearningType.PSEUDO_DELTA
    learning_rate: float = 0.1
    momentum: float = 0.5
    learning_rate_decay_to: float = 0.01

    # Training
    training_epochs: int = 1000
    initial_training_epochs: int = 1000
    error_criteria: float = 0.001
    error_tail: float = 0.99
    all_epoch_learning: bool = False

    # Hebbian-specific
    hebbian_cycles: int = 1
    hebbian_noise_level: float = 0.0

    # Relaxation
    relaxation: RelaxationType = RelaxationType.ASYNCHRONOUS
    max_cycles: int = 400               # 4*N default
    stable_time_required: int = 100

    # Noise (Gaussian noise + threshold = soft sigmoid)
    noise_on_input: bool = False
    noise_during_relaxation: bool = False
    gaussian_noise_type: str = 'relative'  # 'relative' or 'absolute'
    gaussian_relative_range: float = 0.1
    gaussian_absolute_range: float = 10.0

    # Heteroassociative noise
    noise_hetro_associative: bool = False
    hetro_noise_level: float = 0.1

    # Weight constraints
    weight_capping: str = 'none'        # 'none', 'hard', 'soft'
    weight_cap_value: float = 200.0
    weight_decay: bool = False
    weight_decay_value: float = 0.1
    weight_normalisation: str = 'none'  # 'none', 'net', 'unit'
    weight_norm_value: float = 5000.0
    symmetric_weights: bool = False

    # Pseudorehearsal
    max_pseudo_patterns: int = 256
    max_samples_for_pseudo: int = 2000
    pseudo_rehearsal_cutoff: float = -1.0
    unique_pseudo_patterns: bool = True
    remove_learnt_from_pseudo: bool = False
    learning_ratio_pseudo: float = 1.0
    noise_on_pseudo: bool = False
    noise_ratio_pseudo: float = 1.0

    # Real rehearsal (for comparison)
    real_rehearsal: bool = False
    real_rehearsal_percent: float = 0.1

    # Unlearning
    unlearning_const: float = 0.01
    unlearning_super_heated: bool = False
    unlearning_cutoff_high: float = 1.0

    # Sampling
    sample_prob_on: float = 0.5
    units_in_ratio: float = 0.1
    num_check: int = 20
    max_spurious_patterns: int = 1000

    # Basin of attraction
    calc_learnt_basins: bool = False
    calc_spurious_basins: bool = False

    # Temperature
    thermal_delta: bool = False
    start_temperature: float = 1.0

    # Experiment
    num_trials: int = 50
    max_learnt_patterns: int = 100
    initial_num_patterns: int = 10
    step: int = 1

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class HopfieldNetwork:
    """
    Hopfield network with support for multiple learning rules,
    noise injection, and pseudorehearsal.

    Key insight from thesis: Gaussian noise on threshold units creates
    a soft-threshold activation that approximates sigmoid behaviour.
    The noise smooths the energy landscape during training, making
    the binary unit training resemble gradient-based sigmoid training.
    """

    def __init__(self, config: HopfieldConfig):
        self.cfg = config
        self.N = config.N
        self.device = torch.device(config.device)

        # Weight matrix: N x N (no self-connections)
        self.weights = torch.zeros(self.N, self.N, device=self.device,
                                   dtype=torch.float64)
        self.weight_deltas = torch.zeros_like(self.weights)

        # Current state
        self.state = torch.zeros(self.N, device=self.device,
                                 dtype=torch.float64)
        self.unit_input = torch.zeros(self.N, device=self.device,
                                      dtype=torch.float64)

        # Statistics
        self.mean_unit_input = 0.0
        self.temperature = config.start_temperature

    def reset_weights(self):
        """Zero all weights."""
        self.weights.zero_()
        self.weight_deltas.zero_()

    # ================================================================
    # Activation functions
    # ================================================================

    def threshold_activation(self, inputs: torch.Tensor) -> torch.Tensor:
        """Binary threshold: input > 0 -> ON, else -> OFF."""
        return torch.where(inputs > 0,
                          torch.full_like(inputs, self.cfg.ON),
                          torch.full_like(inputs, self.cfg.OFF))

    def soft_threshold_activation(self, inputs: torch.Tensor,
                                   noise_std: float) -> torch.Tensor:
        """
        Gaussian noise + threshold = soft sigmoid activation.

        This is the key insight from the thesis: adding Gaussian noise
        to the input before thresholding creates a stochastic activation
        whose expected value follows a sigmoid curve:

            P(output = ON) = Phi(input / noise_std)

        where Phi is the normal CDF. This is equivalent to training
        with a sigmoid activation of slope 1/noise_std.

        The noise effectively smooths the discrete energy landscape,
        allowing gradient-like learning on binary units.
        """
        noisy_inputs = inputs + torch.randn_like(inputs) * noise_std
        return self.threshold_activation(noisy_inputs)

    # ================================================================
    # Input calculation (matrix-vector multiply)
    # ================================================================

    def calc_inputs(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate net input to each unit: input[i] = sum_{j!=i} w[i][j] * s[j]

        This is the computational hotspot: O(N^2) matrix-vector multiply.
        PyTorch handles GPU acceleration automatically via torch.mv().
        """
        if state is None:
            state = self.state
        # Matrix-vector multiply (GPU-accelerated)
        self.unit_input = torch.mv(self.weights, state)
        # Zero diagonal contribution is already handled by weight matrix
        self.mean_unit_input = self.unit_input.abs().mean().item()
        return self.unit_input

    # ================================================================
    # Network relaxation
    # ================================================================

    def relax_synchronous(self, max_cycles: Optional[int] = None) -> int:
        """
        Synchronous relaxation: update all units simultaneously.

        More parallelisable than async, but may oscillate between
        states (2-cycles). The thesis handles this with a
        'lengthOfTimeStableInRelaxation' counter.
        """
        if max_cycles is None:
            max_cycles = self.cfg.max_cycles
        stable_time = 0
        cycle = 0

        while cycle < max_cycles and stable_time < self.cfg.stable_time_required:
            self.calc_inputs()

            if self.cfg.noise_during_relaxation:
                noise_std = self._get_relaxation_noise_std()
                new_state = self.soft_threshold_activation(
                    self.unit_input, noise_std)
            else:
                new_state = self.threshold_activation(self.unit_input)

            changes = (new_state != self.state).sum().item()
            self.state = new_state

            if changes == 0:
                stable_time += 1
            else:
                stable_time = 0
                cycle += 1

        return cycle

    def relax_asynchronous(self, max_cycles: Optional[int] = None) -> int:
        """
        Asynchronous relaxation: update one unit at a time.

        Guaranteed to converge to a fixed point (not a cycle) for
        symmetric weights. The unit to update is chosen randomly
        from those that want to change.
        """
        if max_cycles is None:
            max_cycles = self.cfg.max_cycles

        self.calc_inputs()
        cycle = 0

        while cycle < max_cycles:
            # Find unstable units
            desired = self.threshold_activation(self.unit_input)
            unstable_mask = (desired != self.state)
            unstable_indices = unstable_mask.nonzero(as_tuple=True)[0]

            if len(unstable_indices) == 0:
                break  # Converged

            # Pick a random unstable unit
            idx = unstable_indices[torch.randint(len(unstable_indices), (1,))].item()

            # Update that unit
            old_val = self.state[idx].clone()
            self.state[idx] = desired[idx]

            # Rank-1 update of inputs: O(N) instead of O(N^2)
            delta_activation = self.state[idx] - old_val
            self.unit_input += self.weights[:, idx] * delta_activation

            cycle += 1

        return cycle

    def relax(self, max_cycles: Optional[int] = None) -> int:
        """Relax using configured method."""
        if self.cfg.relaxation == RelaxationType.SYNCHRONOUS:
            return self.relax_synchronous(max_cycles)
        else:
            return self.relax_asynchronous(max_cycles)

    # ================================================================
    # Pattern stability
    # ================================================================

    def is_stable(self, pattern: torch.Tensor) -> bool:
        """Check if a pattern is a fixed point without modifying state."""
        inputs = torch.mv(self.weights, pattern)
        desired = self.threshold_activation(inputs)
        return torch.all(desired == pattern).item()

    def stability_stats(self, pattern: torch.Tensor) -> dict:
        """
        Compute stability statistics for a pattern.
        Returns the energy ratio measure from the thesis.
        """
        inputs = torch.mv(self.weights, pattern)
        desired = self.threshold_activation(inputs)

        # Signed stability: positive = correct, negative = wrong
        signed_stability = torch.where(
            desired == pattern,
            inputs.abs(),
            -inputs.abs()
        )

        sorted_stability = torch.sort(signed_stability).values
        n_ratio = max(1, int(self.N * self.cfg.units_in_ratio))

        low_sum = sorted_stability[:n_ratio].sum().item()
        high_sum = sorted_stability[-n_ratio:].sum().item()

        mean_stab = signed_stability.mean().item()
        std_stab = signed_stability.std().item()

        return {
            'stable': sorted_stability[0].item() >= 0,
            'least_stable': sorted_stability[0].item(),
            'most_stable': sorted_stability[-1].item(),
            'mean_stable': mean_stab,
            'std_dev': std_stab,
            'ratio': low_sum / high_sum if abs(high_sum) > 1e-10 else 0.0,
            'coeff_variation': std_stab / mean_stab if abs(mean_stab) > 1e-10 else float('inf'),
            'num_unstable': (signed_stability < 0).sum().item(),
        }

    # ================================================================
    # Energy landscape
    # ================================================================

    def energy(self, state: Optional[torch.Tensor] = None) -> float:
        """
        Hopfield energy: E = -0.5 * s^T W s

        The energy function monotonically decreases during relaxation
        (for symmetric weights), guaranteeing convergence.
        """
        if state is None:
            state = self.state
        return -0.5 * torch.dot(state, torch.mv(self.weights, state)).item()

    def energy_ratio(self, state: Optional[torch.Tensor] = None) -> float:
        """
        Energy ratio measure from the thesis.

        Compares current energy to min/max possible energy to discriminate
        learned patterns (low energy, high ratio) from spurious attractors.

        Ratio = (E_current - E_min) / (E_max - E_min)
        """
        if state is None:
            state = self.state
        inputs = torch.mv(self.weights, state)
        desired = self.threshold_activation(inputs)

        # Energy of unstable units (could flip to lower energy)
        unstable_energy = 0.0
        stable_energy = 0.0
        for i in range(self.N):
            if desired[i] != state[i]:
                unstable_energy += abs(inputs[i].item())
            else:
                stable_energy += abs(inputs[i].item())

        total = unstable_energy + stable_energy
        if total < 1e-10:
            return 0.0
        return stable_energy / total

    # ================================================================
    # Learning algorithms
    # ================================================================

    def hebbian_learn(self, patterns: torch.Tensor,
                      learning_rate: Optional[float] = None,
                      noise_level: Optional[float] = None) -> None:
        """
        Hebbian learning: dw[i][j] = lr * (s[i] == s[j] ? +1 : -1)

        With optional noise: each weight update has a probability of
        being corrupted proportional to the unit's input strength.
        This is the thesis's 'noisy Hebbian' variant.
        """
        if learning_rate is None:
            learning_rate = self.cfg.learning_rate
        if noise_level is None:
            noise_level = self.cfg.hebbian_noise_level

        for _ in range(self.cfg.hebbian_cycles):
            for p in range(patterns.shape[0]):
                pat = patterns[p]

                # Outer product: pat (x) pat gives +1 for same, -1 for different
                # For bipolar (+1/-1): outer_product = pat * pat^T
                outer = torch.outer(pat, pat)  # N x N

                if noise_level > 0:
                    # Noise mask: probability of corruption scales with input
                    inputs = torch.mv(self.weights, pat)
                    prob_noise = (inputs.abs() / self.N).clamp(0, 1)
                    noise_mask = torch.rand(self.N, device=self.device) >= \
                                 (noise_level * prob_noise)
                    noise_mask_2d = noise_mask.unsqueeze(1).float()
                    outer = outer * noise_mask_2d

                self.weights += learning_rate * outer

                # Zero diagonal (no self-connections)
                self.weights.fill_diagonal_(0)

                if self.cfg.symmetric_weights:
                    self._symmetrise_weights()
                if self.cfg.weight_decay:
                    self._apply_weight_decay()
                if self.cfg.weight_normalisation != 'none':
                    self._normalise_weights()

    def delta_learn(self, patterns: torch.Tensor,
                    start_idx: int = 0, end_idx: Optional[int] = None,
                    pseudo_patterns: Optional[torch.Tensor] = None,
                    real_rehearsal_patterns: Optional[torch.Tensor] = None,
                    verbose: bool = False) -> float:
        """
        Delta learning rule with online/batch modes.

        Supports pseudorehearsal: interleave learning of new patterns
        with rehearsal of pseudo-generated or real old patterns.

        The Gaussian noise on inputs during delta learning creates
        the soft-threshold effect described in the thesis: the
        effective activation function becomes a smooth sigmoid,
        enabling gradient-like weight updates on binary units.
        """
        if end_idx is None:
            end_idx = patterns.shape[0]

        is_initial = (start_idx == 0)
        total_epochs = (self.cfg.initial_training_epochs if is_initial
                       else self.cfg.training_epochs)

        lr = self.cfg.learning_rate
        total_error = 0.0

        for epoch in range(total_epochs):
            # Randomise pattern order for online learning
            order = torch.randperm(end_idx - start_idx) + start_idx
            cycle_error = 0.0
            total_error *= self.cfg.error_tail
            self.weight_deltas.zero_()

            # Learn target patterns
            for idx in order:
                pat = patterns[idx]
                error = self._delta_update_pattern(pat, lr, ratio=1.0,
                                                    is_pseudo=False)
                cycle_error += error

                if True:  # online mode
                    self._apply_weight_updates()

            # Pseudorehearsal or real rehearsal
            if real_rehearsal_patterns is not None and self.cfg.real_rehearsal:
                n_rehearse = max(1, int(start_idx * self.cfg.real_rehearsal_percent))
                indices = torch.randperm(real_rehearsal_patterns.shape[0])[:n_rehearse]
                for idx in indices:
                    pat = real_rehearsal_patterns[idx]
                    self._delta_update_pattern(pat, lr,
                                               ratio=self.cfg.learning_ratio_pseudo,
                                               is_pseudo=True)
                    self._apply_weight_updates()

            elif pseudo_patterns is not None and pseudo_patterns.shape[0] > 0:
                for i in range(pseudo_patterns.shape[0]):
                    pat = pseudo_patterns[i]
                    self._delta_update_pattern(pat, lr,
                                               ratio=self.cfg.learning_ratio_pseudo,
                                               is_pseudo=True)
                    self._apply_weight_updates()

            total_error += cycle_error

            # Temperature and learning rate decay
            if self.temperature > 0:
                self.temperature -= self.cfg.start_temperature / total_epochs
            lr -= (self.cfg.learning_rate - self.cfg.learning_rate_decay_to) / total_epochs

            if not self.cfg.all_epoch_learning and total_error < self.cfg.error_criteria:
                break

            if verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch}: error={total_error:.4f}, lr={lr:.4f}")

        return total_error

    def _delta_update_pattern(self, pattern: torch.Tensor,
                               learning_rate: float,
                               ratio: float = 1.0,
                               is_pseudo: bool = False) -> float:
        """Single pattern delta update step."""
        # Copy pattern to state
        self.state = pattern.clone()

        # Forward pass
        inputs = self.calc_inputs()

        # Add Gaussian noise (creates soft threshold / sigmoid-like activation)
        if self.cfg.noise_on_input:
            noise_scale = self.cfg.noise_ratio_pseudo if is_pseudo else 1.0
            if self.cfg.gaussian_noise_type == 'relative':
                std = self.mean_unit_input * self.cfg.gaussian_relative_range * noise_scale
            else:
                std = self.cfg.gaussian_absolute_range * noise_scale
            inputs = inputs + torch.randn_like(inputs) * std

        # Compute output
        output = self.threshold_activation(inputs)

        # Error: desired - actual
        error = pattern - output
        scalar_error = error.abs().mean().item()

        # Weight delta: outer product of error and input pattern
        self.weight_deltas += learning_rate * ratio * torch.outer(error, pattern)

        return scalar_error ** 2

    def _apply_weight_updates(self):
        """Apply accumulated weight deltas."""
        self.weights += self.weight_deltas
        self.weight_deltas.zero_()
        self.weights.fill_diagonal_(0)

        if self.cfg.symmetric_weights:
            self._symmetrise_weights()
        if self.cfg.weight_decay:
            self._apply_weight_decay()
        if self.cfg.weight_normalisation != 'none':
            self._normalise_weights()

    # ================================================================
    # Pseudorehearsal
    # ================================================================

    def generate_pseudo_patterns(self, learnt_patterns: torch.Tensor,
                                  verbose: bool = False) -> torch.Tensor:
        """
        Generate pseudoitems by probing the network with random states
        and relaxing to stable attractors.

        This is the core pseudorehearsal mechanism:
        1. Generate random initial state
        2. Relax to nearest attractor
        3. Filter using energy ratio (thesis contribution)
        4. Collect as pseudo-pattern for future rehearsal

        The energy ratio discriminates learned patterns from spurious
        attractors, improving pseudorehearsal quality.
        """
        pseudo_patterns = []
        count = 0
        cutoff = self.cfg.pseudo_rehearsal_cutoff

        while (len(pseudo_patterns) < self.cfg.max_pseudo_patterns and
               count < self.cfg.max_samples_for_pseudo):

            # Random initial state
            self.state = torch.where(
                torch.rand(self.N, device=self.device) < self.cfg.sample_prob_on,
                torch.full((self.N,), self.cfg.ON, device=self.device,
                          dtype=torch.float64),
                torch.full((self.N,), self.cfg.OFF, device=self.device,
                          dtype=torch.float64)
            )

            # Relax
            cycles = self.relax()

            if cycles < self.cfg.max_cycles:  # Converged
                # Energy ratio filter (thesis contribution)
                stats = self.stability_stats(self.state)

                if stats['ratio'] > cutoff or cutoff < 0:
                    pattern = self.state.clone()

                    # Check uniqueness
                    is_unique = True
                    if self.cfg.unique_pseudo_patterns:
                        for existing in pseudo_patterns:
                            if self._patterns_equal(pattern, existing):
                                is_unique = False
                                break

                    # Optionally remove learned patterns from pseudo set
                    is_learnt = False
                    if self.cfg.remove_learnt_from_pseudo:
                        for lp in learnt_patterns:
                            if self._patterns_equal(pattern, lp):
                                is_learnt = True
                                break

                    if is_unique and not is_learnt:
                        pseudo_patterns.append(pattern)

            count += 1

        if verbose:
            print(f"  Generated {len(pseudo_patterns)} pseudo-patterns "
                  f"from {count} samples")

        if len(pseudo_patterns) == 0:
            return torch.zeros(0, self.N, device=self.device, dtype=torch.float64)
        return torch.stack(pseudo_patterns)

    # ================================================================
    # Unlearning
    # ================================================================

    def unlearn(self, num_cycles: int = 10, num_steps: int = 10) -> int:
        """
        Unlearning: anti-Hebbian learning on spurious attractors.

        Generate random states, relax to attractors, and apply
        negative Hebbian learning to destabilise unwanted stable states.
        """
        count = 0
        for _ in range(num_cycles):
            for _ in range(num_steps):
                # Random probe
                self.state = torch.where(
                    torch.rand(self.N, device=self.device) < self.cfg.sample_prob_on,
                    torch.full((self.N,), self.cfg.ON, device=self.device,
                              dtype=torch.float64),
                    torch.full((self.N,), self.cfg.OFF, device=self.device,
                              dtype=torch.float64)
                )
                cycles = self.relax()

                if cycles < self.cfg.max_cycles:
                    # Anti-Hebbian: negative outer product
                    outer = torch.outer(self.state, self.state)
                    self.weights -= self.cfg.unlearning_const * outer
                    self.weights.fill_diagonal_(0)
                    count += 1

        return count

    # ================================================================
    # Sampling and analysis
    # ================================================================

    def sample_network(self, learnt_patterns: torch.Tensor,
                       num_samples: Optional[int] = None) -> dict:
        """
        Sample the network's attractor landscape.

        Generates random initial states, relaxes them, and categorises
        the resulting attractors as learned, spurious, or cycling.
        """
        if num_samples is None:
            num_samples = self.cfg.num_check

        learnt_freq = torch.zeros(learnt_patterns.shape[0], dtype=torch.int64)
        spurious_patterns = []
        spurious_freq = []
        cycling_count = 0

        for _ in range(num_samples):
            # Random initial state
            self.state = torch.where(
                torch.rand(self.N, device=self.device) < self.cfg.sample_prob_on,
                torch.full((self.N,), self.cfg.ON, device=self.device,
                          dtype=torch.float64),
                torch.full((self.N,), self.cfg.OFF, device=self.device,
                          dtype=torch.float64)
            )

            cycles = self.relax()

            if cycles >= self.cfg.max_cycles:
                cycling_count += 1
                continue

            # Check against learned patterns
            found_learnt = False
            for i, lp in enumerate(learnt_patterns):
                if self._patterns_equal(self.state, lp):
                    learnt_freq[i] += 1
                    found_learnt = True
                    break

            if not found_learnt:
                # Check against known spurious patterns
                found_spurious = False
                for j, sp in enumerate(spurious_patterns):
                    if self._patterns_equal(self.state, sp):
                        spurious_freq[j] += 1
                        found_spurious = True
                        break

                if not found_spurious:
                    spurious_patterns.append(self.state.clone())
                    spurious_freq.append(1)

        return {
            'learnt_freq': learnt_freq,
            'num_spurious': len(spurious_patterns),
            'spurious_patterns': spurious_patterns,
            'spurious_freq': spurious_freq,
            'cycling_count': cycling_count,
        }

    def sample_basin(self, attractor: torch.Tensor,
                     num_samples: int = 250) -> float:
        """
        Estimate basin of attraction size using adaptive search.

        Returns overlap level at which ~50% of perturbed states
        still relax back to the attractor (0.5 = entire space,
        1.0 = only the pattern itself).
        """
        current_overlap = 0.75
        branching = 0.025

        num_success = 0
        num_failure = 0
        mean_success = 0.0
        mean_failure = 0.0
        testing_start = num_samples - 100

        for i in range(num_samples):
            # Create partial pattern
            self.state = attractor.clone()
            num_to_flip = int(self.N * (1 - current_overlap))
            if num_to_flip > 0:
                flip_indices = torch.randperm(self.N)[:num_to_flip]
                self.state[flip_indices] = torch.where(
                    self.state[flip_indices] == self.cfg.ON,
                    torch.full_like(self.state[flip_indices], self.cfg.OFF),
                    torch.full_like(self.state[flip_indices], self.cfg.ON)
                )

            # Relax
            self.relax()

            # Check if recovered attractor
            success = self._patterns_equal(self.state, attractor)

            branching *= 0.98

            if success:
                if i > testing_start:
                    mean_success += current_overlap
                    num_success += 1
                current_overlap -= branching
                current_overlap = max(0.5, current_overlap)
            else:
                if i > testing_start:
                    mean_failure += current_overlap
                    num_failure += 1
                current_overlap += branching
                current_overlap = min(1.0, current_overlap)

        total = num_success + num_failure
        if total > 0:
            return (mean_success + mean_failure) / total
        return current_overlap

    # ================================================================
    # Batch operations (GPU-optimised)
    # ================================================================

    def batch_relax(self, patterns: torch.Tensor,
                    max_cycles: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Relax multiple patterns in parallel using synchronous update.

        This is where PyTorch really shines: the matrix-vector multiply
        for all patterns becomes a matrix-matrix multiply, fully utilising
        the GPU.

        patterns: (num_patterns, N) tensor
        Returns: (relaxed_patterns, cycle_counts)
        """
        if max_cycles is None:
            max_cycles = self.cfg.max_cycles

        num_patterns = patterns.shape[0]
        states = patterns.clone()
        cycles = torch.zeros(num_patterns, dtype=torch.int64, device=self.device)
        active = torch.ones(num_patterns, dtype=torch.bool, device=self.device)

        for cycle in range(max_cycles):
            if not active.any():
                break

            # Batch matrix-vector multiply: (num_active, N) x (N, N)^T
            # = batch of N-dimensional dot products
            active_states = states[active]
            inputs = torch.mm(active_states, self.weights.T)  # (num_active, N)

            new_states = torch.where(inputs > 0,
                                     torch.full_like(inputs, self.cfg.ON),
                                     torch.full_like(inputs, self.cfg.OFF))

            # Check convergence
            changes = (new_states != active_states).any(dim=1)
            active_indices = active.nonzero(as_tuple=True)[0]

            states[active] = new_states
            cycles[active_indices[changes]] = cycle + 1

            # Deactivate converged patterns
            converged = ~changes
            active[active_indices[converged]] = False

        return states, cycles

    def batch_stability_check(self, patterns: torch.Tensor) -> torch.Tensor:
        """
        Check stability of multiple patterns in parallel.
        Returns boolean tensor: True if pattern is stable.
        """
        inputs = torch.mm(patterns, self.weights.T)  # (num, N)
        desired = torch.where(inputs > 0,
                             torch.full_like(inputs, self.cfg.ON),
                             torch.full_like(inputs, self.cfg.OFF))
        return (desired == patterns).all(dim=1)

    # ================================================================
    # Utility methods
    # ================================================================

    def _patterns_equal(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """Check if two patterns are identical or inverse."""
        identical = torch.all(torch.abs(a - b) < 1e-10).item()
        inverse = torch.all(torch.abs(a + b) < 1e-10).item()
        return identical or inverse

    def _symmetrise_weights(self):
        """Make weight matrix symmetric: W = (W + W^T) / 2"""
        self.weights = (self.weights + self.weights.T) / 2

    def _apply_weight_decay(self):
        """Multiply all weights by (1 - decay_value)."""
        self.weights *= (1.0 - self.cfg.weight_decay_value)

    def _normalise_weights(self):
        """Normalise weights according to configuration."""
        if self.cfg.weight_normalisation == 'unit':
            # Normalise incoming weights per unit
            norms = self.weights.abs().sum(dim=1, keepdim=True)
            scale = self.cfg.weight_norm_value / norms
            scale = torch.clamp(scale, max=1.0)
            self.weights *= scale
        elif self.cfg.weight_normalisation == 'net':
            # Normalise total network weight
            total = self.weights.abs().sum()
            target = self.cfg.weight_norm_value * self.N
            if total > target:
                self.weights *= target / total

    def _get_relaxation_noise_std(self) -> float:
        """Get noise standard deviation for relaxation."""
        if self.cfg.gaussian_noise_type == 'relative':
            return self.mean_unit_input * self.cfg.gaussian_relative_range
        else:
            return self.cfg.gaussian_absolute_range

    # ================================================================
    # Pattern generation
    # ================================================================

    @staticmethod
    def generate_random_patterns(n_patterns: int, n_units: int,
                                  prob_on: float = 0.5,
                                  device: str = 'cpu') -> torch.Tensor:
        """Generate random bipolar patterns."""
        return torch.where(
            torch.rand(n_patterns, n_units, device=device) < prob_on,
            torch.ones(n_patterns, n_units, device=device, dtype=torch.float64),
            -torch.ones(n_patterns, n_units, device=device, dtype=torch.float64)
        )

    def set_state(self, pattern: torch.Tensor):
        """Set the network state to a given pattern."""
        self.state = pattern.clone()


# ================================================================
# Experiment runner
# ================================================================

class ExperimentRunner:
    """
    Runs the thesis experiments: sequential learning with
    catastrophic forgetting measurement.
    """

    def __init__(self, config: HopfieldConfig):
        self.cfg = config
        self.net = HopfieldNetwork(config)

    def run_sequential_learning(self,
                                 patterns: torch.Tensor,
                                 verbose: bool = True) -> List[dict]:
        """
        Main experiment: learn patterns incrementally and measure
        stability after each addition.

        This replicates the core thesis experiment:
        1. Learn initial batch
        2. Add one pattern at a time
        3. After each addition, check which old patterns survive
        4. Record stability, basin sizes, spurious attractor counts
        """
        results = []
        n_patterns = patterns.shape[0]

        for current_top in range(self.cfg.initial_num_patterns,
                                  min(n_patterns, self.cfg.max_learnt_patterns) + 1,
                                  self.cfg.step):
            if verbose:
                print(f"\n--- Learning patterns 0..{current_top-1} ---")

            current_patterns = patterns[:current_top]

            if self.cfg.learning_type == LearningType.HEBBIAN:
                self.net.reset_weights()
                self.net.hebbian_learn(current_patterns)

            elif self.cfg.learning_type == LearningType.DELTA:
                self.net.reset_weights()
                self.net.delta_learn(current_patterns, verbose=verbose)

            elif self.cfg.learning_type == LearningType.PSEUDO_DELTA:
                if current_top == self.cfg.initial_num_patterns:
                    # Initial learning
                    self.net.reset_weights()
                    self.net.delta_learn(current_patterns, verbose=verbose)
                else:
                    # Generate pseudo patterns before learning new one
                    pseudo = self.net.generate_pseudo_patterns(
                        patterns[:current_top - self.cfg.step],
                        verbose=verbose)

                    if self.cfg.real_rehearsal:
                        self.net.delta_learn(
                            current_patterns,
                            start_idx=current_top - self.cfg.step,
                            real_rehearsal_patterns=patterns[:current_top - self.cfg.step],
                            verbose=verbose)
                    else:
                        self.net.delta_learn(
                            current_patterns,
                            start_idx=current_top - self.cfg.step,
                            pseudo_patterns=pseudo,
                            verbose=verbose)

            elif self.cfg.learning_type == LearningType.UNLEARNING:
                self.net.reset_weights()
                self.net.hebbian_learn(current_patterns)
                self.net.unlearn()

            # Test all learned patterns
            stability = self.net.batch_stability_check(current_patterns)
            num_stable = stability.sum().item()

            # Stability statistics for each pattern
            pattern_stats = []
            for i in range(current_top):
                stats = self.net.stability_stats(current_patterns[i])
                stats['pattern_idx'] = i
                stats['stable'] = stability[i].item()
                pattern_stats.append(stats)

            # Sample network
            sample_results = self.net.sample_network(current_patterns)

            result = {
                'num_patterns': current_top,
                'num_stable': num_stable,
                'fraction_stable': num_stable / current_top,
                'num_spurious': sample_results['num_spurious'],
                'cycling_count': sample_results['cycling_count'],
                'pattern_stats': pattern_stats,
                'learnt_freq': sample_results['learnt_freq'],
            }

            if self.cfg.calc_learnt_basins:
                basins = []
                for i in range(current_top):
                    if stability[i]:
                        basin = self.net.sample_basin(current_patterns[i])
                        basins.append(basin)
                    else:
                        basins.append(0.0)
                result['basin_sizes'] = basins

            results.append(result)

            if verbose:
                print(f"  Stable: {num_stable}/{current_top} "
                      f"({100*num_stable/current_top:.1f}%), "
                      f"Spurious: {sample_results['num_spurious']}")

        return results


# ================================================================
# Modern continual learning comparisons
# ================================================================

class EWCHopfield(HopfieldNetwork):
    """
    Elastic Weight Consolidation applied to Hopfield networks.

    Modern equivalent of pseudorehearsal: instead of rehearsing
    pseudo-patterns, EWC adds a penalty to weight changes that
    would destroy important connections.

    Fisher Information diagonal approximates which weights are
    important for previously learned patterns.
    """

    def __init__(self, config: HopfieldConfig):
        super().__init__(config)
        self.fisher = torch.zeros_like(self.weights)
        self.saved_weights = torch.zeros_like(self.weights)
        self.ewc_lambda = 1.0

    def compute_fisher(self, patterns: torch.Tensor):
        """
        Estimate Fisher Information Matrix (diagonal) from
        the stability of stored patterns.

        For Hopfield networks, the "importance" of a weight w[i][j]
        is related to how much the stability of stored patterns
        depends on that weight.
        """
        self.fisher.zero_()
        for p in range(patterns.shape[0]):
            pat = patterns[p]
            inputs = torch.mv(self.weights, pat)

            # Gradient of stability w.r.t. weights
            # d(stability_i)/d(w[i][j]) = sign(input[i]) * pat[j]
            for i in range(self.N):
                if abs(inputs[i].item()) < self.mean_unit_input:
                    # Near decision boundary: high importance
                    grad = pat.abs()  # |pattern[j]| for all j
                    self.fisher[i] += grad ** 2

        self.fisher /= patterns.shape[0]
        self.saved_weights = self.weights.clone()

    def ewc_penalty(self) -> torch.Tensor:
        """EWC regularisation term: lambda * F * (w - w*)^2"""
        return self.ewc_lambda * (self.fisher *
                                   (self.weights - self.saved_weights) ** 2).sum()

    def delta_learn_ewc(self, patterns: torch.Tensor,
                         start_idx: int = 0, **kwargs) -> float:
        """Delta learning with EWC regularisation."""
        # Standard delta learning
        error = self.delta_learn(patterns, start_idx=start_idx, **kwargs)

        # Apply EWC penalty as weight correction
        if self.fisher.sum() > 0:
            ewc_grad = 2 * self.ewc_lambda * self.fisher * \
                       (self.weights - self.saved_weights)
            self.weights -= self.cfg.learning_rate * ewc_grad
            self.weights.fill_diagonal_(0)

        return error


class PackNetHopfield(HopfieldNetwork):
    """
    PackNet-style approach: prune and freeze weights after learning
    each task. Remaining free weights learn new patterns.

    This is a modern structured approach to preventing forgetting:
    important weights are identified and frozen.
    """

    def __init__(self, config: HopfieldConfig):
        super().__init__(config)
        self.frozen_mask = torch.zeros_like(self.weights, dtype=torch.bool)

    def prune_and_freeze(self, keep_fraction: float = 0.5):
        """
        After learning, identify important weights (by magnitude),
        freeze them, and prune the rest.
        """
        free_weights = self.weights[~self.frozen_mask]
        if free_weights.numel() == 0:
            return

        threshold = torch.quantile(free_weights.abs(),
                                    1.0 - keep_fraction)

        # Freeze important weights
        important = (self.weights.abs() >= threshold) & ~self.frozen_mask
        self.frozen_mask |= important

        # Zero out unimportant free weights
        unimportant = ~self.frozen_mask
        self.weights[unimportant] = 0

    def _apply_weight_updates(self):
        """Override to prevent updates to frozen weights."""
        # Only update non-frozen weights
        updates = self.weight_deltas.clone()
        updates[self.frozen_mask] = 0
        self.weights += updates
        self.weight_deltas.zero_()
        self.weights.fill_diagonal_(0)

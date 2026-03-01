# Neurothermo

A production library for thermodynamic monitoring of neural network training based on the Algorithmic Thermodynamics framework.

## Overview

Neurothermo implements phase transition detection during neural network training by treating the optimization process as a thermodynamic system. The library monitors metrics that indicate whether a model is crystallizing (learning the underlying algorithm) or remaining in a glass state (memorizing without generalization).

### Key Concepts

| Phase | Delta | Interpretation |
|-------|-------|----------------|
| Crystal | < 0.1 | Weights discretized to integer lattice. Model learned the algorithm. |
| Glass | > 0.4 | Amorphous weights. Model may be memorizing. |
| Transition | 0.1 - 0.4 | Between crystalline and glassy states. |
| Liquid | r < 0.42 | Localized (MBL) behavior. Disordered but useful. |
| Topological Insulator | Non-zero Berry phase | Robust generalization properties. |

## Installation

```bash
pip install neurothermo
```

Or copy `neurothermo.py` to your project.

Dependencies:
- numpy
- torch (optional, for PyTorch integration)

## Quick Start

```python
import torch
import torch.nn as nn
from neurothermo import create_monitor

# Your model
model = nn.Linear(100, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Create monitor
monitor = create_monitor(model)

# Training loop
for epoch in range(10):
    x = torch.randn(32, 100)
    y = torch.randn(32, 10)

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # Call after backward(), before optimizer.step()
    metrics = monitor.step(loss=loss.item())
    optimizer.step()

    print(f"Epoch {epoch}: Delta={metrics.get('delta'):.4f}, Phase={metrics.phase.name}")

# Get all metrics at the end
print(monitor.summary())
```

## Performance

During training, only `delta` (O(n)) is computed per step, ensuring minimal overhead. All 17 metrics are computed from accumulated history when `summary()` is called.

```
Typical performance: ~0.2ms per step for 1000 parameters
```

## Metrics

### Phase Detection Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Discretization Margin | delta | Maximum distance of weights from nearest integer. Crystal: ~0, Glass: ~0.5 |
| Purity Index | alpha | Alignment with discrete structure. High value indicates proximity to crystal. |
| Local Complexity | lc | Fraction of active parameters. |
| Vacuum Core | vacuum | Fraction of near-zero weights. |
| Health Score | health | Composite metric for checkpoint verification. |

### Gradient Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Gradient Condition Number | kappa | Ratio of eigenvalues of gradient covariance. Crystal: ~1, Glass: infinity |
| Effective Temperature | t_eff | Magnitude of gradient fluctuations. |
| Ricci Scalar | ricci | Curvature of the weight manifold. |
| Participation Ratio | pr | Number of contributing modes. |
| Learning Uncertainty | hbar | Gradient variance ratio. |

### Topological Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Berry Phase | berry_phase_total | Accumulated geometric phase. Non-zero indicates topological protection. |
| Level Spacing Ratio | r | MBL diagnostic. 0.38: localized, 0.53: ergodic. |
| Algorithmic Gravitational | g_alg | Falls to 0 during crystallization. |
| Synthetic Planck Constant | hbar_eff | Emergent quantum scale. |

### Other Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Differential Entropy | entropy | Disorder in weight space. Crystal: ~0 |
| Heat Capacity | c_v | Peak indicates phase transition. |
| Overlap Coefficient | psi | Feature entanglement measure. |

## API Reference

### create_monitor()

```python
monitor = create_monitor(model, window_size=50)
```

Create a ThermoMonitor instance.

**Parameters:**
- `model` (nn.Module, optional): PyTorch model to monitor
- `window_size` (int): Buffer size for gradient history

**Returns:** ThermoMonitor instance

### ThermoMonitor.step()

```python
metrics = monitor.step(loss=loss_value)
```

Call after `loss.backward()` and before `optimizer.step()`.

**Parameters:**
- `loss` (float, optional): Current loss value

**Returns:** MetricsResult with `delta`, `alpha`, `health`, and `phase`

### ThermoMonitor.step_manual()

```python
metrics = monitor.step_manual(weights, gradients, loss=loss_value)
```

Manual step for non-PyTorch frameworks.

**Parameters:**
- `weights` (np.ndarray): Flattened weight array
- `gradients` (np.ndarray, optional): Flattened gradient array
- `loss` (float, optional): Current loss value

**Returns:** MetricsResult

### ThermoMonitor.summary()

```python
print(monitor.summary())
```

Compute all 17 metrics from accumulated history and return formatted summary string.

### ThermoMonitor.compute_all_metrics()

```python
all_metrics = monitor.compute_all_metrics()
```

Compute all metrics and return as dictionary.

### MetricsResult

```python
delta = metrics.get('delta', 0.0)
phase = metrics.phase
all_values = metrics.to_dict()
```

## Configuration

Create a `config.toml` file:

```toml
[core]
window_size = 50
enable_logging = false

[thresholds]
delta_crystal_threshold = 0.1
delta_glass_threshold = 0.4
purity_high_threshold = 7.0
zero_epsilon = 1.0e-6
numerical_stability_epsilon = 1.0e-9

[computation]
gradient_buffer_window = 50
```

Load configuration:

```python
from neurothermo import NeurothermoConfig, ThermoMonitor

config = NeurothermoConfig.from_toml("config.toml")
monitor = ThermoMonitor(model, config)
```

## Utility Functions

```python
from neurothermo import extract_weights, extract_gradients

weights = extract_weights(model)      # np.ndarray
gradients = extract_gradients(model)  # np.ndarray or None
```

## Phase Detection Logic

```
if delta < 0.1:
    return CRYSTAL
elif delta > 0.4:
    return GLASS
else:
    return TRANSITION
```

Full phase detection in `summary()` also considers:
- Berry phase for topological insulator detection
- Level spacing ratio for liquid (MBL) state

## Example Output

```
======================================================================
NEUROTHERMO SUMMARY
======================================================================
Steps: 100, Epochs: 10

ALL METRICS:
--------------------------------------------------
  delta: 0.496850
  alpha: 0.699466
  lc: 0.987000
  vacuum: 0.000000
  health: 0.383997
  kappa: 3.624279
  t_eff: 10.103856
  ricci: 1080.978862
  pr: 89.898342
  hbar: 0.022767
  berry_phase_total: 182.212374
  r: 0.382966
  g_alg: 120.558213
  hbar_eff: 0.000000
  entropy: 3.352495
  psi: 1.000000

Final Phase: GLASS
  Glass: amorphous weights. Model may be memorizing.

DELTA STATISTICS:
  mean: 0.4994, std: 0.0006
  range: [0.4969, 0.5000]

PHASE DISTRIBUTION:
  GLASS: 100 (100.0%)

======================================================================
```

## Theoretical Background

This library is based on the Algorithmic Thermodynamics framework, which treats neural network training as a thermodynamic process:

1. **Discretization Margin (delta)**: Weights that have learned an algorithm tend to cluster around discrete values (integers), while memorizing models maintain amorphous weight distributions.

2. **Phase Transitions**: During training, models can undergo phase transitions from glass (disordered) to crystal (ordered) states, analogous to physical systems.

3. **Topological Protection**: Non-trivial Berry phases indicate topological properties that confer robustness to the learned representations.

4. **Many-Body Localization**: The level spacing ratio diagnoses whether the system exhibits localized (useful for computation) or ergodic (thermalized) behavior.

## License

MIT License

## References

The metrics and framework are derived from the Algorithmic Thermodynamics framework for neural network training analysis.

[1] grisun0. Algorithmic Induction via Structural Weight Transfer. Zenodo, 2026. https://doi.org/10.5281/zenodo.18072858

[2] grisun0. From Boltzmann Stochasticity to Hamiltonian Integrability: Emergence of Topological Crystals and Synthetic Planck Constants. Zenodo, 2026. https://doi.org/10.5281/zenodo.18407920

[3] grisun0. Thermodynamic Grokking in Binary Parity (k=3): A First Look at 100 Seeds. Zenodo, 2026. https://doi.org/10.5281/zenodo.18489853

[4] grisun0. Schrödinger Topological Crystallization: Phase Space Discovery in Hamiltonian Neural Networks. Zenodo, 2026. https://doi.org/10.5281/zenodo.18725428

[5] grisun0. Constraint Preservation in a Neural Quantum Simulator. Zenodo, 2026. https://doi.org/10.5281/zenodo.18795537

[6] grisun0. The Dirac Discrete Crystal. Zenodo, 2026. https://doi.org/10.5281/zenodo.18810160




![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)

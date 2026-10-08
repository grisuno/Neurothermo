# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `ComputeConfig`, `CoreConfig`, `MetricsResult`, `NeurothermoConfig`, `PhaseState`, `ThermoMonitor`, `ThresholdConfig`, `__init__`. Core file: `neurothermo.py` (30 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: 28/02/2026 Licencia: GPL v3  Descripción:  QUICK START - neurother.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 0 | yes |
| `neurothermo.py` | py | utility | 30 | yes |

## Key Symbols

- `PhaseState` (class, `neurothermo.py:32`) `class PhaseState(Enum)` - Thermodynamic phase states.
- `ThresholdConfig` (class, `neurothermo.py:43`) `class ThresholdConfig`
- `ComputeConfig` (class, `neurothermo.py:53`) `class ComputeConfig`
- `CoreConfig` (class, `neurothermo.py:58`) `class CoreConfig`
- `NeurothermoConfig` (class, `neurothermo.py:65`) `class NeurothermoConfig`
- `from_toml` (method, `neurothermo.py:71`) `def from_toml(cls, path)`
- `MetricsResult` (class, `neurothermo.py:84`) `class MetricsResult` - Container for step metrics (only delta/alpha/health/phase during training).
- `__init__` (method, `neurothermo.py:87`) `def __init__(self, metrics, phase)`
- `get` (method, `neurothermo.py:91`) `def get(self, name, default)`
- `to_dict` (method, `neurothermo.py:94`) `def to_dict(self)`
- `phase` (method, `neurothermo.py:98`) `def phase(self)`
- `_detect_phase` (method, `neurothermo.py:102`) `def _detect_phase(delta)` - Fast phase detection from delta only.
- `ThermoMonitor` (class, `neurothermo.py:111`) `class ThermoMonitor` - Monitoring class.
- `__init__` (method, `neurothermo.py:118`) `def __init__(self, model, config)`
- `_setup_logger` (method, `neurothermo.py:135`) `def _setup_logger(self)`
- `_extract_weights` (method, `neurothermo.py:144`) `def _extract_weights(self)`
- `_extract_gradients` (method, `neurothermo.py:151`) `def _extract_gradients(self)`
- `step` (method, `neurothermo.py:157`) `def step(self, loss)` - Fast step: ONLY computes delta. Stores data for final metrics.
- `step_manual` (method, `neurothermo.py:163`) `def step_manual(self, weights, gradients, loss)` - Manual step with provided arrays.
- `_do_step` (method, `neurothermo.py:172`) `def _do_step(self, weights, gradients, loss)` - Compute ONLY delta. Everything else deferred to summary().
- `epoch_end` (method, `neurothermo.py:208`) `def epoch_end(self)`
- `get_phase_description` (method, `neurothermo.py:211`) `def get_phase_description(self, phase)`
- `reset` (method, `neurothermo.py:222`) `def reset(self)`
- `compute_all_metrics` (method, `neurothermo.py:232`) `def compute_all_metrics(self)` - Compute ALL 17 metrics from history. Call after training.
- `summary` (method, `neurothermo.py:329`) `def summary(self)` - Generate summary with ALL metrics.
- `step_count` (method, `neurothermo.py:381`) `def step_count(self)`
- `last_result` (method, `neurothermo.py:385`) `def last_result(self)`
- `create_monitor` (method, `neurothermo.py:389`) `def create_monitor(model, window_size)` - Create monitor. During training only delta is computed (fast).
- `extract_weights` (method, `neurothermo.py:398`) `def extract_weights(model)`
- `extract_gradients` (method, `neurothermo.py:404`) `def extract_gradients(model)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 1
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 1 (strength 0.5): Inferred shared context (layer utility) with no import path between community 0 (root) and community 1 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `neurothermo.py`

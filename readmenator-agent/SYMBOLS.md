# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `ComputeConfig` | class | `neurothermo.py:53` | `class ComputeConfig` |
| `CoreConfig` | class | `neurothermo.py:58` | `class CoreConfig` |
| `MetricsResult` | class | `neurothermo.py:84` | `class MetricsResult` |
| `NeurothermoConfig` | class | `neurothermo.py:65` | `class NeurothermoConfig` |
| `PhaseState` | class | `neurothermo.py:32` | `class PhaseState(Enum)` |
| `ThermoMonitor` | class | `neurothermo.py:111` | `class ThermoMonitor` |
| `ThresholdConfig` | class | `neurothermo.py:43` | `class ThresholdConfig` |
| `__init__` | method | `neurothermo.py:87` | `def __init__(self, metrics, phase)` |
| `__init__` | method | `neurothermo.py:118` | `def __init__(self, model, config)` |
| `_detect_phase` | method | `neurothermo.py:102` | `def _detect_phase(delta)` |
| `_do_step` | method | `neurothermo.py:172` | `def _do_step(self, weights, gradients, loss)` |
| `_extract_gradients` | method | `neurothermo.py:151` | `def _extract_gradients(self)` |
| `_extract_weights` | method | `neurothermo.py:144` | `def _extract_weights(self)` |
| `_setup_logger` | method | `neurothermo.py:135` | `def _setup_logger(self)` |
| `compute_all_metrics` | method | `neurothermo.py:232` | `def compute_all_metrics(self)` |
| `create_monitor` | method | `neurothermo.py:389` | `def create_monitor(model, window_size)` |
| `epoch_end` | method | `neurothermo.py:208` | `def epoch_end(self)` |
| `extract_gradients` | method | `neurothermo.py:404` | `def extract_gradients(model)` |
| `extract_weights` | method | `neurothermo.py:398` | `def extract_weights(model)` |
| `from_toml` | method | `neurothermo.py:71` | `def from_toml(cls, path)` |
| `get` | method | `neurothermo.py:91` | `def get(self, name, default)` |
| `get_phase_description` | method | `neurothermo.py:211` | `def get_phase_description(self, phase)` |
| `last_result` | method | `neurothermo.py:385` | `def last_result(self)` |
| `phase` | method | `neurothermo.py:98` | `def phase(self)` |
| `reset` | method | `neurothermo.py:222` | `def reset(self)` |
| `step` | method | `neurothermo.py:157` | `def step(self, loss)` |
| `step_count` | method | `neurothermo.py:381` | `def step_count(self)` |
| `step_manual` | method | `neurothermo.py:163` | `def step_manual(self, weights, gradients, loss)` |
| `summary` | method | `neurothermo.py:329` | `def summary(self)` |
| `to_dict` | method | `neurothermo.py:94` | `def to_dict(self)` |

# Subsystem: root

## app.py
- Layer: utility
- Doc: _*_ coding: utf8 _*_
- Language: py
- Depends on: `neurothermo.py`

## install.sh
- Layer: utility
- Language: sh

## neurothermo.py
- Layer: utility
- Language: py
- Symbols:
  - `PhaseState` (class, line 32) `class PhaseState(Enum)`
  - `ThresholdConfig` (class, line 43) `class ThresholdConfig`
  - `ComputeConfig` (class, line 53) `class ComputeConfig`
  - `CoreConfig` (class, line 58) `class CoreConfig`
  - `NeurothermoConfig` (class, line 65) `class NeurothermoConfig`
  - `MetricsResult` (class, line 84) `class MetricsResult`
  - `_detect_phase` (method, line 102) `def _detect_phase(delta)`
  - `ThermoMonitor` (class, line 111) `class ThermoMonitor`
  - `create_monitor` (method, line 389) `def create_monitor(model, window_size)`
  - `extract_weights` (method, line 398) `def extract_weights(model)`
  - `extract_gradients` (method, line 404) `def extract_gradients(model)`
  - `from_toml` (method, line 71) `def from_toml(cls, path)`
  - `__init__` (method, line 87) `def __init__(self, metrics, phase)`
  - `get` (method, line 91) `def get(self, name, default)`
  - `to_dict` (method, line 94) `def to_dict(self)`
  - `phase` (method, line 98) `def phase(self)`
  - `__init__` (method, line 118) `def __init__(self, model, config)`
  - `_setup_logger` (method, line 135) `def _setup_logger(self)`
  - `_extract_weights` (method, line 144) `def _extract_weights(self)`
  - `_extract_gradients` (method, line 151) `def _extract_gradients(self)`
  - `step` (method, line 157) `def step(self, loss)`
  - `step_manual` (method, line 163) `def step_manual(self, weights, gradients, loss)`
  - `_do_step` (method, line 172) `def _do_step(self, weights, gradients, loss)`
  - `epoch_end` (method, line 208) `def epoch_end(self)`
  - `get_phase_description` (method, line 211) `def get_phase_description(self, phase)`
  - `reset` (method, line 222) `def reset(self)`
  - `compute_all_metrics` (method, line 232) `def compute_all_metrics(self)`
  - `summary` (method, line 329) `def summary(self)`
  - `step_count` (method, line 381) `def step_count(self)`
  - `last_result` (method, line 385) `def last_result(self)`
- Imported by: `app.py`

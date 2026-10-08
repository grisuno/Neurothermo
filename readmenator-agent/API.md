# API

## neurothermo.py
Imported by: `app.py`
- `NeurothermoConfig.from_toml` (method) `neurothermo.py:71` `def from_toml(cls, path)`
- `MetricsResult.__init__` (method) `neurothermo.py:87` `def __init__(self, metrics, phase)`
- `MetricsResult.get` (method) `neurothermo.py:91` `def get(self, name, default)`
- `MetricsResult.to_dict` (method) `neurothermo.py:94` `def to_dict(self)`
- `MetricsResult.phase` (method) `neurothermo.py:98` `def phase(self)`
- `ThermoMonitor.__init__` (method) `neurothermo.py:118` `def __init__(self, model, config)`
- `ThermoMonitor.step` (method) `neurothermo.py:157` `def step(self, loss)` -- Fast step: ONLY computes delta.
- `ThermoMonitor.step_manual` (method) `neurothermo.py:163` `def step_manual(self, weights, gradients, loss)` -- Manual step with provided arrays.
- `ThermoMonitor.epoch_end` (method) `neurothermo.py:208` `def epoch_end(self)`
- `ThermoMonitor.get_phase_description` (method) `neurothermo.py:211` `def get_phase_description(self, phase)`
- `ThermoMonitor.reset` (method) `neurothermo.py:222` `def reset(self)`
- `ThermoMonitor.compute_all_metrics` (method) `neurothermo.py:232` `def compute_all_metrics(self)` -- Compute ALL 17 metrics from history.
- `ThermoMonitor.summary` (method) `neurothermo.py:329` `def summary(self)` -- Generate summary with ALL metrics.
- `ThermoMonitor.step_count` (method) `neurothermo.py:381` `def step_count(self)`
- `ThermoMonitor.last_result` (method) `neurothermo.py:385` `def last_result(self)`
- `ThermoMonitor.create_monitor` (method) `neurothermo.py:389` `def create_monitor(model, window_size)` -- Create monitor.
- `ThermoMonitor.extract_weights` (method) `neurothermo.py:398` `def extract_weights(model)`
- `ThermoMonitor.extract_gradients` (method) `neurothermo.py:404` `def extract_gradients(model)`

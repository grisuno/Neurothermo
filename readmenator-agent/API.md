# API

## neurothermo.py

### _detect_phase `def _detect_phase(delta)`
- Defined: `neurothermo.py:102`
- Doc: Fast phase detection from delta only.
- Imported by: `app.py`

### create_monitor `def create_monitor(model, window_size)`
- Defined: `neurothermo.py:389`
- Doc: Create monitor. During training only delta is computed (fast).
- Imported by: `app.py`

### extract_weights `def extract_weights(model)`
- Defined: `neurothermo.py:398`
- Imported by: `app.py`

### extract_gradients `def extract_gradients(model)`
- Defined: `neurothermo.py:404`
- Imported by: `app.py`

### from_toml `def from_toml(cls, path)`
- Defined: `neurothermo.py:71`
- Imported by: `app.py`

### __init__ `def __init__(self, metrics, phase)`
- Defined: `neurothermo.py:87`
- Imported by: `app.py`

### get `def get(self, name, default)`
- Defined: `neurothermo.py:91`
- Imported by: `app.py`

### to_dict `def to_dict(self)`
- Defined: `neurothermo.py:94`
- Imported by: `app.py`

### phase `def phase(self)`
- Defined: `neurothermo.py:98`
- Imported by: `app.py`

### __init__ `def __init__(self, model, config)`
- Defined: `neurothermo.py:118`
- Imported by: `app.py`

### _setup_logger `def _setup_logger(self)`
- Defined: `neurothermo.py:135`
- Imported by: `app.py`

### _extract_weights `def _extract_weights(self)`
- Defined: `neurothermo.py:144`
- Imported by: `app.py`

### _extract_gradients `def _extract_gradients(self)`
- Defined: `neurothermo.py:151`
- Imported by: `app.py`

### step `def step(self, loss)`
- Defined: `neurothermo.py:157`
- Doc: Fast step: ONLY computes delta. Stores data for final metrics.
- Imported by: `app.py`

### step_manual `def step_manual(self, weights, gradients, loss)`
- Defined: `neurothermo.py:163`
- Doc: Manual step with provided arrays.
- Imported by: `app.py`

### _do_step `def _do_step(self, weights, gradients, loss)`
- Defined: `neurothermo.py:172`
- Doc: Compute ONLY delta. Everything else deferred to summary().
- Imported by: `app.py`

### epoch_end `def epoch_end(self)`
- Defined: `neurothermo.py:208`
- Imported by: `app.py`

### get_phase_description `def get_phase_description(self, phase)`
- Defined: `neurothermo.py:211`
- Imported by: `app.py`

### reset `def reset(self)`
- Defined: `neurothermo.py:222`
- Imported by: `app.py`

### compute_all_metrics `def compute_all_metrics(self)`
- Defined: `neurothermo.py:232`
- Doc: Compute ALL 17 metrics from history. Call after training.
- Imported by: `app.py`

### summary `def summary(self)`
- Defined: `neurothermo.py:329`
- Doc: Generate summary with ALL metrics.
- Imported by: `app.py`

### step_count `def step_count(self)`
- Defined: `neurothermo.py:381`
- Imported by: `app.py`

### last_result `def last_result(self)`
- Defined: `neurothermo.py:385`
- Imported by: `app.py`

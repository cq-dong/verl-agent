# PowerOPD Integration in TGSA-GRPO

This document describes the PowerOPD (arXiv:2606.17199) integration into the TGSA-GRPO framework.

## Overview

PowerOPD addresses the **unbounded log-ratio problem** in OPD/TGSA:
- Standard log-ratio: `log(π_T/π_θ)` can be ∈ (-∞, +∞)
- When π → 0, log-ratio → -∞, causing gradient instability
- PowerOPD: `π_T^α - π_θ^α` ∈ [-1, 1] naturally bounded

## Key Changes

### 1. Core Module: `gigpo/poweropd.py`

Provides bounded power-based reward functions:

```python
from gigpo.poweropd import compute_power_diff, compute_power_margin

# Case 2 singleton signal (replaces log-ratio + tanh)
diff = compute_power_diff(log_prob_T, log_prob_S, alpha=5.0)  # ∈ [-1, 1]

# Case 1 margin mode (replaces log-margin)
margin = compute_power_margin(log_prob_T, log_prob_runnerup, alpha=5.0)
```

### 2. TGSA Extension

Enable PowerOPD in TGSA via config:

```yaml
algorithm:
  tgsa:
    use_poweropd: true
    power_alpha: 5.0    # Higher = more selective to high-prob tokens
```

**Effects:**
- Case 2 (singleton): `tanh(gamma * z_score(log π_T - log π_θ))` → `gamma * z_score(π_T^α - π_θ^α)`
- Case 1 (margin, optional): `log π_T - log π_runnerup` → `π_T^α - π_runnerup^α`

### 3. Pure OPD Extension

Enable PowerOPD in pure distillation:

```yaml
algorithm:
  adv_estimator: opd
  opd_config:
    use_poweropd: true
    power_alpha: 5.0
```

**Effect:** Standard KL `E[log π_θ - log π_T]` → bounded power diff `π_T^α - π_θ^α`

## Hyperparameter: `power_alpha`

The power coefficient α controls selectivity:

| α | Behavior | Use Case |
|---|----------|----------|
| 1.0 | Linear, considers all tokens equally | Baseline |
| 5.0 | (Recommended) Emphasizes high-probability tokens | Default |
| 10.0 | Highly selective, only top tokens matter | Sparse rewards |

PowerOPD paper finds larger α generally more stable (fewer low-support tokens contributing noise).

## Comparison Experiments

Four configurations for ablation study:

| Config | File | Purpose |
|--------|------|---------|
| TGSA+PowerOPD | `qwen2.5-7b-poweropd.yaml` | Full method with bounded rewards |
| TGSA+Vanilla | `qwen2.5-7b-tgsa-vanilla.yaml` | Baseline with log-ratio |
| Pure PowerOPD | `qwen2.5-7b-pure-poweropd.yaml` | Pure distillation with bounded rewards |
| Pure OPD | `qwen2.5-7b-pure-opd.yaml` | Standard pure distillation |

## Expected Benefits

1. **Gradient Stability**: No extreme values when π → 0
2. **No Tanh Saturation**: Power diff naturally bounded, no compression needed
3. **Training Consistency**: Batch composition has less impact on signal scale
4. **Selective Focus**: Large α filters low-probability noise

## Monitoring

Check the following stats in tensorboard:

```
tgsa_signal/poweropd_enabled    # Should be 1.0 when active
tgsa_signal/power_alpha         # Your configured alpha
opd/poweropd_enabled            # For pure OPD mode
```

## References

- PowerOPD: arXiv:2606.17199
- TGSA-GRPO: See idea.md and TGSA_IMPLEMENTATION.md

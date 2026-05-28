# Modeling

This directory contains model development code for fitting dynamics models to Cleo simulation outputs. It is **not** part of the published `cleo` package.

## Structure

```
modeling/
├── __init__.py
├── data.py          # Cleo sim → training data conversion (Neo → arrays/H5)
├── models/          # Model definitions (LDS, RNN, LFADS, etc.)
│   └── __init__.py
└── scripts/         # Training scripts, analysis, experiments
```

## Quick Start

```python
import cleo
from modeling.data import neo_to_trials, save_trials_h5

# Run your Cleo simulation
sim = cleo.CLSimulator(net)
# ... inject devices, run sim ...
sim.run(duration)

# Export and convert to trial-structured data
block = sim.to_neo()
trials = neo_to_trials(
    block,
    trial_duration_ms=1000,
    bin_size_ms=10,
    signal_names=["fiber"],  # stimulator signals
)

# Save for model training
save_trials_h5("training_data.h5", trials)
```

## Data Flow

```
Cleo Simulation
    │
    ▼ sim.to_neo()
Neo Block (spikes, analog signals)
    │
    ▼ modeling.data.neo_to_trials()
Trial-structured dict {rates, signals}
    │
    ├──▶ save_trials_h5() → HDF5 for offline training
    │
    └──▶ Direct use in model.fit()
         │
         ▼ Trained model
         │
         └──▶ Plug into cleo.IOProcessor for closed-loop control
```

## Dependencies

Core Cleo deps plus:
- `scipy` (smoothing, resampling)
- `h5py` (data persistence)
- Model-specific: `torch`, `jax`, etc. as needed

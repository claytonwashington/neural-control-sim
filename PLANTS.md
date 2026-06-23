# Plants Registry

This document catalogs all Cleo neural simulation plant configurations used in this project.

---

## Plant 1: Unidirectional Excitatory (v0)

**Function**: `build_plant()`  
**File**: [plant.py](file:///snel/home/cbwash2/cleo-worktrees/bidir-v2-plant/modeling/plant.py)  
**Status**: ✅ Active (original baseline)

| Component | Details |
|-----------|---------|
| **Excitatory Opsin** | ChrimsonR (cation channel, E=0 mV, peak ~590 nm) |
| **Inhibitory Opsin** | GtACR2 (anion channel, E=−69.5 mV, peak ~470 nm) |
| **Light Sources** | `fiber_red` (ChrimsonR), `fiber_blue` (GtACR2) |
| **Recording** | 50-channel `MultiUnitActivity` (MUA) — unsorted spike counts per channel per timestep |
| **Neurons** | 1000 LIF (800 E + 200 I), randomly connected |
| **Data File** | `data/training_trials.h5` |

> [!WARNING]
> **GtACR2 has broken inhibition biophysics.** Its reversal potential (E=−69.5 mV) is nearly 
> identical to the resting potential (E_L=−70 mV), producing only ~0.5 mV of hyperpolarizing 
> drive. This makes inhibition paradoxical or negligible. Results from this plant's inhibitory 
> channel should be interpreted with extreme caution.

### Experiments on Plant 1
| Exp | Name | Best R² | Notes |
|-----|------|---------|-------|
| 1-27 | Various (see ideas/modeling.md) | 0.939 (acausal), 0.864 (distilled) | Full sweep history |

---

## ~~Plant 2: Bidirectional v1 (ChrimsonR + GtACR2)~~

**Status**: ❌ **Retired** — GtACR2 biophysics make this plant unusable for bidirectional control.

This is effectively the same as Plant 1. The GtACR2 "inhibition" channel does not produce 
meaningful rate suppression. Disregarded for all future work.

---

## Plant 3: Bidirectional v2 (ChR2-H134R + eNpHR3.0)

**Function**: `build_plant_v2()`  
**File**: [plant.py](file:///snel/home/cbwash2/cleo-worktrees/bidir-v2-plant/modeling/plant.py)  
**Status**: ✅ Active (current primary plant)

| Component | Details |
|-----------|---------|
| **Excitatory Opsin** | ChR2-H134R (cation channel, E=0 mV, peak ~450 nm) |
| **Inhibitory Opsin** | eNpHR3.0 (chloride pump, E=−400 mV, peak ~590 nm) |
| **Light Sources** | `fiber_exc` (ChR2-H134R, blue), `fiber_inh` (eNpHR3.0, yellow) |
| **Recording** | 50-channel `MultiUnitActivity` (MUA) — unsorted spike counts per channel per timestep |
| **Neurons** | 1000 LIF (800 E + 200 I), randomly connected |
| **Data File** | `data/training_trials_bidir_v2.h5` |

> [!TIP]
> eNpHR3.0 is a chloride **pump** (not a channel), with E=−400 mV — providing ~330 mV of 
> hyperpolarizing drive vs GtACR2's ~0.5 mV. This enables genuine, monotonic inhibition 
> (287 → 6 Hz, 98% suppression).

### Experiments on Plant 3
| Exp | Name | Best R² | Notes |
|-----|------|---------|-------|
| 28 | Acausal NODE | 0.956 | z64, h256, lr=5e-4 |
| 29 | Aligned Distillation | 0.917 | α=0.9, pw=200, lr=1e-3 |
| 30 | Periodic Re-Encoding | 0.935 | K=1 best, K=20 = 0.930 |
| 31 | EnKF Eval | 0.9998 | K=1, Q=0.1, R=0.01 |
| 32 | Optoclamp MPC vs PI | ❌ | PI beats MPC; tuning sweep in progress |

---

## Recording Modalities

### Current: MultiUnitActivity (MUA)
- **What it records**: Unsorted spike counts per electrode channel per timestep
- **Output shape**: `(n_timesteps, n_channels)` — integer counts
- **Processing**: Converted to instantaneous firing rates via exponential smoothing
- **Pros**: Dense, smooth, easy to model with continuous dynamics
- **Cons**: Not realistic for real experiments; loses single-neuron identity

### Planned: SortedSpiking
- **What it records**: Individual spike times with neuron identity (spike sorting)
- **Output shape**: Variable-length spike trains `(neuron_id, spike_time)`
- **Processing**: Must be binned, smoothed, or embedded to form model inputs
- **Pros**: Realistic; preserves single-neuron information; enables electrode-invariant representations
- **Cons**: Sparse; variable-length; harder to model; sensitive to electrode placement

### Available but unused: LFP (TKLFPSignal / RWSLFPSignal)
- **What it records**: Local field potential (continuous voltage signal)
- **Not currently planned** but could complement spiking data

---

## Electrode Placement & Representational Drift

A key challenge for real-world deployment: the same neural population will look different 
through different electrode placements. Models trained on one electrode configuration may 
not generalize.

**Goal**: Learn representations that are *invariant* to electrode placement, so that:
1. A model trained on Plant 1 can transfer to Plant 3 (same neurons, different recording)
2. A model trained on one probe position can work after electrode drift
3. The latent space captures neural dynamics, not recording artifacts

This connects to the "representational drift" problem in neuroscience: 
even stable neural computations appear to change over time when viewed through 
a fixed electrode array.

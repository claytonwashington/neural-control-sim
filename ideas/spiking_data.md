# Spiking Data — Progress Tracker

> Tracks progress on building models that operate on spike-sorted neural data
> instead of MUA (multi-unit activity) aggregate rates.

## Motivation

The current modeling pipeline uses 50-channel MUA rates — a convenient abstraction
where spike sorting is "free" because we control the simulation. In real experiments,
the observation is spike-sorted single-unit activity from variable electrode configurations.
Building models that work on spike-sorted data is a prerequisite for deploying the 
control pipeline on real neural data.

## Generated Datasets

| File | Plant | Trials | Placements | Neurons (padded) | Time bins | Size | Status |
|------|-------|--------|------------|-------------------|-----------|------|--------|
| `data/spiking_plant3.h5` | Plant 3 (ChR2+eNpHR3.0) | 50 | 6 | ~119 | 3000 | 337 MB | ✅ Generated |
| `data/spiking_plant1.h5` | Plant 1 (ChrimsonR+GtACR2) | 50 | 6 | ~130 | 3000 | 344 MB | ✅ Generated |

### HDF5 Structure (per placement group)
```
placement_0/
  x_sorted:          (50, max_n_sorted, 3000)  — binned spike counts per sorted neuron
  x_mua:             (50, 50, 3000)            — MUA rates (50ch, 10ms bins)
  n_sorted_per_trial: (50,)                    — actual neuron count per trial (for masking padding)
  probe_coords:      (50, 3)                   — electrode xyz positions
  u:                 (50, 2, 30000)            — control inputs at 1ms resolution
```

### Key Data Properties
- `n_sorted` varies per trial (stochastic from spike sorting) — arrays zero-padded to max per placement
- Each placement has different probe/fiber positions in the tissue volume
- Plant 3 has 6 placements with n_sorted ranging from 82-119 neurons
- Plant 1 has 6 placements with n_sorted ranging from 85-130 neurons
- Time resolution: 10ms bins for spiking data, 1ms for control inputs

---

## Roadmap

### Phase 1: Baseline Spiking CA-NODE (single placement)
**Status**: Not started
**Experiment**: TBD (next in ideas/modeling.md)

Train a standard Latent CA-NODE on spiking data from a single probe placement,
proving we can learn dynamics from sorted spike trains instead of MUA.

**Key differences from MUA pipeline:**
- Input observation dim: n_x goes from 50 (MUA channels) → ~119 (sorted neurons)
- Must mask zero-padded neurons using `n_sorted_per_trial`
- Decoder output matches n_sorted (not fixed 50)
- Training data: 10ms bins → may need to adjust dt and horizon

**Questions to resolve before training:**
1. Should we bin at 10ms (current) or finer resolution?
2. Should training use Euler (to match MPC) or dopri5 (current standard)?
3. Which placement to start with? Placement 0 is consistent (centered probe)
4. What train/test split? Use standard 40/10 (same as MUA work)?

### Phase 2: Cross-placement generalization evaluation
**Status**: Blocked by Phase 1

Train on placement 0, evaluate on placements 1-5. Quantify how much R² degrades
when the electrode configuration changes. This establishes the "electrode invariance gap"
that Phase 3 aims to close.

### Phase 3: Electrode-invariant encoder
**Status**: Blocked by Phase 2

Replace the fixed-dimension GRU encoder with a permutation-invariant architecture
(Set Transformer, DeepSets, or point cloud encoder) that maps {spike_train_i}_{i=1..N}
to z₀ regardless of N or neuron identity.

**Architecture options:**
- **DeepSets**: φ(spike_train_i) → pool → ρ(pooled) → z₀. Simple, fast.
- **Set Transformer**: Multi-head self-attention over spike trains. More expressive but slower.
- **Point cloud + spikes**: Include electrode xyz as per-neuron features.

**Training strategy:**
- Train on all 6 placements simultaneously (each placement is a "view" of the same system)
- Augment by randomly subsampling neurons during training
- Evaluate on held-out placements or held-out neuron subsets

---

## Notes & Decisions Log

| Date | Decision | Context |
|------|----------|---------|
| 2026-06-13 | Zero-pad x_sorted per placement group | Fix for stochastic n_sorted shape mismatch |
| 2026-06-14 | Train baseline model before invariant encoder | Need working spiking pipeline first |
| 2026-06-17 | Document created | Tracking spiking data progress |

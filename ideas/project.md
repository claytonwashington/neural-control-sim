# Cleo Project Ideas

## 1. Organoid Closed-Loop Control via Subspace Targeting
**Concept**: Simulate a cortical organoid — a random ball of excitatory and inhibitory neurons with no structured connectivity — using Brian2 inside Cleo. The organoid has no intrinsic "task"; it just produces spontaneous, chaotic bursts and avalanches. The goal is to take this blank biological slate and *impose synthetic structure* onto it by forcing its high-dimensional neural state to trace a designer trajectory (e.g., a circle or figure-eight) in a low-dimensional subspace.
**Recording**: Simulate a ~100-channel HD-MEA grid underneath the organoid to read out population firing rates.
**Optogenetics**: Express opsins in the organoid (e.g., ChR2 in excitatory neurons, potentially an inhibitory opsin in inhibitory neurons) and illuminate with multi-site light sources.
**Pipeline**:
1. Run a baseline (lasers-off) simulation and record smoothed firing rates across all MEA channels.
2. Run PCA on the baseline data to find the network's natural "path of least resistance" (PC1, PC2 define a 2D subspace).
3. Define a synthetic target trajectory (limit cycle, figure-eight, etc.) *within* that PCA subspace.
4. Use subspace system identification (N4SID) to fit a linear state-space model mapping optical inputs → neural state.
5. Design a controller (e.g., LQR/LQG or MPC) using the identified linear model to drive the organoid's state along the target trajectory in real time.

## 2. Closed-Loop PI Control of Firing Rates
**Concept**: Build a network with excitatory and inhibitory populations and use closed-loop feedback (PI controller) to maintain a target firing rate in the excitatory population.
**Optogenetics**: Use an excitatory opsin (ChR2) and an inhibitory opsin (eNpHR3.0).
**Intervention**: The controller dynamically adjusts the irradiance of a light source based on real-time population firing rates read via an electrode or imaging module.

## 3. Spatial Optogenetic Patterning
**Concept**: Simulate a 2D sheet of neurons and explore the effects of spatially restricted illumination (e.g., Koehler illumination or a 2P scanning beam).
**Optogenetics**: Express a generic 4-state Markov opsin.
**Intervention**: Project complex light patterns or moving spots and analyze the resulting propagating waves of activity across the cortical sheet.

## 4. Opsin Crosstalk Analysis
**Concept**: Investigate the unintended activation (crosstalk) when using multiple opsins with overlapping action spectra in the same network.
**Optogenetics**: Co-express two different opsins (e.g., ChR2 and Chrimson) in distinct subnetworks.
**Intervention**: Stimulate with varying wavelengths and irradiance levels to map out the effective independent control bandwidths before crosstalk disrupts the targeted activity.

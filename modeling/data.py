"""Data utilities for converting Cleo simulation outputs to model-ready formats.

Cleo exports data as Neo Block objects. This module provides functions to convert
those into numpy arrays and trial-structured datasets suitable for model fitting.

Data format convention:
    Rates:   (n_trials, n_channels, n_bins_per_trial)  — contiguous 3D array
    Inputs:  (n_trials, n_inputs, n_bins_per_trial)    — contiguous 3D array
    Times:   (n_bins_per_trial,)                       — shared time axis
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import neo
except ImportError:
    raise ImportError("neo is required: pip install neo")

try:
    import h5py
except ImportError:
    h5py = None


def neo_spikes_to_rates(
    block: neo.Block,
    bin_size_ms: float = 10.0,
    smooth_sigma_ms: float | None = 50.0,
) -> tuple[NDArray, NDArray]:
    """Convert spike trains from a Neo Block to a binned firing rate matrix.

    Parameters
    ----------
    block : neo.Block
        Neo Block exported from CLSimulator.to_neo()
    bin_size_ms : float
        Bin width in milliseconds for spike counting
    smooth_sigma_ms : float or None
        Gaussian smoothing sigma in ms. None for no smoothing (raw counts).

    Returns
    -------
    rates : ndarray, shape (n_neurons, n_bins)
        Firing rates in Hz (or spike counts if smooth_sigma_ms is None)
    t_bins : ndarray, shape (n_bins,)
        Bin center times in seconds
    """
    from scipy.ndimage import gaussian_filter1d

    seg = block.segments[0]
    spike_trains = seg.spiketrains
    if not spike_trains:
        raise ValueError("No spike trains found in Neo Block")

    t_start = min(st.t_start.rescale("s").magnitude for st in spike_trains)
    t_stop = max(st.t_stop.rescale("s").magnitude for st in spike_trains)
    bin_size_s = bin_size_ms / 1000.0
    bin_edges = np.arange(t_start, t_stop + bin_size_s, bin_size_s)
    n_bins = len(bin_edges) - 1
    t_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    n_neurons = len(spike_trains)
    counts = np.zeros((n_neurons, n_bins))
    for i, st in enumerate(spike_trains):
        times = st.rescale("s").magnitude
        counts[i], _ = np.histogram(times, bins=bin_edges)

    if smooth_sigma_ms is not None:
        sigma_bins = smooth_sigma_ms / bin_size_ms
        rates = gaussian_filter1d(counts, sigma=sigma_bins, axis=1) / bin_size_s
    else:
        rates = counts / bin_size_s

    return rates, t_bins


def chop_to_trials(
    x: NDArray,
    trial_bins: int,
    overlap_bins: int = 0,
    u: NDArray | None = None,
) -> dict:
    """Chop continuous (n_channels, n_total_bins) data into trial-stacked arrays.

    Parameters
    ----------
    x : ndarray, shape (n_channels, n_total_bins)
        Continuous state trajectory (e.g., firing rates)
    trial_bins : int
        Number of bins per trial
    overlap_bins : int
        Overlap between adjacent trials
    u : ndarray or None, shape (n_inputs, n_total_bins)
        Optional input trajectory to chop in parallel

    Returns
    -------
    dict with keys:
        'x' : ndarray, shape (n_trials, n_channels, trial_bins)
        'u' : ndarray, shape (n_trials, n_inputs, trial_bins) — if u provided
        'n_trials' : int
        'trial_bins' : int
    """
    step = trial_bins - overlap_bins
    n_total = x.shape[1]
    starts = list(range(0, n_total - trial_bins + 1, step))

    x_trials = np.stack([x[:, s : s + trial_bins] for s in starts])  # (n_trials, C, T)

    result = {
        "x": x_trials,
        "n_trials": len(starts),
        "trial_bins": trial_bins,
    }

    if u is not None:
        result["u"] = np.stack([u[:, s : s + trial_bins] for s in starts])

    return result


def save_continuous_h5(filepath: str, x: NDArray, u: NDArray, t: NDArray, **attrs) -> None:
    """Save continuous paired trajectories to HDF5.

    Parameters
    ----------
    filepath : str
        Output path
    x : ndarray, shape (n_channels, n_steps)
        State trajectory (smoothed firing rates)
    u : ndarray, shape (n_inputs, n_steps)
        Input trajectory (stimulator values)
    t : ndarray, shape (n_steps,)
        Time points in seconds
    **attrs : dict
        Additional metadata (dt, tau_smooth, etc.)
    """
    if h5py is None:
        raise ImportError("h5py is required: pip install h5py")

    with h5py.File(filepath, "w") as f:
        f.create_dataset("x", data=x, compression="gzip")
        f.create_dataset("u", data=u, compression="gzip")
        f.create_dataset("t", data=t)
        for k, v in attrs.items():
            f.attrs[k] = v


def save_trials_h5(filepath: str, trial_data: dict, **attrs) -> None:
    """Save trial-structured data as contiguous 3D arrays to HDF5.

    Parameters
    ----------
    filepath : str
        Output path
    trial_data : dict
        Output from chop_to_trials(). Must contain 'x' key.
    **attrs : dict
        Additional metadata
    """
    if h5py is None:
        raise ImportError("h5py is required: pip install h5py")

    with h5py.File(filepath, "w") as f:
        # Single contiguous arrays — fast loading, easy batching
        f.create_dataset("x", data=trial_data["x"], compression="gzip")
        if "u" in trial_data:
            f.create_dataset("u", data=trial_data["u"], compression="gzip")
        f.attrs["n_trials"] = trial_data["n_trials"]
        f.attrs["trial_bins"] = trial_data["trial_bins"]
        for k, v in attrs.items():
            f.attrs[k] = v


def load_trials_h5(filepath: str) -> dict:
    """Load trial-structured data from HDF5.

    Returns dict with 'x' as (n_trials, n_channels, trial_bins) array,
    plus 'u' if present and all attrs.
    """
    if h5py is None:
        raise ImportError("h5py is required: pip install h5py")

    with h5py.File(filepath, "r") as f:
        result = {"x": f["x"][:]}  # full load into RAM
        if "u" in f:
            result["u"] = f["u"][:]
        for k, v in f.attrs.items():
            result[k] = v
    return result


def load_continuous_h5(filepath: str) -> dict:
    """Load continuous trajectory data from HDF5.

    Returns dict with 'x', 'u', 't' arrays and all attrs.
    """
    if h5py is None:
        raise ImportError("h5py is required: pip install h5py")

    with h5py.File(filepath, "r") as f:
        result = {"x": f["x"][:], "u": f["u"][:], "t": f["t"][:]}
        for k, v in f.attrs.items():
            result[k] = v
    return result

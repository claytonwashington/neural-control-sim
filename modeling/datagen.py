"""Data generation via OU-noise-driven Cleo simulation.

Drives optic fibers with Ornstein-Uhlenbeck processes and records
smoothed multi-unit firing rates for system identification.

Supports both:
- Original plant (1 combined light device "fibers" with 2 fiber positions)
- Bidirectional plant (2 separate light devices "fiber_red" and "fiber_blue")

The input u shape is:
- Original: (2, n_steps)  — 2 fibers in one Light device
- Bidirectional: (2, n_steps) — u[0]=red (excitatory), u[1]=blue (inhibitory)

Supports parallel multi-trial generation using multiprocessing.

Usage:
    from modeling.datagen import generate_multi_trial
    trials = generate_multi_trial(n_trials=10, trial_duration_s=30, n_workers=10)
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
from numpy.typing import NDArray

import brian2.only as b2
from brian2 import ms, mwatt, mm

import cleo
from cleo.ioproc import LatencyIOProcessor, exp_firing_rate_estimate


def ou_process(
    n_steps: int,
    n_channels: int,
    dt: float,
    tau: float = 0.05,
    sigma: float = 2.0,
    mu: float = 5.0,
    seed: int | None = None,
) -> NDArray:
    """Generate Ornstein-Uhlenbeck noise for driving optical inputs.

    dX = (mu - X) / tau * dt + sigma * sqrt(2 * dt / tau) * dW

    Output is clipped to [0, inf) since irradiance cannot be negative.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    n_channels : int
        Number of independent OU channels (one per fiber)
    dt : float
        Time step in seconds
    tau : float
        Mean reversion time constant in seconds (default 50ms)
    sigma : float
        Volatility (controls amplitude of fluctuations in mW/mm^2)
    mu : float
        Mean reversion level in mW/mm^2
    seed : int or None
        Random seed

    Returns
    -------
    u : ndarray, shape (n_channels, n_steps)
        OU input trajectory, clipped to >= 0
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    u = np.zeros((n_channels, n_steps))
    u[:, 0] = mu

    noise_scale = sigma * np.sqrt(2 * dt / tau)

    for t in range(1, n_steps):
        drift = (mu - u[:, t - 1]) / tau * dt
        diffusion = noise_scale * rng.standard_normal(n_channels)
        u[:, t] = u[:, t - 1] + drift + diffusion

    # Irradiance must be non-negative
    np.clip(u, 0, None, out=u)
    return u


class DataCollectionIOP(LatencyIOProcessor):
    """IOProcessor that drives fibers with pre-computed inputs and records rates.

    Applies OU-noise input trajectory to light stimulators and records
    exponentially-smoothed firing rates from MUA channels.

    Supports both single-device (original plant) and multi-device
    (bidirectional plant) configurations.

    Parameters
    ----------
    u_trajectory : ndarray, shape (n_inputs, n_steps)
        Pre-computed input trajectory in mW/mm^2
    sample_period_ms : float
        Sampling period in milliseconds
    tau_smooth_ms : float
        Exponential smoothing time constant in milliseconds
    light_names : list[str]
        Names of the Light stimulator devices.
        For original plant: ["fibers"] (one device with 2 fiber coords).
        For bidirectional plant: ["fiber_red", "fiber_blue"] (two devices, 1 coord each).
    probe_name : str
        Name of the Probe recorder device
    mua_name : str
        Name of the MUA signal within the probe
    """

    def __init__(
        self,
        u_trajectory: NDArray,
        sample_period_ms: float = 1.0,
        tau_smooth_ms: float = 20.0,
        light_names: list[str] | str = "fibers",
        probe_name: str = "probe",
        mua_name: str = "mua",
    ):
        super().__init__(sample_period=sample_period_ms * ms)
        self.u = u_trajectory  # (n_inputs, n_steps)
        self.tau_smooth = tau_smooth_ms * ms
        self.probe_name = probe_name
        self.mua_name = mua_name

        # Normalize light_names to a list
        if isinstance(light_names, str):
            self.light_names = [light_names]
        else:
            self.light_names = list(light_names)

        # Storage
        self.x_history: list[NDArray] = []
        self.u_history: list[NDArray] = []
        self._rates: NDArray | None = None
        self._step: int = 0

    def process(self, state_dict, t_samp):
        """Process one sample: record rates, apply next input."""
        # 1. Get spike counts from MUA
        i_chan, t_spikes, counts = state_dict[self.probe_name][self.mua_name]
        counts = np.asarray(counts, dtype=float)

        # 2. Initialize rates on first call
        if self._rates is None:
            from brian2 import Hz
            self._rates = np.zeros(len(counts)) * Hz

        # 3. Exponential smoothing -> continuous rates
        self._rates = exp_firing_rate_estimate(
            counts,
            self.sample_period,
            self._rates,
            self.tau_smooth,
        )
        self.x_history.append(np.array(self._rates, dtype=np.float64))

        # 4. Look up pre-computed OU input (clamp to available steps)
        step = min(self._step, self.u.shape[1] - 1)
        u_now = self.u[:, step]
        self.u_history.append(u_now.copy())
        self._step += 1

        # 5. Drive light devices
        out = {}
        if len(self.light_names) == 1:
            # Original plant: single device with all fiber coords
            out[self.light_names[0]] = u_now * mwatt / mm**2
        else:
            # Bidirectional plant: one channel per device
            for i, name in enumerate(self.light_names):
                out[name] = u_now[i] * mwatt / mm**2

        return out, t_samp

    def get_results(self) -> dict:
        """Return collected data as numpy arrays.

        Returns
        -------
        dict with keys:
            'x' : ndarray, shape (n_channels, n_steps_collected)
            'u' : ndarray, shape (n_inputs, n_steps_collected)
        """
        return {
            "x": np.column_stack(self.x_history) if self.x_history else np.array([]),
            "u": np.column_stack(self.u_history) if self.u_history else np.array([]),
        }


def _detect_light_config(devices: dict) -> tuple[list[str], int]:
    """Detect whether the plant uses a single or bidirectional light config.

    Returns
    -------
    light_names : list[str]
        Light device names for the IOProcessor.
    n_inputs : int
        Total number of input channels.
    """
    if "light" in devices:
        # Original plant: single Light device with multiple fiber coords
        n_inputs = len(devices["light"].coords)
        return ["fibers"], n_inputs
    elif "light_red" in devices and "light_blue" in devices:
        # Bidirectional plant: two separate Light devices
        n_red = len(devices["light_red"].coords)
        n_blue = len(devices["light_blue"].coords)
        return ["fiber_red", "fiber_blue"], n_red + n_blue
    else:
        raise ValueError(
            "Could not detect light configuration from devices dict. "
            "Expected either 'light' or ('light_red', 'light_blue') keys."
        )


def generate_dataset(
    sim: cleo.CLSimulator,
    devices: dict,
    duration_s: float = 300.0,
    sample_period_ms: float = 1.0,
    tau_smooth_ms: float = 20.0,
    ou_tau: float = 0.05,
    ou_sigma: float = 2.0,
    ou_mu: float = 5.0,
    seed: int = 42,
) -> dict:
    """Run a single Cleo simulation with OU-noise inputs and collect paired data.

    Automatically detects whether the plant uses a single or bidirectional
    light configuration.

    Parameters
    ----------
    sim : cleo.CLSimulator
        Simulator from build_plant()
    devices : dict
        Device references from build_plant()
    duration_s : float
        Simulation duration in seconds
    sample_period_ms : float
        IOProcessor sample period in ms
    tau_smooth_ms : float
        Exponential smoothing time constant in ms
    ou_tau : float
        OU mean reversion time in seconds
    ou_sigma : float
        OU volatility (mW/mm^2)
    ou_mu : float
        OU mean level (mW/mm^2)
    seed : int
        Random seed for OU noise

    Returns
    -------
    dict with keys:
        'x' : ndarray, shape (n_channels, n_steps)
        'u' : ndarray, shape (n_inputs, n_steps)
        't' : ndarray, shape (n_steps,)
        'dt' : float (sample period in seconds)
        'tau_smooth_ms' : float
        'n_channels' : int
        'n_inputs' : int
    """
    dt_s = sample_period_ms / 1000.0
    n_steps = int(duration_s / dt_s)

    light_names, n_inputs = _detect_light_config(devices)

    # Generate OU input trajectory
    print(f"Generating OU inputs: {n_inputs} channels, {n_steps} steps, "
          f"tau={ou_tau}s, sigma={ou_sigma}, mu={ou_mu}")
    print(f"  Light config: {light_names}")
    u = ou_process(n_steps, n_inputs, dt_s, tau=ou_tau, sigma=ou_sigma, mu=ou_mu, seed=seed)

    # Set up IOProcessor
    iop = DataCollectionIOP(
        u_trajectory=u,
        sample_period_ms=sample_period_ms,
        tau_smooth_ms=tau_smooth_ms,
        light_names=light_names,
    )
    sim.set_io_processor(iop)

    # Run simulation
    print(f"Running Cleo simulation for {duration_s}s ...")
    sim.run(duration_s * b2.second)
    print("Simulation complete.")

    # Collect results
    results = iop.get_results()
    n_collected = results["x"].shape[1] if results["x"].ndim == 2 else 0
    t = np.arange(n_collected) * dt_s

    return {
        "x": results["x"],
        "u": results["u"],
        "t": t,
        "dt": dt_s,
        "tau_smooth_ms": tau_smooth_ms,
        "n_channels": results["x"].shape[0] if results["x"].ndim == 2 else 0,
        "n_inputs": n_inputs,
    }


# =============================================================================
# Single-trial worker for multiprocessing
# =============================================================================

def _run_single_trial(args: dict) -> dict:
    """Worker function for parallel trial generation.

    Must be a top-level function (not lambda/closure) for pickling.
    Imports Brian2 fresh in each spawned process via start_scope().

    Parameters
    ----------
    args : dict
        All parameters needed to run one trial.

    Returns
    -------
    dict with 'x' and 'u' arrays for this trial.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import brian2.only as b2
    b2.start_scope()

    from modeling.plant import build_plant

    trial_idx = args["trial_idx"]
    n_trials = args["n_trials"]
    ou_seed = args["ou_seed"]

    print(f"[Worker] Trial {trial_idx + 1}/{n_trials} (PID={mp.current_process().pid}, OU seed={ou_seed})")

    sim, devices = build_plant(
        n_exc=args["n_exc"],
        n_inh=args["n_inh"],
        n_channels=args["n_channels"],
        seed=args["plant_seed"],
    )

    data = generate_dataset(
        sim, devices,
        duration_s=args["trial_duration_s"],
        sample_period_ms=args["sample_period_ms"],
        tau_smooth_ms=args["tau_smooth_ms"],
        ou_tau=args["ou_tau"],
        ou_sigma=args["ou_sigma"],
        ou_mu=args["ou_mu"],
        seed=ou_seed,
    )

    print(f"[Worker] Trial {trial_idx + 1}/{n_trials} done. x={data['x'].shape}, u={data['u'].shape}")
    return {"x": data["x"], "u": data["u"]}


def generate_multi_trial(
    n_trials: int = 10,
    trial_duration_s: float = 30.0,
    sample_period_ms: float = 1.0,
    tau_smooth_ms: float = 20.0,
    ou_tau: float = 0.05,
    ou_sigma: float = 2.0,
    ou_mu: float = 5.0,
    base_seed: int = 42,
    plant_seed: int = 42,
    n_exc: int = 800,
    n_inh: int = 200,
    n_channels: int = 50,
    n_workers: int | None = None,
) -> dict:
    """Generate multi-trial dataset with independent initial conditions and OU noise.

    Each trial rebuilds the plant (same connectivity via plant_seed, but
    different random initial membrane voltages) and uses a different OU noise
    realization. Trials run in parallel using multiprocessing with 'spawn'
    context (required by Brian2).

    Parameters
    ----------
    n_trials : int
        Number of independent trials
    trial_duration_s : float
        Duration of each trial in seconds
    sample_period_ms : float
        IOProcessor sample period in ms
    tau_smooth_ms : float
        Exponential smoothing time constant in ms
    ou_tau, ou_sigma, ou_mu : float
        OU process parameters
    base_seed : int
        Base seed for OU noise (trial i uses base_seed + i)
    plant_seed : int
        Seed for plant connectivity (fixed across trials)
    n_exc, n_inh, n_channels : int
        Plant parameters
    n_workers : int or None
        Number of parallel workers. None = n_trials (one per trial).
        Set to 1 for sequential execution (debugging).

    Returns
    -------
    dict with keys:
        'x' : ndarray, shape (n_trials, n_channels, n_steps_per_trial)
        'u' : ndarray, shape (n_trials, n_inputs, n_steps_per_trial)
        't' : ndarray, shape (n_steps_per_trial,)
        'dt' : float
        'n_trials' : int
        'n_channels' : int
        'n_inputs' : int
        'tau_smooth_ms' : float
        'trial_duration_s' : float
    """
    if n_workers is None:
        n_workers = n_trials

    # Build args for each trial
    trial_args = []
    for i in range(n_trials):
        trial_args.append({
            "trial_idx": i,
            "n_trials": n_trials,
            "ou_seed": base_seed + i,
            "plant_seed": plant_seed,
            "n_exc": n_exc,
            "n_inh": n_inh,
            "n_channels": n_channels,
            "trial_duration_s": trial_duration_s,
            "sample_period_ms": sample_period_ms,
            "tau_smooth_ms": tau_smooth_ms,
            "ou_tau": ou_tau,
            "ou_sigma": ou_sigma,
            "ou_mu": ou_mu,
        })

    print(f"Launching {n_trials} trials with {n_workers} parallel workers")
    print(f"  Each trial: {trial_duration_s}s, {n_exc}E/{n_inh}I, {n_channels}ch")

    if n_workers == 1:
        # Sequential fallback (useful for debugging)
        results = [_run_single_trial(a) for a in trial_args]
    else:
        # Parallel execution with 'spawn' context (required by Brian2)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_run_single_trial, trial_args)

    # Stack into contiguous 3D arrays
    dt_s = sample_period_ms / 1000.0
    min_steps = min(r["x"].shape[1] for r in results)
    x_stacked = np.stack([r["x"][:, :min_steps] for r in results])
    u_stacked = np.stack([r["u"][:, :min_steps] for r in results])
    t = np.arange(min_steps) * dt_s

    print(f"\nMulti-trial generation complete.")
    print(f"  x: {x_stacked.shape}, u: {u_stacked.shape}")
    print(f"  {n_trials} trials x {trial_duration_s}s = {n_trials * trial_duration_s}s total")

    return {
        "x": x_stacked,
        "u": u_stacked,
        "t": t,
        "dt": dt_s,
        "n_trials": n_trials,
        "n_channels": x_stacked.shape[1],
        "n_inputs": u_stacked.shape[1],
        "tau_smooth_ms": tau_smooth_ms,
        "trial_duration_s": trial_duration_s,
    }

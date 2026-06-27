#!/usr/bin/env python3
"""Generate spiking training data from Plants 1 and 3.

Records SortedSpiking alongside MUA for both plants, using the same
seeds as existing MUA-only data so the underlying dynamics are identical.

Multiple random (ε-constrained) probe placements per trial enable
future electrode-invariant representation learning.

Usage:
    python -m modeling.scripts.generate_spiking_data \
        --plant v2 \
        --n-trials 50 \
        --n-placements 6 \
        --output data/spiking_plant3.h5

    python -m modeling.scripts.generate_spiking_data \
        --plant v1 \
        --n-trials 50 \
        --n-placements 6 \
        --output data/spiking_plant1.h5
"""

import argparse
import h5py
import multiprocessing as mp
import numpy as np
import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _make_probe_coords_with_offset(
    n_channels: int,
    volume_um: float,
    offset_xz: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Generate probe coordinates with optional XZ offset.

    The grid spans 80% of the volume, same as the default in plant.py.
    """
    half = volume_um / 2
    n_side = int(np.ceil(np.sqrt(n_channels)))
    xs = np.linspace(-half * 0.8, half * 0.8, n_side) + offset_xz[0]
    zs = np.linspace(-half * 0.8, half * 0.8, n_side) + offset_xz[1]
    xx, zz = np.meshgrid(xs, zs)
    coords = np.column_stack([
        xx.ravel()[:n_channels],
        np.zeros(n_channels),
        zz.ravel()[:n_channels],
    ])
    return coords


def _generate_probe_offsets(
    n_placements: int,
    volume_um: float,
    n_channels: int,
    rng: np.random.RandomState,
) -> list[tuple[float, float]]:
    """Generate ε-constrained random probe offsets.

    Each offset guarantees at least one electrode falls within ε of
    the neuron population center of mass (≈ origin). ε = grid_spacing / 2.
    """
    n_side = int(np.ceil(np.sqrt(n_channels)))
    grid_span = volume_um * 0.8  # 80% of half-width on each side
    grid_spacing = grid_span / max(n_side - 1, 1)
    epsilon = grid_spacing / 2  # ~28 µm for 50ch in 500µm

    offsets = [(0.0, 0.0)]  # first placement is always centered
    for _ in range(n_placements - 1):
        dx = rng.uniform(-epsilon, epsilon)
        dz = rng.uniform(-epsilon, epsilon)
        offsets.append((dx, dz))

    return offsets


def _run_trial_placement(args: dict) -> dict:
    """Worker: run one (trial, placement) combination.

    Builds the plant fresh (same connectivity via plant_seed), adds a
    SortedSpiking probe at the specified offset, runs the sim with OU
    stimulation, then bins spikes at the requested resolution.

    Must be top-level for pickling. Uses Brian2 start_scope().
    """
    import warnings
    warnings.filterwarnings("ignore")

    import os
    import brian2.only as b2
    b2.prefs.codegen.runtime.cython.cache_dir = f'/tmp/brian2_cache_{os.getpid()}'
    b2.start_scope()

    trial_idx = args["trial_idx"]
    placement_idx = args["placement_idx"]
    n_trials = args["n_trials"]
    ou_seed = args["ou_seed"]
    offset_xz = args["offset_xz"]
    plant_type = args["plant_type"]
    bin_ms = args["bin_ms"]

    # Seed Brian2's runtime RNG so the spiking dynamics are reproducible.
    # (build_fn's np.random.seed pins connectivity/positions; the OU input is
    # seeded separately. Without this, Brian2's stochastic elements made every
    # run differ.) sim_seed is deterministic per (trial, placement).
    b2.seed(args["sim_seed"])

    pid = mp.current_process().pid
    print(f"[PID={pid}] Trial {trial_idx+1}/{n_trials}, "
          f"placement {placement_idx}, offset=({offset_xz[0]:.1f}, {offset_xz[1]:.1f})µm")

    # Build the plant (same seed → same connectivity and neuron positions)
    if plant_type == "v3":
        from modeling.plant import build_plant_v3 as build_fn
        _extra = dict(
            runtime_seed=args["sim_seed"],
            I_bg_pA=args["I_bg_pA"], a_nS=args["a_nS"],
            b_pA=args["b_pA"], tau_w_ms=args["tau_w_ms"],
        )
    elif plant_type == "v2":
        from modeling.plant import build_plant_v2 as build_fn
        _extra = {}
    else:
        from modeling.plant import build_plant as build_fn
        _extra = {}

    sim, devices = build_fn(
        n_exc=args["n_exc"],
        n_inh=args["n_inh"],
        n_channels=args["n_channels"],
        seed=args["plant_seed"],
        **_extra,
    )
    ng = devices["ng"]

    # Add SortedSpiking probe at the specified offset
    from cleo.ephys import SortedSpiking, Probe

    sorted_sig = SortedSpiking(name="sorted", snr_cutoff=args["snr_cutoff"])
    probe_coords = _make_probe_coords_with_offset(
        args["n_channels"], args["volume_um"], offset_xz
    )
    sorted_probe = Probe(
        name="sorted_probe",
        coords=probe_coords * b2.um,
        signals=[sorted_sig],
        save_history=True,
    )
    sim.inject(sorted_probe, ng)
    n_sorted = sorted_sig.n_sorted

    # ------------------------------------------------------------------
    # LFP recording (purely additive — does NOT touch x_sorted/x_mua).
    #
    # Both LFP proxies are ALWAYS generated and archived (project policy;
    # experiments use RWSLFP). They are placed on a SEPARATE probe at the
    # *same* coordinates as the SortedSpiking probe so the existing
    # whole-group SortedSpiking injection is left byte-for-byte unchanged;
    # the LFP signals only add passive SpikeMonitors, which do not perturb
    # the Brian2 dynamics that produce x_mua.
    #
    # Modeling assumptions:
    #   * Pyramidal cells == excitatory population (ng[:n_exc]). LFP is
    #     dominated by synaptic currents onto pyramidal cells, so RWSLFP
    #     currents target only synapses landing on j < n_exc, and the
    #     TKLFP/RWSLFP reference population is the excitatory subgroup.
    #   * TKLFP: excitatory spikes use the 'exc' kernel and inhibitory
    #     spikes the 'inh' kernel; both subgroups contribute (Telenczuk
    #     et al. kernel LFP).
    #   * RWSLFP-from-spikes: AMPA = recurrent excitatory synapses
    #     (exc_syn), GABA = recurrent inhibitory synapses (inh_syn), each
    #     restricted to synapses onto pyramidal cells (j < n_exc). Synaptic
    #     weights are read from the plant namespace (w_exc, w_inh) and
    #     passed as positive magnitudes (the AMPA/GABA sign convention is
    #     handled internally by wslfp). Because the plant uses instantaneous
    #     delta-synapses (I_syn += w) with no explicit current dynamics, we
    #     use the documented spike-based path
    #     (wslfp.spikes_to_biexp_currents) with the library-default
    #     biexponential kernel (tau1/tau2 AMPA = 2/0.4 ms, GABA = 5/0.25 ms,
    #     syn_delay = 1 ms).
    #   * LFP is sampled at the IOProcessor period (== sample_period_ms,
    #     1 ms) just like the raw MUA, then bin-averaged into bin_ms bins
    #     exactly as x_mua, yielding (n_channels, T_binned) per trial.
    # ------------------------------------------------------------------
    from cleo.ephys import TKLFPSignal, RWSLFPSignalFromSpikes

    n_exc = args["n_exc"]
    exc_syn = devices["exc_syn"]
    inh_syn = devices["inh_syn"]
    # Synaptic weight magnitudes (amps) from the plant namespace.
    w_exc_A = float(exc_syn.namespace["w_exc"] / b2.amp)
    w_inh_A = float(inh_syn.namespace["w_inh"] / b2.amp)

    tklfp_sig = TKLFPSignal(name="tklfp")
    rwslfp_sig = RWSLFPSignalFromSpikes(name="rwslfp")
    lfp_probe = Probe(
        name="lfp_probe",
        coords=probe_coords * b2.um,   # co-located with the sorted probe
        signals=[tklfp_sig, rwslfp_sig],
        save_history=True,
    )

    sample_period = args["sample_period_ms"] * b2.ms
    # Excitatory subgroup: 'exc' TKLFP kernel + AMPA/GABA currents onto
    # pyramidal cells (j < n_exc) for RWSLFP.
    sim.inject(
        lfp_probe, ng[:n_exc],
        tklfp_type="exc",
        sample_period=sample_period,
        ampa_syns=[(exc_syn[f"j < {n_exc}"], {"weight": w_exc_A})],
        gaba_syns=[(inh_syn[f"j < {n_exc}"], {"weight": w_inh_A})],
    )
    # Inhibitory subgroup contributes only inhibitory spikes to TKLFP.
    sim.inject(
        lfp_probe, ng[n_exc:],
        tklfp_type="inh",
        sample_period=sample_period,
    )

    print(f"[PID={pid}] Trial {trial_idx+1} pl={placement_idx}: "
          f"{n_sorted} sorted neurons")

    # Run simulation with OU stimulation (same datagen pipeline)
    from modeling.datagen import generate_dataset

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

    x_mua = data["x"]  # (n_channels, T_fine) — smoothed MUA rates
    u = data["u"]       # (n_inputs, T_fine)
    T_fine = x_mua.shape[1]

    # Extract sorted spikes from signal's t and i arrays
    t_spikes = np.array(sorted_sig.t)   # spike times in seconds
    i_spikes = np.array(sorted_sig.i)   # sorted neuron indices

    # Bin at requested resolution
    bin_s = bin_ms / 1000.0
    T_binned = int(args["trial_duration_s"] / bin_s)
    x_sorted = np.zeros((n_sorted, T_binned), dtype=np.int16)

    for spike_t, spike_i in zip(t_spikes, i_spikes):
        bi = min(int(spike_t / bin_s), T_binned - 1)
        si = int(spike_i)
        if 0 <= si < n_sorted:
            x_sorted[si, bi] += 1

    # Also downsample MUA to match bin resolution
    bin_factor = int(bin_ms / args["sample_period_ms"])
    T_use = T_binned * bin_factor
    if T_use <= T_fine:
        x_mua_binned = x_mua[:, :T_use].reshape(
            x_mua.shape[0], T_binned, bin_factor
        ).mean(axis=2)
    else:
        x_mua_binned = x_mua

    # ------------------------------------------------------------------
    # Bin LFP exactly like x_mua: signal.lfp is (T_fine, n_channels) at
    # the 1 ms IOProcessor period -> transpose to (n_channels, T_fine),
    # then average within each bin_ms window -> (n_channels, T_binned).
    # ------------------------------------------------------------------
    def _bin_lfp(lfp_TxC):
        lfp_CxT = np.asarray(lfp_TxC, dtype=np.float64).T  # (n_channels, T_fine)
        if T_use <= lfp_CxT.shape[1]:
            return lfp_CxT[:, :T_use].reshape(
                lfp_CxT.shape[0], T_binned, bin_factor
            ).mean(axis=2)
        return lfp_CxT

    # TKLFP is stored with Brian units (uvolt); strip to plain microvolts.
    x_lfp_tklfp = _bin_lfp(tklfp_sig.lfp / b2.uvolt)
    # RWSLFP is dimensionless (arbitrary units, normalized downstream).
    x_lfp_rwslfp = _bin_lfp(np.asarray(rwslfp_sig.lfp))

    n_spikes = int(x_sorted.sum())
    n_active = int((x_sorted.sum(axis=1) > 0).sum())
    print(f"[PID={pid}] Trial {trial_idx+1} pl={placement_idx} done: "
          f"{n_spikes} spikes, {n_active}/{n_sorted} active neurons")

    return {
        "trial_idx": trial_idx,
        "placement_idx": placement_idx,
        "x_sorted": x_sorted,          # (n_sorted, T_binned) int16
        "x_mua": x_mua_binned,         # (n_channels, T_binned) float
        "x_lfp_tklfp": x_lfp_tklfp,    # (n_channels, T_binned) float (uV)
        "x_lfp_rwslfp": x_lfp_rwslfp,  # (n_channels, T_binned) float (a.u.)
        "u": u,                         # (n_inputs, T_fine) float
        "n_sorted": n_sorted,
        "probe_coords": probe_coords,   # (n_channels, 3) float
        "offset_xz": offset_xz,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate spiking training data with multiple probe placements"
    )
    parser.add_argument("--plant", choices=["v1", "v2", "v3"], required=True,
                        help="Plant version: v1 (ChrimsonR+GtACR2) or v2 (H134R+eNpHR3.0)")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-placements", type=int, default=6,
                        help="Probe placements per trial (1 centered + N-1 random)")
    parser.add_argument("--trial-duration", type=float, default=30.0,
                        help="Trial duration in seconds")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 path")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base OU seed (trial i uses base_seed + i)")
    parser.add_argument("--plant-seed", type=int, default=42,
                        help="Plant connectivity seed (same as existing data)")
    parser.add_argument("--placement-seed", type=int, default=12345,
                        help="Seed for random probe placement offsets")
    parser.add_argument("--bin-ms", type=float, default=10.0,
                        help="Spike binning resolution in ms (default: 10)")
    parser.add_argument("--snr-cutoff", type=float, default=3.0,
                        help="SortedSpiking SNR cutoff (default: 3.0)")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--n-exc", type=int, default=800)
    parser.add_argument("--n-inh", type=int, default=200)
    parser.add_argument("--n-channels", type=int, default=50)
    parser.add_argument("--volume-um", type=float, default=500.0)
    parser.add_argument("--ou-mu", type=float, default=5.0)
    parser.add_argument("--ou-sigma", type=float, default=2.0)
    parser.add_argument("--I-bg-pA", type=float, default=15.0, help="v3 AdEx background current (pA)")
    parser.add_argument("--a-nS", type=float, default=4.0, help="v3 AdEx subthreshold adaptation (nS)")
    parser.add_argument("--b-pA", type=float, default=80.5, help="v3 AdEx spike-triggered adaptation (pA)")
    parser.add_argument("--tau-w-ms", type=float, default=144.0, help="v3 AdEx adaptation tau (ms)")
    args = parser.parse_args()

    t_start = time.time()

    # Generate probe placement offsets
    placement_rng = np.random.RandomState(args.placement_seed)
    offsets = _generate_probe_offsets(
        args.n_placements, args.volume_um, args.n_channels, placement_rng
    )

    print("=" * 70)
    print(f"Spiking Data Generation — Plant {args.plant}")
    print(f"  {args.n_trials} trials × {args.n_placements} placements = "
          f"{args.n_trials * args.n_placements} total runs")
    print(f"  Trial duration: {args.trial_duration}s")
    print(f"  Bin size: {args.bin_ms}ms, SNR cutoff: {args.snr_cutoff}")
    print(f"  Plant seed: {args.plant_seed}, OU base seed: {args.base_seed}")
    print(f"  Probe placements:")
    for i, (dx, dz) in enumerate(offsets):
        tag = "centered" if i == 0 else "random"
        print(f"    [{i}] ({tag}) offset=({dx:.1f}, {dz:.1f})µm")
    print("=" * 70)

    # Build all (trial, placement) jobs
    jobs = []
    for trial_idx in range(args.n_trials):
        for placement_idx, offset_xz in enumerate(offsets):
            jobs.append({
                "trial_idx": trial_idx,
                "placement_idx": placement_idx,
                "n_trials": args.n_trials,
                "ou_seed": args.base_seed + trial_idx,
                "sim_seed": args.base_seed + trial_idx * 1000 + placement_idx,
                "offset_xz": offset_xz,
                "plant_type": args.plant,
                "plant_seed": args.plant_seed,
                "n_exc": args.n_exc,
                "n_inh": args.n_inh,
                "n_channels": args.n_channels,
                "volume_um": args.volume_um,
                "trial_duration_s": args.trial_duration,
                "sample_period_ms": 1.0,
                "tau_smooth_ms": 20.0,
                "ou_tau": 0.05,
                "ou_sigma": args.ou_sigma,
                "ou_mu": args.ou_mu,
                "I_bg_pA": args.I_bg_pA,
                "a_nS": args.a_nS,
                "b_pA": args.b_pA,
                "tau_w_ms": args.tau_w_ms,
                "bin_ms": args.bin_ms,
                "snr_cutoff": args.snr_cutoff,
            })

    print(f"\nLaunching {len(jobs)} jobs with {args.n_workers} workers...")

    if args.n_workers == 1:
        results = [_run_trial_placement(a) for a in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.n_workers) as pool:
            results = pool.map(_run_trial_placement, jobs)

    # Organize results and save HDF5
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with h5py.File(args.output, "w") as f:
        # Time vectors
        sample_dt = 1.0 / 1000.0
        bin_dt = args.bin_ms / 1000.0
        T_fine = int(args.trial_duration / sample_dt)
        T_binned = int(args.trial_duration / bin_dt)

        f.create_dataset("t_fine", data=np.arange(T_fine) * sample_dt)
        f.create_dataset("t", data=np.arange(T_binned) * bin_dt)

        # Control inputs (shared across placements — from placement 0)
        u_list = []
        for trial_idx in range(args.n_trials):
            for r in results:
                if r["trial_idx"] == trial_idx and r["placement_idx"] == 0:
                    u_list.append(r["u"])
                    break
        u_stacked = np.stack(u_list)
        f.create_dataset("u", data=u_stacked, compression="gzip")

        # Per-placement groups
        for p_idx in range(args.n_placements):
            grp = f.create_group(f"placement_{p_idx}")

            p_results = sorted(
                [r for r in results if r["placement_idx"] == p_idx],
                key=lambda r: r["trial_idx"],
            )

            # n_sorted varies across trials due to stochastic spike sorting
            n_sorted_per_trial = np.array([r["n_sorted"] for r in p_results])
            max_n_sorted = int(n_sorted_per_trial.max())
            grp.attrs["max_n_sorted"] = max_n_sorted
            grp.attrs["offset_xz"] = list(p_results[0]["offset_xz"])
            grp.create_dataset("n_sorted_per_trial", data=n_sorted_per_trial)

            # Pad x_sorted to (n_trials, max_n_sorted, T_binned)
            T_binned = p_results[0]["x_sorted"].shape[1]
            x_sorted = np.zeros(
                (len(p_results), max_n_sorted, T_binned), dtype=np.int16
            )
            for i, r in enumerate(p_results):
                n_i = r["n_sorted"]
                x_sorted[i, :n_i, :] = r["x_sorted"]
            x_mua = np.stack([r["x_mua"] for r in p_results])
            # LFP: (n_trials, n_channels, T_binned), same binning as x_mua.
            x_lfp_tklfp = np.stack([r["x_lfp_tklfp"] for r in p_results])
            x_lfp_rwslfp = np.stack([r["x_lfp_rwslfp"] for r in p_results])

            grp.create_dataset("x_sorted", data=x_sorted, compression="gzip")
            grp.create_dataset("x_mua", data=x_mua, compression="gzip")
            grp.create_dataset("x_lfp_tklfp", data=x_lfp_tklfp, compression="gzip")
            grp.create_dataset("x_lfp_rwslfp", data=x_lfp_rwslfp, compression="gzip")
            grp.create_dataset("probe_coords", data=p_results[0]["probe_coords"])

            total_spikes = int(x_sorted.sum())
            print(f"  Placement {p_idx}: x_sorted={x_sorted.shape}, "
                  f"max_n_sorted={max_n_sorted}, "
                  f"n_sorted range=[{n_sorted_per_trial.min()}, "
                  f"{n_sorted_per_trial.max()}], spikes={total_spikes}")

        # Metadata
        f.attrs["plant"] = args.plant
        f.attrs["n_trials"] = args.n_trials
        f.attrs["n_placements"] = args.n_placements
        f.attrs["bin_ms"] = args.bin_ms
        f.attrs["snr_cutoff"] = args.snr_cutoff
        f.attrs["plant_seed"] = args.plant_seed
        f.attrs["base_seed"] = args.base_seed
        f.attrs["placement_seed"] = args.placement_seed
        f.attrs["trial_duration_s"] = args.trial_duration
        f.attrs["n_exc"] = args.n_exc
        f.attrs["n_inh"] = args.n_inh

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved to {args.output}")
    fsize = os.path.getsize(args.output) / (1024 ** 2)
    print(f"File size: {fsize:.1f} MB")


if __name__ == "__main__":
    main()

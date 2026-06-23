#!/usr/bin/env python3
"""Generate the 50-trial bidirectional v2 training dataset.

Uses build_plant_v2 (ChR2-H134R + eNpHR3.0).
Avoids monkey-patching by using a dedicated top-level worker function.

Usage:
    cd /snel/home/cbwash2/cleo-worktrees/bidir-v2-plant
    python -m modeling.scripts.generate_bidir_v2_data --n-workers 50
"""

import h5py
import numpy as np
import os
import sys
import argparse
import multiprocessing as mp
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _run_single_trial_v2(args_dict):
    """Worker function using build_plant_v2. Must be top-level for pickling."""
    import warnings
    warnings.filterwarnings("ignore")
    import brian2.only as b2
    b2.start_scope()

    from modeling.plant import build_plant_v2
    from modeling.datagen import generate_dataset

    trial_idx = args_dict["trial_idx"]
    n_trials = args_dict["n_trials"]
    ou_seed = args_dict["ou_seed"]

    print(f"[Worker] Trial {trial_idx + 1}/{n_trials} "
          f"(PID={mp.current_process().pid}, OU seed={ou_seed})")

    sim, devices = build_plant_v2(
        n_exc=args_dict["n_exc"],
        n_inh=args_dict["n_inh"],
        n_channels=args_dict["n_channels"],
        seed=args_dict["plant_seed"],
    )

    data = generate_dataset(
        sim, devices,
        duration_s=args_dict["trial_duration_s"],
        sample_period_ms=args_dict["sample_period_ms"],
        tau_smooth_ms=args_dict["tau_smooth_ms"],
        ou_tau=args_dict["ou_tau"],
        ou_sigma=args_dict["ou_sigma"],
        ou_mu=args_dict["ou_mu"],
        seed=ou_seed,
    )

    print(f"[Worker] Trial {trial_idx + 1}/{n_trials} done. "
          f"x={data['x'].shape}, u={data['u'].shape}")
    return {"x": data["x"], "u": data["u"]}


def main():
    parser = argparse.ArgumentParser(description="Generate bidir v2 training data")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--trial-duration", type=float, default=30.0)
    parser.add_argument("--n-workers", type=int, default=50)
    parser.add_argument("--output", type=str,
                        default="/snel/home/cbwash2/cleo/data/training_trials_bidir_v2.h5")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--plant-seed", type=int, default=42)
    parser.add_argument("--sample-period", type=float, default=1.0)
    parser.add_argument("--tau-smooth", type=float, default=20.0)
    parser.add_argument("--ou-tau", type=float, default=0.05)
    parser.add_argument("--ou-sigma", type=float, default=2.0)
    parser.add_argument("--ou-mu", type=float, default=5.0)
    args = parser.parse_args()

    print("=" * 60)
    print("Bidir V2 Data Generation (ChR2-H134R + eNpHR3.0)")
    print("=" * 60)
    print(f"  {args.n_trials} trials x {args.trial_duration}s = "
          f"{args.n_trials * args.trial_duration}s total")
    print(f"  Output: {args.output}")
    print(f"  Workers: {args.n_workers}")
    print(f"  OU params: mu={args.ou_mu}, sigma={args.ou_sigma}, tau={args.ou_tau}")

    # Build trial args
    trial_args = []
    for i in range(args.n_trials):
        trial_args.append({
            "trial_idx": i,
            "n_trials": args.n_trials,
            "ou_seed": args.base_seed + i,
            "plant_seed": args.plant_seed,
            "n_exc": 800,
            "n_inh": 200,
            "n_channels": 50,
            "trial_duration_s": args.trial_duration,
            "sample_period_ms": args.sample_period,
            "tau_smooth_ms": args.tau_smooth,
            "ou_tau": args.ou_tau,
            "ou_sigma": args.ou_sigma,
            "ou_mu": args.ou_mu,
        })

    t0 = time.time()

    # Run with spawn context (required by Brian2)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.n_workers) as pool:
        results = pool.map(_run_single_trial_v2, trial_args)

    # Stack results
    dt_s = args.sample_period / 1000.0
    min_steps = min(r["x"].shape[1] for r in results)
    x_stacked = np.stack([r["x"][:, :min_steps] for r in results])
    u_stacked = np.stack([r["u"][:, :min_steps] for r in results])
    t = np.arange(min_steps) * dt_s

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  x: {x_stacked.shape}, u: {u_stacked.shape}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("x", data=x_stacked, compression="gzip")
        f.create_dataset("u", data=u_stacked, compression="gzip")
        f.create_dataset("t", data=t)
        f.attrs["dt"] = dt_s
        f.attrs["n_trials"] = args.n_trials
        f.attrs["n_channels"] = x_stacked.shape[1]
        f.attrs["n_inputs"] = u_stacked.shape[1]
        f.attrs["tau_smooth_ms"] = args.tau_smooth
        f.attrs["trial_duration_s"] = args.trial_duration
        f.attrs["plant_type"] = "bidirectional_v2"
        f.attrs["opsin_exc"] = "ChR2(H134R) (450nm)"
        f.attrs["opsin_inh"] = "eNpHR3.0 (590nm)"

    print(f"\nSaved to {args.output}")
    print(f"  x: {x_stacked.shape}")
    print(f"  u: {u_stacked.shape}")
    print(f"  n_inputs: {u_stacked.shape[1]}")


if __name__ == "__main__":
    main()

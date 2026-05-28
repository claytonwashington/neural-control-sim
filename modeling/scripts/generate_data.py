#!/usr/bin/env python
"""Generate training data: run Cleo plant with OU noise, save u(t) and x(t).

Default: 10 trials x 30s, parallelized across cores.

Usage:
    # 10 trials of 30s each, 10 parallel workers (default)
    python -m modeling.scripts.generate_data

    # Custom: 5 trials of 60s, 5 workers
    python -m modeling.scripts.generate_data --n-trials 5 --trial-length 60 --n-workers 5

    # Sequential (debugging)
    python -m modeling.scripts.generate_data --n-workers 1
"""

import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser(description="Generate OU-driven Cleo training data")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of independent trials")
    parser.add_argument("--trial-length", type=float, default=30.0,
                        help="Duration of each trial in seconds")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Parallel workers (default: n_trials)")
    parser.add_argument("--sample-period", type=float, default=1.0, help="Sample period in ms")
    parser.add_argument("--tau-smooth", type=float, default=20.0, help="Smoothing tau in ms")
    parser.add_argument("--ou-tau", type=float, default=0.05, help="OU tau in seconds")
    parser.add_argument("--ou-sigma", type=float, default=2.0, help="OU sigma (mW/mm^2)")
    parser.add_argument("--ou-mu", type=float, default=5.0, help="OU mu (mW/mm^2)")
    parser.add_argument("--n-exc", type=int, default=800, help="Excitatory neurons")
    parser.add_argument("--n-inh", type=int, default=200, help="Inhibitory neurons")
    parser.add_argument("--n-channels", type=int, default=50, help="MUA channels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for HDF5 files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 1: Digital Twin Data Generation")
    print("=" * 60)
    print(f"  {args.n_trials} trials x {args.trial_length}s = "
          f"{args.n_trials * args.trial_length}s total")
    print(f"  {args.n_exc}E/{args.n_inh}I neurons, {args.n_channels} channels")
    print(f"  Workers: {args.n_workers or args.n_trials}")
    t_start = time.time()

    from modeling.datagen import generate_multi_trial
    from modeling.data import save_trials_h5

    data = generate_multi_trial(
        n_trials=args.n_trials,
        trial_duration_s=args.trial_length,
        sample_period_ms=args.sample_period,
        tau_smooth_ms=args.tau_smooth,
        ou_tau=args.ou_tau,
        ou_sigma=args.ou_sigma,
        ou_mu=args.ou_mu,
        base_seed=args.seed,
        plant_seed=args.seed,
        n_exc=args.n_exc,
        n_inh=args.n_inh,
        n_channels=args.n_channels,
        n_workers=args.n_workers,
    )

    # Save as contiguous 3D trial-stacked arrays
    out_path = os.path.join(args.output_dir, "training_trials.h5")
    print(f"\nSaving to {out_path}")
    print(f"  x shape: {data['x'].shape}  (n_trials, n_channels, n_steps)")
    print(f"  u shape: {data['u'].shape}  (n_trials, n_inputs, n_steps)")

    save_trials_h5(
        out_path,
        trial_data={
            "x": data["x"],
            "u": data["u"],
            "n_trials": data["n_trials"],
            "trial_bins": data["x"].shape[2],
        },
        dt=data["dt"],
        tau_smooth_ms=data["tau_smooth_ms"],
        trial_duration_s=data["trial_duration_s"],
        n_channels=data["n_channels"],
        n_inputs=data["n_inputs"],
    )

    elapsed = time.time() - t_start
    print(f"\nDone! Saved: {out_path}")
    print(f"Total wall time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate training data: run Cleo plant with OU noise, save u(t) and x(t).

Supports two modes:
  --mode single   : One long continuous trajectory (legacy)
  --mode multi    : Multiple independent trials (default, recommended)

Usage:
    # 10 trials of 30s each (default, recommended for training)
    python -m modeling.scripts.generate_data

    # Custom: 5 trials of 60s
    python -m modeling.scripts.generate_data --n-trials 5 --trial-length 60

    # Single continuous trajectory (legacy)
    python -m modeling.scripts.generate_data --mode single --duration 300
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Generate OU-driven Cleo training data")

    # Mode
    parser.add_argument("--mode", type=str, default="multi",
                        choices=["single", "multi"],
                        help="single: one long trajectory. multi: independent trials (default)")

    # Multi-trial params
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of independent trials (multi mode)")
    parser.add_argument("--trial-length", type=float, default=30.0,
                        help="Duration of each trial in seconds (multi mode)")

    # Single mode params
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Sim duration in seconds (single mode)")

    # Shared params
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
    t_start = time.time()

    if args.mode == "multi":
        _run_multi(args)
    else:
        _run_single(args)

    elapsed = time.time() - t_start
    print(f"\nTotal wall time: {elapsed/60:.1f} minutes")


def _run_multi(args):
    """Generate multiple independent trials."""
    from modeling.datagen import generate_multi_trial
    from modeling.data import save_trials_h5

    print(f"\nMode: MULTI-TRIAL")
    print(f"  {args.n_trials} trials x {args.trial_length}s = "
          f"{args.n_trials * args.trial_length}s total")
    print(f"  {args.n_exc}E/{args.n_inh}I neurons, {args.n_channels} channels")

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

    print(f"\nDone! Saved: {out_path}")


def _run_single(args):
    """Generate one long continuous trajectory."""
    from modeling.plant import build_plant
    from modeling.datagen import generate_dataset
    from modeling.data import save_continuous_h5, chop_to_trials, save_trials_h5

    print(f"\nMode: SINGLE TRAJECTORY")
    print(f"  {args.duration}s continuous")
    print(f"  {args.n_exc}E/{args.n_inh}I neurons, {args.n_channels} channels")

    sim, devices = build_plant(
        n_exc=args.n_exc,
        n_inh=args.n_inh,
        n_channels=args.n_channels,
        seed=args.seed,
    )

    data = generate_dataset(
        sim, devices,
        duration_s=args.duration,
        sample_period_ms=args.sample_period,
        tau_smooth_ms=args.tau_smooth,
        ou_tau=args.ou_tau,
        ou_sigma=args.ou_sigma,
        ou_mu=args.ou_mu,
        seed=args.seed,
    )

    # Save continuous
    cont_path = os.path.join(args.output_dir, "training_continuous.h5")
    print(f"\nSaving continuous data to {cont_path}")
    print(f"  x: {data['x'].shape}, u: {data['u'].shape}")
    save_continuous_h5(
        cont_path, x=data["x"], u=data["u"], t=data["t"],
        dt=data["dt"], tau_smooth_ms=data["tau_smooth_ms"],
        n_channels=data["n_channels"], n_inputs=data["n_inputs"],
    )

    # Also chop into trials
    trial_duration_ms = 1000.0
    trial_bins = int(trial_duration_ms / args.sample_period)
    trials = chop_to_trials(data["x"], trial_bins, u=data["u"])

    trial_path = os.path.join(args.output_dir, "training_trials.h5")
    print(f"Saving {trials['n_trials']} trials to {trial_path}")
    save_trials_h5(trial_path, trials, dt=data["dt"],
                   tau_smooth_ms=data["tau_smooth_ms"],
                   trial_duration_ms=trial_duration_ms)

    print(f"\nDone! Saved: {cont_path}, {trial_path}")


if __name__ == "__main__":
    main()

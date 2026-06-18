#!/usr/bin/env python3
"""Generate the 50-trial bidirectional training dataset.

Usage:
    cd /mnt/cbwash2/cleo-worktrees/bidirectional-plant
    python -m modeling.scripts.generate_bidirectional_data
"""

import h5py
import numpy as np
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modeling.datagen import generate_multi_trial


def main():
    parser = argparse.ArgumentParser(description="Generate bidirectional training data")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--trial-duration", type=float, default=30.0,
                        help="Trial duration in seconds")
    parser.add_argument("--n-workers", type=int, default=10,
                        help="Parallel workers (default 10)")
    parser.add_argument("--output", type=str,
                        default="/mnt/cbwash2/cleo/data/training_trials_bidirectional.h5")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--plant-seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.n_trials} trials x {args.trial_duration}s")
    print(f"  Output: {args.output}")
    print(f"  Workers: {args.n_workers}")

    data = generate_multi_trial(
        n_trials=args.n_trials,
        trial_duration_s=args.trial_duration,
        base_seed=args.base_seed,
        plant_seed=args.plant_seed,
        n_workers=args.n_workers,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("x", data=data["x"], compression="gzip")
        f.create_dataset("u", data=data["u"], compression="gzip")
        f.create_dataset("t", data=data["t"])
        f.attrs["dt"] = data["dt"]
        f.attrs["n_trials"] = data["n_trials"]
        f.attrs["n_channels"] = data["n_channels"]
        f.attrs["n_inputs"] = data["n_inputs"]
        f.attrs["tau_smooth_ms"] = data["tau_smooth_ms"]
        f.attrs["trial_duration_s"] = data["trial_duration_s"]
        f.attrs["plant_type"] = "bidirectional"
        f.attrs["opsin_exc"] = "ChrimsonR (590nm)"
        f.attrs["opsin_inh"] = "GtACR2 (470nm)"

    print(f"\nSaved to {args.output}")
    print(f"  x: {data['x'].shape}")
    print(f"  u: {data['u'].shape}")
    print(f"  n_inputs: {data['n_inputs']}")


if __name__ == "__main__":
    main()

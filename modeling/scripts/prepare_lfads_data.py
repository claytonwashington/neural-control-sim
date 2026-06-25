#!/usr/bin/env python3
"""Convert spiking_plant3.h5 to lfads-torch HDF5 format.

Chops continuous 30s trials into overlapping segments and formats
for lfads-torch SessionDataModule.

Usage:
    python -m modeling.scripts.prepare_lfads_data \\
        --input data/spiking_plant3.h5 --placement 0 \\
        --output data/spiking_plant3_lfads_p0.h5 \\
        --seg-len 100 --overlap 50
"""

import argparse
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prepare LFADS data from spiking HDF5")
    parser.add_argument("--input", required=True, help="Input spiking HDF5 file")
    parser.add_argument("--output", required=True, help="Output lfads-torch HDF5 file")
    parser.add_argument("--placement", type=int, default=0)
    parser.add_argument("--seg-len", type=int, default=100,
                        help="Segment length in 10ms bins (default: 100 = 1000ms)")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Overlap in 10ms bins (default: 50 = 500ms)")
    parser.add_argument("--n-test-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-mua", action="store_true",
                        help="Also export MUA data as separate datasets")
    args = parser.parse_args()

    print("Loading %s (placement %d)..." % (args.input, args.placement))
    with h5py.File(args.input, "r") as f:
        grp = f["placement_%d" % args.placement]
        x_sorted = grp["x_sorted"][:]             # (50, max_n, 3000) int16
        n_sorted = grp["n_sorted_per_trial"][:]    # (50,)
        u_1ms = f["u"][:]                          # (50, 2, 30000)
        if args.include_mua:
            x_mua = grp["x_mua"][:]               # (50, 50, 3000) float64

    n_trials, max_n, n_bins = x_sorted.shape
    print("  %d trials, %d max neurons, %d bins (%.1fs)" %
          (n_trials, max_n, n_bins, n_bins * 0.01))

    # Convert to (trials, time, channels) format
    # Keep as raw spike counts for Poisson NLL
    x_counts = x_sorted.astype(np.float32).transpose(0, 2, 1)  # (50, 3000, max_n)

    # Downsample u from 1ms to 10ms, then transpose
    u_10ms = u_1ms.reshape(n_trials, 2, n_bins, 10).mean(axis=-1)  # (50, 2, 3000)
    u_10ms = u_10ms.transpose(0, 2, 1).astype(np.float32)  # (50, 3000, 2)

    # Train/test split (same seed as CA-NODE for fair comparison)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_trials)
    test_idx = perm[:args.n_test_trials]
    train_idx = perm[args.n_test_trials:]
    print("  Train: %d trials, Test: %d trials" % (len(train_idx), len(test_idx)))

    # Chop into overlapping segments
    stride = args.seg_len - args.overlap
    print("  Segment: %d steps (%dms), overlap: %d steps (%dms), stride: %d" %
          (args.seg_len, args.seg_len * 10, args.overlap, args.overlap * 10, stride))

    def chop_trials(indices, x_data, u_data):
        segs_x, segs_u = [], []
        for i in indices:
            for t in range(0, x_data.shape[1] - args.seg_len + 1, stride):
                segs_x.append(x_data[i, t:t+args.seg_len])  # (seg_len, n_ch)
                segs_u.append(u_data[i, t:t+args.seg_len])  # (seg_len, 2)
        return np.array(segs_x, dtype=np.float32), np.array(segs_u, dtype=np.float32)

    train_x, train_u = chop_trials(train_idx, x_counts, u_10ms)
    valid_x, valid_u = chop_trials(test_idx, x_counts, u_10ms)

    # Per trial: (3000 - 100) / 50 + 1 = 59 segments
    print("  Train segments: %d, Valid segments: %d" %
          (train_x.shape[0], valid_x.shape[0]))
    print("  Shapes: encod_data=%s, ext_input=%s" %
          (train_x.shape, train_u.shape))

    # Save lfads-torch format
    print("\nSaving to %s..." % args.output)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("train_encod_data", data=train_x)
        f.create_dataset("train_recon_data", data=train_x)  # same as encod
        f.create_dataset("train_ext_input",  data=train_u)
        f.create_dataset("valid_encod_data", data=valid_x)
        f.create_dataset("valid_recon_data", data=valid_x)
        f.create_dataset("valid_ext_input",  data=valid_u)

    print("Done! HDF5 keys:")
    with h5py.File(args.output, "r") as f:
        for k in f.keys():
            print("  %s: %s %s" % (k, f[k].shape, f[k].dtype))

    if args.include_mua:
        # Also create MUA version
        mua_output = args.output.replace(".h5", "_mua.h5")
        x_mua_t = x_mua.astype(np.float32).transpose(0, 2, 1)  # (50, 3000, 50)
        train_mua, train_mua_u = chop_trials(train_idx, x_mua_t, u_10ms)
        valid_mua, valid_mua_u = chop_trials(test_idx, x_mua_t, u_10ms)
        print("\nSaving MUA version to %s..." % mua_output)
        with h5py.File(mua_output, "w") as f:
            f.create_dataset("train_encod_data", data=train_mua)
            f.create_dataset("train_recon_data", data=train_mua)
            f.create_dataset("train_ext_input",  data=train_mua_u)
            f.create_dataset("valid_encod_data", data=valid_mua)
            f.create_dataset("valid_recon_data", data=valid_mua)
            f.create_dataset("valid_ext_input",  data=valid_mua_u)
        print("  MUA shapes: encod=%s" % (train_mua.shape,))


if __name__ == "__main__":
    main()

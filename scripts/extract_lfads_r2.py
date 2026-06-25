"""Extract R-squared from best LFADS PBT checkpoint using lfads-torch post_run API.

Usage:
    python scripts/extract_lfads_r2.py \
      --run-dir results/lfads_spiking_ext/260622_spiking_ext \
      --bin-width 0.01 --device cuda:6
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

sys.path.insert(0, "/snel/home/cbwash2/lfads-torch-cuda12/lfads-torch")
from lfads_torch.model import LFADS
from lfads_torch.post_run.analysis import run_posterior_sampling


def find_best_trial(run_dir):
    """Find best trial from progress.csv files."""
    trial_dirs = sorted(glob.glob(os.path.join(run_dir, "run_model_*")))
    best_metric = float("inf")
    best_dir = None
    for td in trial_dirs:
        progress = os.path.join(td, "progress.csv")
        if not os.path.exists(progress):
            continue
        import csv
        with open(progress) as f:
            rows = list(csv.DictReader(f))
        if not rows or "valid/recon_smth" not in rows[-1]:
            continue
        val = float(rows[-1]["valid/recon_smth"])
        if val < best_metric:
            best_metric = val
            best_dir = td
    if best_dir is None:
        raise ValueError("No valid trials found")
    return best_dir, best_metric


def load_model_from_trial(trial_dir, device):
    """Load LFADS model from a PBT trial directory using its config."""
    # Find the checkpoint
    ckpt_files = sorted(glob.glob(os.path.join(trial_dir, "checkpoint_*", "*.ckpt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {trial_dir}")
    ckpt_path = ckpt_files[-1]

    # Load the model config from the params.json file
    params_file = os.path.join(trial_dir, "params.json")
    with open(params_file) as f:
        params = json.load(f)

    # The params contain the hydra overrides. Load model via config
    model_name = params.get("model", "cleo_spiking_ext")
    datamodule_name = params.get("datamodule", "cleo_spiking")

    # Use hydra to load configs
    config_dir = "/snel/home/cbwash2/lfads-torch-cuda12/lfads-torch/configs"
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="pbt_cleo.yaml",
                     overrides=[f"model={model_name}", f"datamodule={datamodule_name}"])

    # Build the model from config
    from hydra.utils import instantiate
    model = instantiate(cfg.model)

    # Load state dict from checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, ckpt_path, datamodule_name


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bin-width", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    # Find best trial
    best_dir, recon_smth = find_best_trial(args.run_dir)
    print(f"Best trial: {os.path.basename(best_dir)}, recon_smth={recon_smth:.4f}")

    # Load model
    model, ckpt_path, dm_name = load_model_from_trial(best_dir, args.device)
    print(f"Loaded checkpoint: {ckpt_path}")

    # Load data directly
    # Figure out the data file from the datamodule config
    dm_config_path = f"/snel/home/cbwash2/lfads-torch-cuda12/lfads-torch/configs/datamodule/{dm_name}.yaml"
    with open(dm_config_path) as f:
        dm_cfg = OmegaConf.load(f)
    data_path = dm_cfg.datafile_pattern
    print(f"Data: {data_path}")

    with h5py.File(data_path, "r") as f:
        valid_encod = torch.tensor(np.array(f["valid_encod_data"]), dtype=torch.float32)
        valid_recon = torch.tensor(np.array(f["valid_recon_data"]), dtype=torch.float32)
        if "valid_ext_input" in f:
            valid_ext = torch.tensor(np.array(f["valid_ext_input"]), dtype=torch.float32)
        else:
            valid_ext = torch.zeros(valid_encod.shape[0], valid_encod.shape[1], 0)

    n_samps, n_steps, n_neurons = valid_recon.shape
    print(f"Validation: {n_samps} samples, {n_steps} steps, {n_neurons} neurons")

    # Run posterior sampling manually
    from lfads_torch.tuples import SessionBatch
    ic_enc_seq_len = getattr(model.hparams, "ic_enc_seq_len", 0)
    if ic_enc_seq_len > 0:
        sv_mask = torch.ones(n_samps, n_steps - ic_enc_seq_len, dtype=torch.float32)
        ext_input_trimmed = valid_ext[:, ic_enc_seq_len:, :]
        recon_trimmed = valid_recon[:, ic_enc_seq_len:, :]
    else:
        sv_mask = torch.ones(n_samps, n_steps, dtype=torch.float32)
        ext_input_trimmed = valid_ext
        recon_trimmed = valid_recon

    batch = SessionBatch(
        encod_data=valid_encod.to(args.device),
        recon_data=recon_trimmed.to(args.device),
        ext_input=ext_input_trimmed.to(args.device),
        truth=torch.full((n_samps, 0, 0), float("nan")).to(args.device),
        sv_mask=sv_mask.to(args.device),
    )

    # Average over posterior samples using model.forward directly
    with torch.no_grad():
        for i in range(args.n_samples):
            output_dict = model.forward(
                {0: batch},
                sample_posteriors=True,
                output_means=True,
            )
            # output_dict is {session_id: SessionOutput(output_params, factors, ...)}
            rates = output_dict[0].output_params.detach()  # (n_samps, n_steps, n_neurons)
            if i == 0:
                rate_sum = rates
            else:
                rate_sum = rate_sum + rates
        mean_rates = (rate_sum / args.n_samples).cpu().numpy()

    # Both are in spikes/bin (model gives exp(log_rate) = expected spikes/bin)
    actual = recon_trimmed.numpy()  # raw spike counts per bin
    pred = mean_rates  # predicted rates (spikes/bin)

    # -- Raw R2 (on raw spike counts, noisy) --
    r2_per_neuron_raw = [compute_r2(actual[:, :, n].flatten(), pred[:, :, n].flatten())
                         for n in range(n_neurons)]
    r2_overall_raw = compute_r2(actual.flatten(), pred.flatten())

    # -- Smoothed R2 (gaussian smooth both, fair comparison with CA-NODE) --
    from scipy.ndimage import gaussian_filter1d
    sigma = 5  # 5 bins = 50ms smoothing at 10ms bins
    actual_smth = np.stack([gaussian_filter1d(actual[:, :, n], sigma, axis=1) for n in range(n_neurons)], axis=2)
    pred_smth = np.stack([gaussian_filter1d(pred[:, :, n], sigma, axis=1) for n in range(n_neurons)], axis=2)

    r2_per_neuron_smth = [compute_r2(actual_smth[:, :, n].flatten(), pred_smth[:, :, n].flatten())
                          for n in range(n_neurons)]
    r2_overall_smth = compute_r2(actual_smth.flatten(), pred_smth.flatten())

    results = {
        "recon_smth_loss": recon_smth,
        "r2_raw_overall": float(r2_overall_raw),
        "r2_raw_mean": float(np.mean(r2_per_neuron_raw)),
        "r2_smoothed_overall": float(r2_overall_smth),
        "r2_smoothed_mean": float(np.mean(r2_per_neuron_smth)),
        "r2_smoothed_median": float(np.median(r2_per_neuron_smth)),
        "r2_smoothed_per_neuron": [float(r) for r in r2_per_neuron_smth],
        "smoothing_sigma_bins": sigma,
        "n_posterior_samples": args.n_samples,
        "n_neurons": n_neurons,
    }

    print()
    print("=" * 50)
    print("LFADS R-squared Results")
    print("=" * 50)
    print(f"  recon_smth (loss):      {recon_smth:.4f}")
    print(f"  R2 raw (overall):       {r2_overall_raw:.4f}")
    print(f"  R2 raw (mean/neuron):   {np.mean(r2_per_neuron_raw):.4f}")
    print(f"  R2 smoothed (overall):  {r2_overall_smth:.4f}")
    print(f"  R2 smoothed (mean):     {np.mean(r2_per_neuron_smth):.4f}")
    print(f"  R2 smoothed (median):   {np.median(r2_per_neuron_smth):.4f}")

    out_file = os.path.join(args.run_dir, "r2_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()

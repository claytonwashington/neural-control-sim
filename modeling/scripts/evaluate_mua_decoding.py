import os
import sys
import json
import argparse
import glob
import numpy as np
import h5py
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

def get_repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, get_repo_root())
from modeling.scripts.validation_plots import load_model, infer, infer_n4sid

# For LFADS
try:
    sys.path.insert(0, "/snel/home/cbwash2/lfads-torch-cuda12/lfads-torch")
    from lfads_torch.tuples import SessionBatch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
except ImportError:
    pass

def load_lfads_model(trial_dir, device):
    ckpt_files = sorted(glob.glob(os.path.join(trial_dir, "checkpoint_*", "*.ckpt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {trial_dir}")
    ckpt_path = ckpt_files[-1]
    with open(os.path.join(trial_dir, "params.json")) as f:
        params = json.load(f)
    model_name = params.get("model", "cleo_spiking_ext")
    datamodule_name = params.get("datamodule", "cleo_spiking")
    config_dir = "/snel/home/cbwash2/lfads-torch-cuda12/lfads-torch/configs"
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="pbt_cleo.yaml", overrides=[f"model={model_name}", f"datamodule={datamodule_name}"])
    model = instantiate(cfg.model)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model = model.to(device)
    model.eval()
    return model

def get_lfads_predictions(model, x, u, device):
    # x: (trials, neurons, time)
    # u: (trials, time, u_dim)
    n_samps = x.shape[0]
    n_neurons = x.shape[1]
    n_steps = x.shape[2]
    
    ve = torch.tensor(x.transpose(0, 2, 1), dtype=torch.float32) # (T, steps, N)
    vr = torch.tensor(x.transpose(0, 2, 1), dtype=torch.float32)
    vx = torch.tensor(u, dtype=torch.float32) # (T, steps, U)
    
    batch = SessionBatch(
        encod_data=ve.to(device),
        recon_data=vr.to(device),
        ext_input=vx.to(device),
        truth=torch.full((n_samps, 0, 0), float("nan")).to(device),
        sv_mask=torch.ones(n_samps, n_steps).to(device),
    )
    with torch.no_grad():
        out = model.forward({0: batch}, sample_posteriors=False, output_means=True)
        rates = out[0].output_params.detach().cpu().numpy() # (trials, time, neurons)
    # Return as (channels, T) = (neurons, trials * time)
    return rates.transpose(2, 0, 1).reshape(n_neurons, -1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate MUA decoding from model predictions")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-file", default=None)
    args = parser.parse_args()

    manifest_path = os.path.join(args.run_dir, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} not found.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    model_type = manifest.get("model_type")
    data_path = args.data_file or manifest.get("data_file")
    if not os.path.isabs(data_path):
        data_path = os.path.join(get_repo_root(), data_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading data from {data_path}...")
    with h5py.File(data_path, "r") as f:
        p0 = f["placement_0"]
        x_sorted = np.array(p0["x_sorted"]) # (trials, neurons, time)
        x_mua = np.array(p0["x_mua"])       # (trials, channels, time)
        u = np.array(f["u"])                # (trials, time, u_dim)
        dt = f.attrs.get("dt", 0.001)

    # 40/10 split
    n_trials = x_sorted.shape[0]
    n_train = 40
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_trials)

    x_train, u_train = x_sorted[train_idx], u[train_idx]
    x_val, u_val = x_sorted[val_idx], u[val_idx]
    mua_train, mua_val = x_mua[train_idx], x_mua[val_idx]

    T_pts = x_train.shape[2]
    x_train_flat = x_train.transpose(1, 0, 2).reshape(x_train.shape[1], -1)
    u_train_flat = u_train.transpose(2, 0, 1).reshape(u_train.shape[2], -1)
    x_val_flat = x_val.transpose(1, 0, 2).reshape(x_val.shape[1], -1)
    u_val_flat = u_val.transpose(2, 0, 1).reshape(u_val.shape[2], -1)

    mua_train_flat = mua_train.transpose(0, 2, 1).reshape(-1, mua_train.shape[1])
    mua_val_flat = mua_val.transpose(0, 2, 1).reshape(-1, mua_val.shape[1])

    if model_type == "lfads":
        best_dir = None
        best_loss = float("inf")
        for d in glob.glob(os.path.join(args.run_dir, "*", "run_model_*")):
            try:
                with open(os.path.join(d, "result.json")) as f:
                    for line in f:
                        res = json.loads(line)
                        if "recon_smth" in res and res["recon_smth"] < best_loss:
                            best_loss = res["recon_smth"]
                            best_dir = d
            except:
                pass
        print(f"Loading LFADS from {best_dir}...")
        model = load_lfads_model(best_dir, device)
        pred_train_flat = get_lfads_predictions(model, x_train, u_train, device).T
        pred_val_flat = get_lfads_predictions(model, x_val, u_val, device).T
    elif model_type == "n4sid":
        ckpt_path = os.path.join(args.run_dir, "model.npz")
        pred_train_flat = infer_n4sid(ckpt_path, x_train_flat, u_train_flat).T
        pred_val_flat = infer_n4sid(ckpt_path, x_val_flat, u_val_flat).T
    else:
        ckpt_path = os.path.join(args.run_dir, "model.pt")
        model, _ = load_model(model_type, ckpt_path, device)
        model.eval()
        pred_train_flat = infer(model_type, model, x_train_flat, u_train_flat, dt, device).T
        pred_val_flat = infer(model_type, model, x_val_flat, u_val_flat, dt, device).T

    # Remove NaNs (infer returns NaN for unpredicted window borders)
    valid_train = ~np.isnan(pred_train_flat).any(axis=1)
    valid_val = ~np.isnan(pred_val_flat).any(axis=1)

    print(f"Fitting RidgeCV on {valid_train.sum()} training points...")
    X_train_features = np.log(np.clip(pred_train_flat[valid_train], 1e-7, None))
    Y_train = mua_train_flat[valid_train]
    
    reg = RidgeCV(alphas=np.logspace(-3, 3, 7))
    reg.fit(X_train_features, Y_train)

    print("Evaluating on validation set...")
    X_val_features = np.log(np.clip(pred_val_flat[valid_val], 1e-7, None))
    Y_val = mua_val_flat[valid_val]
    
    Y_val_pred = reg.predict(X_val_features)
    
    overall_r2 = r2_score(Y_val, Y_val_pred)
    per_channel_r2 = r2_score(Y_val, Y_val_pred, multioutput='raw_values')
    
    print(f"Overall MUA Decoding R2: {overall_r2:.4f}")
    
    results = {
        "mua_r2_overall": float(overall_r2),
        "mua_r2_mean_channel": float(np.mean(per_channel_r2)),
        "mua_r2_median_channel": float(np.median(per_channel_r2)),
        "alpha": float(reg.alpha_),
    }
    out_path = os.path.join(args.run_dir, "mua_decoding_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()

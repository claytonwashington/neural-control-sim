#!/usr/bin/env python
"""Run a parallel hyperparameter sweep for discrete-time sequence models (GRU & ED-CA-NODE) on GPU.

Launches multiple configurations in parallel using subprocesses,
monitors their execution, logs outputs, and summarizes results in a table.
"""

import argparse
import os
import subprocess
import time
import json
import numpy as np
from modeling.scripts.preflight_check import add_preflight_args, validate_preflight


def _require_tmux():
    """Refuse to run sweep outside tmux."""
    if not os.environ.get("TMUX"):
        print(
            "ERROR: Sweep scripts must run inside a tmux session.\n"
            "  Training runs must be in tmux to survive disconnects.\n"
            "  Use: tmux new-session -s <session_name>",
            file=sys.stderr,
        )
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Parallel hyperparameter sweep for sequence models")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-test-trials", type=int, default=10,
                        help="Number of test trials (default: 10)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="Number of GPU cards to use (defaults to 4 to use half of gpu1)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum parallel runs (capped by n_gpus)")
    parser.add_argument("--output-dir", type=str, default="results/sweep_sequence")
    add_preflight_args(parser)
    args = parser.parse_args()

    # Cap workers at available GPUs
    args.max_workers = min(args.max_workers, args.n_gpus)

    os.makedirs(args.output_dir, exist_ok=True)

    # Define sweep configs for GRU and Discrete-CA (ED-CA-NODE)
    configs = [
        # --- GRU configs (5) ---
        {"model": "gru", "hidden": 128, "n_layers": 1, "lr": 1e-3, "compile": True},
        {"model": "gru", "hidden": 128, "n_layers": 2, "lr": 1e-3, "compile": True},
        {"model": "gru", "hidden": 256, "n_layers": 1, "lr": 1e-3, "compile": True},
        {"model": "gru", "hidden": 128, "n_layers": 1, "lr": 1e-4, "compile": True},
        {"model": "gru", "hidden": 256, "n_layers": 1, "lr": 1e-4, "compile": True},

        # --- Discrete Control-Affine (ED-CA-NODE) configs (5) ---
        {"model": "discrete-ca", "hidden": 128, "n_layers": 2, "lr": 1e-4, "compile": True},
        {"model": "discrete-ca", "hidden": 128, "n_layers": 3, "lr": 1e-4, "compile": True},
        {"model": "discrete-ca", "hidden": 256, "n_layers": 2, "lr": 1e-4, "compile": True},
        {"model": "discrete-ca", "hidden": 128, "n_layers": 2, "lr": 2e-4, "compile": True},
        {"model": "discrete-ca", "hidden": 128, "n_layers": 2, "lr": 5e-5, "compile": True},
    ]

    n_configs = len(configs)
    print("=" * 70)
    print(f"Starting Sequence Model Sweep (Total configurations: {n_configs})")
    print(f"Max parallel workers: {args.max_workers} across {args.n_gpus} GPUs")
    print(f"Dataset: {args.data}")
    print(f"Epochs per run: {args.n_epochs}")
    print("=" * 70)

    active_processes = []
    completed_runs = []
    pending_configs = list(enumerate(configs))
    free_gpus = list(range(args.n_gpus))

    t_start = time.time()

    while pending_configs or active_processes:
        # Launch new workers if slots are available
        while len(active_processes) < args.max_workers and pending_configs and free_gpus:
            gpu_id = free_gpus.pop(0)
            idx, config = pending_configs.pop(0)
            
            run_id = f"run_{idx:02d}_{config['model']}_h{config['hidden']}_l{config['n_layers']}_lr{config['lr']}"
            if config["compile"]:
                run_id += "_compiled"

            log_path = os.path.join(args.output_dir, f"{run_id}.log")
            model_path = os.path.join(args.output_dir, f"{run_id}.pt")

            # Build command
            cmd = [
                "python", "-m", "modeling.scripts.fit_gru",
                "--model", config["model"],
                "--data", args.data,
                "--n-epochs", str(args.n_epochs),
                "--batch-size", str(args.batch_size),
                "--hidden", str(config["hidden"]),
                "--n-layers", str(config["n_layers"]),
                "--lr", str(config["lr"]),
                "--device", args.device,
                "--output-dir", os.path.join(args.output_dir, run_id),
                "--save-model", model_path,
                "--n-test-trials", str(args.n_test_trials),
            ]
            if config["compile"]:
                cmd.append("--compile")

            log_file = open(log_path, "w")

            env = os.environ.copy()
            env["PREFLIGHT_TOKEN"] = _preflight_token
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            print(f"[{idx+1}/{n_configs}] Launching {run_id} on GPU {gpu_id} ...")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                text=True
            )
            active_processes.append({
                "process": proc,
                "config": config,
                "idx": idx,
                "run_id": run_id,
                "log_path": log_path,
                "log_file": log_file,
                "gpu_id": gpu_id,
                "t_launch": time.time()
            })

        time.sleep(2.0)

        still_active = []
        for job in active_processes:
            poll = job["process"].poll()
            if poll is None:
                still_active.append(job)
            else:
                job["log_file"].close()
                elapsed = time.time() - job["t_launch"]
                print(f"  * Config {job['idx']+1} ({job['run_id']}) finished on GPU {job['gpu_id']} in {elapsed:.1f}s")
                free_gpus.append(job["gpu_id"])

                # Parse metrics from logs (reusing function)
                metrics = parse_run_log(job["log_path"])
                metrics["elapsed_s"] = elapsed
                metrics["idx"] = job["idx"]
                metrics["config"] = job["config"]
                metrics["run_id"] = job["run_id"]
                metrics["gpu_id"] = job["gpu_id"]
                completed_runs.append(metrics)

        active_processes = still_active

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"Sweep complete in {total_time/60:.2f} minutes!")
    print("=" * 70)

    # Sort results by windowed R² (descending)
    completed_runs.sort(key=lambda r: -r.get("mean_windowed_r2", -999))

    # Print summary table
    print(f"\n{'Idx':<4} {'Model':<11} {'Hidden':<6} {'Layers':<6} {'LR':<7} "
          f"{'Time':<8} {'Train L':<9} {'Val L':<9} "
          f"{'Win R2':<8} {'Win MSE':<8} {'OL R2':<8}")
    print("-" * 90)
    for run in completed_runs:
        cfg = run["config"]
        time_str = f"{run['elapsed_s']/60:.1f}m"
        train_l = f"{run.get('best_train_loss', float('nan')):.5f}"
        val_l = f"{run.get('best_val_loss', float('nan')):.5f}"
        win_r2 = f"{run.get('mean_windowed_r2', float('nan')):.4f}"
        win_mse = f"{run.get('mean_windowed_mse', float('nan')):.4f}"
        ol_r2 = f"{run.get('mean_test_r2', float('nan')):.4f}"
        print(f"{run['idx']+1:<4} {cfg['model']:<11} {cfg['hidden']:<6} {cfg['n_layers']:<6} "
              f"{cfg['lr']:<7} {time_str:<8} {train_l:<9} {val_l:<9} "
              f"{win_r2:<8} {win_mse:<8} {ol_r2:<8}")

    # Save summary json
    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(completed_runs, f, indent=2)
    print(f"\nSaved sweep summary to {summary_path}")


def parse_run_log(log_path):
    """Parse output log to extract training history and test performance."""
    best_train = float("inf")
    best_val = float("inf")
    test_r2s = []
    test_mses = []
    windowed_r2s = []
    windowed_mses = []

    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if "train:" in line and "val:" in line:
                try:
                    parts = line.split("|")
                    train_val = parts[1].strip().split(":")
                    val_val = parts[2].strip().split(":")
                    train_loss = float(train_val[1].strip())
                    val_loss = float(val_val[1].strip())
                    if train_loss < best_train:
                        best_train = train_loss
                    if val_loss < best_val:
                        best_val = val_loss
                except:
                    pass
            elif "Trial" in line and "MSE=" in line and "R2=" in line and "window" not in line and "R_win" not in line:
                try:
                    parts = line.split(":")
                    metrics = parts[1].strip().split(",")
                    mse = float(metrics[0].split("=")[1].strip())
                    r2 = float(metrics[1].split("=")[1].strip())
                    test_r2s.append(r2)
                    test_mses.append(mse)
                except:
                    pass
            elif "R_win=" in line:
                try:
                    parts = line.split("R_win=")
                    r2 = float(parts[1].strip())
                    mse_part = line.split("MSE=")[1].split(",")[0]
                    mse = float(mse_part.strip())
                    windowed_r2s.append(r2)
                    windowed_mses.append(mse)
                except:
                    pass
            elif "Avg 200-step" in line:
                try:
                    mse_part = line.split("MSE=")[1].split(",")[0]
                    r2_part = line.split("R2=")[1].strip()
                    windowed_r2s = [float(r2_part)]
                    windowed_mses = [float(mse_part)]
                except:
                    pass

    result = {
        "best_train_loss": best_train if best_train != float("inf") else None,
        "best_val_loss": best_val if best_val != float("inf") else None,
    }
    if len(test_r2s) >= 1:
        result["mean_test_r2"] = float(np.mean(test_r2s))
    if windowed_r2s:
        result["mean_windowed_r2"] = float(np.mean(windowed_r2s))
        result["mean_windowed_mse"] = float(np.mean(windowed_mses))

    return result


if __name__ == "__main__":
    main()

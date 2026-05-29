#!/usr/bin/env python
"""Run a parallel hyperparameter sweep and architecture search for CA-NODE on GPU.

Launches multiple configurations in parallel using subprocesses,
monitors their execution, logs outputs, and summarizes results in a table.
"""

import argparse
import os
import subprocess
import time
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Parallel hyperparameter sweep for CA-NODE")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum parallel runs on the GPU")
    parser.add_argument("--output-dir", type=str, default="results/sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Define the sweep configurations
    # We will test hidden dimensions, layers, learning rate, and solver method
    configs = [
        # Baseline dopri5 check (to compare speed/accuracy)
        {"hidden": 128, "n_layers": 2, "lr": 1e-3, "method": "dopri5", "compile": False},
        {"hidden": 128, "n_layers": 2, "lr": 1e-3, "method": "dopri5", "compile": True},
        
        # Scaling up models with dopri5 and compile
        {"hidden": 256, "n_layers": 2, "lr": 1e-3, "method": "dopri5", "compile": True},
        {"hidden": 256, "n_layers": 3, "lr": 1e-3, "method": "dopri5", "compile": True},
        
        # Slower learning rates with dopri5
        {"hidden": 128, "n_layers": 2, "lr": 5e-4, "method": "dopri5", "compile": True},
        {"hidden": 256, "n_layers": 2, "lr": 5e-4, "method": "dopri5", "compile": True},
        {"hidden": 256, "n_layers": 3, "lr": 5e-4, "method": "dopri5", "compile": True},
        
        # Keep one compiled rk4 for a clean speed/accuracy benchmark
        {"hidden": 128, "n_layers": 2, "lr": 1e-3, "method": "rk4", "compile": True},
    ]

    print("=" * 70)
    print(f"Starting CA-NODE Parallel Sweep (Total configurations: {len(configs)})")
    print(f"Max parallel workers: {args.max_workers}")
    print(f"Dataset: {args.data}")
    print(f"Epochs per run: {args.n_epochs}")
    print("=" * 70)

    active_processes = []
    completed_runs = []
    
    # Queue up configurations
    pending_configs = list(enumerate(configs))

    t_start = time.time()

    # Keep track of which GPU indices (0 to 7) are free
    free_gpus = list(range(8))

    while pending_configs or active_processes:
        # Launch new workers if slots are available and we have free GPUs
        while len(active_processes) < args.max_workers and pending_configs and free_gpus:
            gpu_id = free_gpus.pop(0)
            idx, config = pending_configs.pop(0)
            run_id = f"run_{idx:02d}_h{config['hidden']}_l{config['n_layers']}_lr{config['lr']}_{config['method']}"
            if config["compile"]:
                run_id += "_compiled"

            log_path = os.path.join(args.output_dir, f"{run_id}.log")
            model_path = os.path.join(args.output_dir, f"{run_id}.pt")

            # Build command
            cmd = [
                "python", "-m", "modeling.scripts.fit_canode",
                "--data", args.data,
                "--n-epochs", str(args.n_epochs),
                "--batch-size", str(args.batch_size),
                "--hidden", str(config["hidden"]),
                "--n-layers", str(config["n_layers"]),
                "--lr", str(config["lr"]),
                "--method", config["method"],
                "--device", args.device,
                "--output-dir", os.path.join(args.output_dir, run_id),
                "--save-model", model_path,
            ]
            if config["compile"]:
                cmd.append("--compile")

            # Open log file
            log_file = open(log_path, "w")
            
            # Dynamically assign GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            print(f"[{idx+1}/{len(configs)}] Launching {run_id} on GPU {gpu_id} ...")
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

        # Wait a bit and check on active processes
        time.sleep(2.0)
        
        still_active = []
        for job in active_processes:
            poll = job["process"].poll()
            if poll is None:
                # Still running
                still_active.append(job)
            else:
                # Finished!
                job["log_file"].close()
                elapsed = time.time() - job["t_launch"]
                print(f"  * Config {job['idx']+1} ({job['run_id']}) finished on GPU {job['gpu_id']} in {elapsed:.1f}s with code {poll}")
                
                # Free the GPU
                free_gpus.append(job["gpu_id"])
                
                # Parse log to get metrics
                metrics = parse_run_log(job["log_path"])
                metrics["elapsed_s"] = elapsed
                metrics["idx"] = job["idx"]
                metrics["config"] = job["config"]
                metrics["run_id"] = job["run_id"]
                completed_runs.append(metrics)
                
        active_processes = still_active

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"Sweep complete in {total_time/60:.2f} minutes!")
    print("=" * 70)

    # Sort results by validation loss
    completed_runs.sort(key=lambda r: r.get("best_val_loss", float("inf")))

    # Print summary table
    print(f"\n{'Idx':<3} {'Method':<7} {'Hidden':<6} {'Layers':<6} {'LR':<7} {'Comp':<5} {'Time (s)':<9} {'Train L':<8} {'Val L':<8} {'Test1 R2':<8} {'Test2 R2':<8}")
    print("-" * 87)
    for run in completed_runs:
        cfg = run["config"]
        time_str = f"{run['elapsed_s']:.1f}"
        train_l = f"{run.get('best_train_loss', float('nan')):.5f}"
        val_l = f"{run.get('best_val_loss', float('nan')):.5f}"
        t1_r2 = f"{run.get('test_r2_val1', float('nan')):.3f}"
        t2_r2 = f"{run.get('test_r2_val2', float('nan')):.3f}"
        print(f"{run['idx']+1:<3} {cfg['method']:<7} {cfg['hidden']:<6} {cfg['n_layers']:<6} {cfg['lr']:<7} {str(cfg['compile']):<5} {time_str:<9} {train_l:<8} {val_l:<8} {t1_r2:<8} {t2_r2:<8}")

    # Write summary json
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
    
    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            # Parse train/val losses
            # Format: Epoch   10/500 | train: 0.101428 | val: 0.106882 | lr: 9.99e-04
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
            # Parse test trial outputs
            # Format: Trial 8: MSE=1.2974, R2=0.2110
            elif "Trial" in line and "MSE=" in line and "R2=" in line:
                try:
                    parts = line.split(":")
                    metrics = parts[1].strip().split(",")
                    mse = float(metrics[0].split("=")[1].strip())
                    r2 = float(metrics[1].split("=")[1].strip())
                    test_r2s.append(r2)
                    test_mses.append(mse)
                except:
                    pass

    result = {
        "best_train_loss": best_train if best_train != float("inf") else None,
        "best_val_loss": best_val if best_val != float("inf") else None,
    }
    if len(test_r2s) >= 2:
        result["test_r2_val1"] = test_r2s[0]
        result["test_r2_val2"] = test_r2s[1]
        result["test_mse_val1"] = test_mses[0]
        result["test_mse_val2"] = test_mses[1]
        result["mean_test_r2"] = np.mean(test_r2s)
    elif len(test_r2s) == 1:
        result["test_r2_val1"] = test_r2s[0]
        result["mean_test_r2"] = test_r2s[0]
        
    return result


if __name__ == "__main__":
    main()

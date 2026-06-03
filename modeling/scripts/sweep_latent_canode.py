#!/usr/bin/env python
"""Run a parallel hyperparameter sweep for Latent CA-NODE on GPU.

Launches multiple configurations in parallel using subprocesses.
Each subprocess is assigned a dedicated GPU via CUDA_VISIBLE_DEVICES.
"""

import argparse
import os
import subprocess
import time
import json
import numpy as np
from modeling.config import DEFAULT_SEED, DEFAULT_TEST_TRIALS


def main():
    parser = argparse.ArgumentParser(description="Parallel hyperparameter sweep for Latent CA-NODE")
    parser.add_argument("--data", type=str, default="data/training_trials.h5")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-test-trials", type=int, default=None,
                        help="Number of test trials (defaults to DEFAULT_TEST_TRIALS in config)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Number of GPUs available")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated list of GPU indices to use (e.g. 1,2,3). Overrides --n-gpus")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum parallel runs (capped by active GPUs)")
    parser.add_argument("--output-dir", type=str, default="results/sweep_latent_canode")
    parser.add_argument("--grid", type=str, default="small",
                        choices=["small", "large"])
    parser.add_argument("--no-wandb", action="store_true", default=False,
                        help="Disable wandb logging for all child runs")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="wandb group name for sweep runs (auto-generated if not set)")
    args = parser.parse_args()

    # Auto-generate wandb group name for sweep
    if not args.no_wandb and args.wandb_group is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_group = f"sweep_latent_{args.grid}_{ts}"

    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.n_gpus))

    args.max_workers = min(args.max_workers, len(gpu_ids))
    os.makedirs(args.output_dir, exist_ok=True)

    if args.grid == "large":
        configs = []
        for hidden in [128, 256, 512]:
            for past_window in [50, 100, 200]:
                for lr in [1e-3, 5e-4]:
                    configs.append({
                        "hidden": hidden,
                        "n_layers": 2,
                        "z_dim": 32,
                        "past_window": past_window,
                        "lr": lr,
                        "method": "dopri5",
                    })
    else:
        configs = []
        for past_window in [50, 100, 200]:
            for hidden in [128, 256]:
                for lr in [1e-3, 5e-4]:
                    configs.append({
                        "hidden": hidden,
                        "n_layers": 2,
                        "z_dim": 32,
                        "past_window": past_window,
                        "lr": lr,
                        "method": "dopri5",
                    })

    n_configs = len(configs)
    print("=" * 70)
    print(f"Starting Latent CA-NODE Parallel Sweep (Total configurations: {n_configs})")
    print(f"Max parallel workers: {args.max_workers} across {args.n_gpus} GPUs")
    print("=" * 70)

    active_processes = []
    completed_runs = []
    pending_configs = list(enumerate(configs))
    free_gpus = list(gpu_ids)
    
    n_test_flag = []
    if args.n_test_trials is not None:
        n_test_flag = ["--n-test-trials", str(args.n_test_trials)]

    seed_flag = ["--seed", str(args.seed)]

    t_start = time.time()

    while pending_configs or active_processes:
        while len(active_processes) < args.max_workers and pending_configs and free_gpus:
            gpu_id = free_gpus.pop(0)
            idx, config = pending_configs.pop(0)
            run_id = f"run_{idx:02d}_pw{config['past_window']}_h{config['hidden']}_lr{config['lr']}_{config['method']}"
            
            log_path = os.path.join(args.output_dir, f"{run_id}.log")
            model_path = os.path.join(args.output_dir, f"{run_id}.pt")

            cmd = [
                "python", "-m", "modeling.scripts.fit_latent_canode",
                "--data", args.data,
                "--n-epochs", str(args.n_epochs),
                "--batch-size", str(args.batch_size),
                "--z-dim", str(config["z_dim"]),
                "--hidden", str(config["hidden"]),
                "--n-layers", str(config["n_layers"]),
                "--past-window", str(config["past_window"]),
                "--lr", str(config["lr"]),
                "--method", config["method"],
                "--device", args.device,
                "--output-dir", os.path.join(args.output_dir, run_id),
                "--save-model", model_path,
            ] + n_test_flag + seed_flag

            if not args.no_wandb:
                cmd += ["--wandb-group", args.wandb_group, "--wandb-name", run_id]
            else:
                cmd += ["--no-wandb"]

            log_file = open(log_path, "w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            print(f"[{idx+1}/{n_configs}] Launching {run_id} on GPU {gpu_id} ...")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, text=True)
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

                metrics = parse_run_log(job["log_path"])
                metrics["elapsed_s"] = elapsed
                metrics["idx"] = job["idx"]
                metrics["config"] = job["config"]
                metrics["run_id"] = job["run_id"]
                metrics["gpu_id"] = job["gpu_id"]
                completed_runs.append(metrics)

        active_processes = still_active

    print(f"\nSweep complete in {(time.time() - t_start)/60:.2f} minutes!")

    completed_runs.sort(key=lambda r: -r.get("mean_windowed_r2", -999))
    print(f"\n{'Idx':<4} {'Method':<7} {'PW':<4} {'Hidden':<6} {'LR':<7} "
          f"{'GPU':<4} {'Time':<8} {'Train L':<9} {'Val L':<9} {'Win R2':<8} {'Win MSE':<8} {'Inf Time':<8}")
    print("-" * 105)
    for run in completed_runs:
        cfg = run["config"]
        time_str = f"{run['elapsed_s']/60:.1f}m"
        train_l = f"{run.get('best_train_loss', float('nan')):.5f}"
        val_l = f"{run.get('best_val_loss', float('nan')):.5f}"
        win_r2 = f"{run.get('mean_windowed_r2', float('nan')):.4f}"
        win_mse = f"{run.get('mean_windowed_mse', float('nan')):.4f}"
        inf_time = f"{run.get('inference_ms', float('nan')):.2f}ms"
        print(f"{run['idx']+1:<4} {cfg['method']:<7} {cfg['past_window']:<4} {cfg['hidden']:<6} "
              f"{cfg['lr']:<7} {run.get('gpu_id','?'):<4} {time_str:<8} {train_l:<9} {val_l:<9} {win_r2:<8} {win_mse:<8} {inf_time:<8}")

    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(completed_runs, f, indent=2)


def parse_run_log(log_path):
    best_train = float("inf")
    best_val = float("inf")
    windowed_r2s = []
    windowed_mses = []
    inference_ms = None

    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if "train:" in line and "val:" in line:
                try:
                    parts = line.split("|")
                    train_val = parts[1].strip().split(":")[1].split("(")[0].strip()
                    val_val = parts[2].strip().split(":")[1].strip()
                    train_loss = float(train_val)
                    val_loss = float(val_val)
                    best_train = min(best_train, train_loss)
                    best_val = min(best_val, val_loss)
                except:
                    pass
            elif "MSE=" in line and "R2=" in line and "Avg 200-step" not in line:
                try:
                    parts = line.split(":")
                    metrics = parts[1].strip().split(",")
                    mse = float(metrics[0].split("=")[1].strip())
                    r2 = float(metrics[1].split("=")[1].strip())
                    windowed_mses.append(mse)
                    windowed_r2s.append(r2)
                except:
                    pass
            elif "Avg 200-step" in line or "Avg " in line and "-step: MSE" in line:
                try:
                    mse_part = line.split("MSE=")[1].split(",")[0]
                    r2_part = line.split("R2=")[1].strip()
                    windowed_r2s = [float(r2_part)]
                    windowed_mses = [float(mse_part)]
                except:
                    pass
            elif "Avg Inference Time" in line:
                try:
                    parts = line.split(":")
                    ms_val = float(parts[1].replace("ms", "").strip())
                    inference_ms = ms_val
                except:
                    pass

    result = {
        "best_train_loss": best_train if best_train != float("inf") else None,
        "best_val_loss": best_val if best_val != float("inf") else None,
    }
    if windowed_r2s:
        result["mean_windowed_r2"] = float(np.mean(windowed_r2s))
        result["mean_windowed_mse"] = float(np.mean(windowed_mses))
    if inference_ms is not None:
        result["inference_ms"] = inference_ms
    return result


if __name__ == "__main__":
    main()

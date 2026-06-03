#!/usr/bin/env python
"""Run a parallel hyperparameter sweep and architecture search for CA-NODE on GPU.

Launches multiple configurations in parallel using subprocesses,
monitors their execution, logs outputs, and summarizes results in a table.

Each subprocess is assigned a dedicated GPU via CUDA_VISIBLE_DEVICES.
When a GPU finishes, it is returned to the pool and the next queued
configuration is launched on it.
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
    parser.add_argument("--n-test-trials", type=int, default=None,
                        help="Number of test trials (default: 2 for <=10 trials, 10 for >10)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Number of GPU cards available (0..n_gpus-1)")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated list of GPU indices to use (e.g., '1,2,3,4,5,6'). Overrides --n-gpus.")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Maximum parallel runs (capped by active GPUs)")
    parser.add_argument("--output-dir", type=str, default="results/sweep")
    parser.add_argument("--grid", type=str, default="small",
                        choices=["small", "large", "skip_linear", "spectral", "multirate"],
                        help="Sweep grid size: 'small' (8 configs), 'large' (27 configs), 'skip_linear' (16 configs), 'spectral' (8 configs), or 'multirate' (6 configs)")
    parser.add_argument("--no-wandb", action="store_true", default=False,
                        help="Disable wandb logging for all child runs")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="wandb group name for sweep runs (auto-generated if not set)")
    args = parser.parse_args()

    # Auto-generate wandb group name for sweep
    if not args.no_wandb and args.wandb_group is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_group = f"sweep_{args.grid}_{ts}"

    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.n_gpus))

    # Cap workers at available GPUs
    args.max_workers = min(args.max_workers, len(gpu_ids))

    os.makedirs(args.output_dir, exist_ok=True)

    # Define the sweep configurations
    if args.grid == "large":
        # Full 3×3×3 grid: hidden × layers × lr
        configs = []
        for hidden in [128, 256, 512]:
            for n_layers in [2, 3, 4]:
                for lr in [5e-4, 2e-4, 1e-4]:
                    configs.append({
                        "hidden": hidden,
                        "n_layers": n_layers,
                        "lr": lr,
                        "method": "dopri5",
                        "compile": True,
                    })
    elif args.grid == "skip_linear":
        # Sweeps over skip types and skip weight decays to prevent overfitting
        configs = []
        for lr in [1e-4, 5e-4]:
            # 1. No skip baseline
            configs.append({
                "hidden": 128,
                "n_layers": 2,
                "lr": lr,
                "method": "dopri5",
                "compile": True,
                "skip": False,
            })
            # 2. MLP skip baseline (Phase 2.5 style)
            configs.append({
                "hidden": 128,
                "n_layers": 2,
                "lr": lr,
                "method": "dopri5",
                "compile": True,
                "skip": True,
                "skip_type": "mlp",
                "skip_weight_decay": None,
            })
            # 3. Linear skip with varying weight decay
            for swd in [None, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                configs.append({
                    "hidden": 128,
                    "n_layers": 2,
                    "lr": lr,
                    "method": "dopri5",
                    "compile": True,
                    "skip": True,
                    "skip_type": "linear",
                    "skip_weight_decay": swd,
                })
    elif args.grid == "spectral":
        # Sweeps over spectral_alpha values to find the optimal high-frequency balance
        configs = []
        for sa in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            configs.append({
                "hidden": 128,
                "n_layers": 2,
                "lr": 1e-4,
                "method": "dopri5",
                "compile": True,
                "skip": False,
                "spectral_alpha": sa,
            })
    elif args.grid == "multirate":
        # Sweeps over sub-steps and initial split types for MultiRate ODE
        configs = []
        for m_steps in [1, 5, 10]:
            for init_type in ["zero_fast", "learnable"]:
                configs.append({
                    "hidden": 128,
                    "n_layers": 2,
                    "lr": 1e-4,
                    "method": "dopri5",
                    "compile": True,
                    "model_type": "multi-rate",
                    "sub_steps": m_steps,
                    "multirate_init": init_type,
                })
    else:
        # Small grid (Phase 1 style)
        configs = [
            {"hidden": 128, "n_layers": 2, "lr": 1e-3, "method": "dopri5", "compile": False},
            {"hidden": 128, "n_layers": 2, "lr": 1e-3, "method": "dopri5", "compile": True},
            {"hidden": 256, "n_layers": 2, "lr": 1e-3, "method": "dopri5", "compile": True},
            {"hidden": 256, "n_layers": 3, "lr": 1e-3, "method": "dopri5", "compile": True},
            {"hidden": 128, "n_layers": 2, "lr": 5e-4, "method": "dopri5", "compile": True},
            {"hidden": 256, "n_layers": 2, "lr": 5e-4, "method": "dopri5", "compile": True},
            {"hidden": 256, "n_layers": 3, "lr": 5e-4, "method": "dopri5", "compile": True},
            {"hidden": 128, "n_layers": 2, "lr": 1e-3, "method": "rk4", "compile": True},
        ]

    n_configs = len(configs)
    print("=" * 70)
    print(f"Starting CA-NODE Parallel Sweep (Total configurations: {n_configs})")
    print(f"Max parallel workers: {args.max_workers} across GPUs {gpu_ids}")
    print(f"Dataset: {args.data}")
    print(f"Epochs per run: {args.n_epochs}")
    print(f"Grid: {args.grid}")
    print("=" * 70)

    active_processes = []
    completed_runs = []

    # Queue up configurations
    pending_configs = list(enumerate(configs))

    # Keep track of which GPU indices are free
    free_gpus = list(gpu_ids)

    # Build n_test_trials argument
    n_test_flag = []
    if args.n_test_trials is not None:
        n_test_flag = ["--n-test-trials", str(args.n_test_trials)]

    t_start = time.time()

    while pending_configs or active_processes:
        # Launch new workers if slots are available and we have free GPUs
        while len(active_processes) < args.max_workers and pending_configs and free_gpus:
            gpu_id = free_gpus.pop(0)
            idx, config = pending_configs.pop(0)
            run_id = f"run_{idx:02d}_h{config['hidden']}_l{config['n_layers']}_lr{config['lr']}_{config['method']}"
            mtype = config.get("model_type", "standard")
            if mtype == "multi-rate":
                run_id += f"_mr_m{config.get('sub_steps', 10)}_init_{config.get('multirate_init', 'zero_fast')}"
            else:
                if "spectral_alpha" in config:
                    run_id += f"_sa{config['spectral_alpha']}"
                if config.get("skip", True):
                    stype = config.get("skip_type", "mlp")
                    swd = config.get("skip_weight_decay", None)
                    run_id += f"_skip_{stype}"
                    if swd is not None:
                        run_id += f"_swd{swd}"
                else:
                    run_id += "_noskip"
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
            ] + n_test_flag
            # Forward wandb arguments to child process
            if not args.no_wandb:
                cmd += ["--wandb-group", args.wandb_group]
                cmd += ["--wandb-name", run_id]
            else:
                cmd.append("--no-wandb")
            if config["compile"]:
                cmd.append("--compile")
            if "skip" in config:
                if config["skip"]:
                    cmd.append("--skip")
                else:
                    cmd.append("--no-skip")
            if "skip_type" in config:
                cmd += ["--skip-type", config["skip_type"]]
            if "skip_weight_decay" in config and config["skip_weight_decay"] is not None:
                cmd += ["--skip-weight-decay", str(config["skip_weight_decay"])]
            if "spectral_alpha" in config:
                cmd += ["--spectral-alpha", str(config["spectral_alpha"])]
            if "model_type" in config:
                cmd += ["--model-type", config["model_type"]]
            if "sub_steps" in config:
                cmd += ["--sub-steps", str(config["sub_steps"])]
            if "multirate_init" in config:
                cmd += ["--multirate-init", config["multirate_init"]]

            # Open log file
            log_file = open(log_path, "w")

            # Dynamically assign GPU
            env = os.environ.copy()
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
                metrics["gpu_id"] = job["gpu_id"]
                completed_runs.append(metrics)

        active_processes = still_active

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"Sweep complete in {total_time/60:.2f} minutes!")
    print("=" * 70)

    # Sort results by windowed R² (descending), fall back to val loss
    completed_runs.sort(key=lambda r: -r.get("mean_windowed_r2", -999))

    # Print summary table
    if args.grid == "multirate":
        print(f"\n{'Idx':<4} {'Hidden':<6} {'Layers':<5} {'LR':<7} {'Type':<10} {'Steps':<5} {'Init':<10} "
              f"{'GPU':<4} {'Time':<6} {'Train L':<8} {'Val L':<8} "
              f"{'Win R2':<8} {'Win MSE':<8} {'OL R2':<8}")
        print("-" * 105)
        for run in completed_runs:
            cfg = run["config"]
            time_str = f"{run['elapsed_s']/60:.1f}m"
            train_l_val = run.get('best_train_loss')
            train_l = f"{train_l_val:.4f}" if train_l_val is not None else "nan"
            val_l_val = run.get('best_val_loss')
            val_l = f"{val_l_val:.4f}" if val_l_val is not None else "nan"
            win_r2_val = run.get('mean_windowed_r2')
            win_r2 = f"{win_r2_val:.4f}" if win_r2_val is not None else "nan"
            win_mse_val = run.get('mean_windowed_mse')
            win_mse = f"{win_mse_val:.4f}" if win_mse_val is not None else "nan"
            ol_r2_val = run.get('mean_test_r2')
            ol_r2 = f"{ol_r2_val:.4f}" if ol_r2_val is not None else "nan"
            print(f"{run['idx']+1:<4} {cfg['hidden']:<6} {cfg['n_layers']:<5} "
                  f"{cfg['lr']:<7} {cfg.get('model_type', 'mr'):<10} {cfg.get('sub_steps', 10):<5} {cfg.get('multirate_init', 'zero'):<10} {run.get('gpu_id','?'):<4} "
                  f"{time_str:<6} {train_l:<8} {val_l:<8} {win_r2:<8} {win_mse:<8} {ol_r2:<8}")
    else:
        print(f"\n{'Idx':<4} {'Hidden':<6} {'Layers':<5} {'LR':<7} {'Skip':<8} {'SWD':<6} {'SA':<5} "
              f"{'GPU':<4} {'Time':<6} {'Train L':<8} {'Val L':<8} "
              f"{'Win R2':<8} {'Win MSE':<8} {'OL R2':<8}")
        print("-" * 102)
        for run in completed_runs:
            cfg = run["config"]
            time_str = f"{run['elapsed_s']/60:.1f}m"
            train_l_val = run.get('best_train_loss')
            train_l = f"{train_l_val:.4f}" if train_l_val is not None else "nan"
            val_l_val = run.get('best_val_loss')
            val_l = f"{val_l_val:.4f}" if val_l_val is not None else "nan"
            win_r2_val = run.get('mean_windowed_r2')
            win_r2 = f"{win_r2_val:.4f}" if win_r2_val is not None else "nan"
            win_mse_val = run.get('mean_windowed_mse')
            win_mse = f"{win_mse_val:.4f}" if win_mse_val is not None else "nan"
            ol_r2_val = run.get('mean_test_r2')
            ol_r2 = f"{ol_r2_val:.4f}" if ol_r2_val is not None else "nan"
            skip_str = cfg.get("skip_type", "mlp") if cfg.get("skip", True) else "none"
            swd_val = cfg.get("skip_weight_decay", None)
            swd_str = "none" if swd_val is None else f"{swd_val:.0e}"
            sa_val = cfg.get("spectral_alpha", 0.0)
            print(f"{run['idx']+1:<4} {cfg['hidden']:<6} {cfg['n_layers']:<5} "
                  f"{cfg['lr']:<7} {skip_str:<8} {swd_str:<6} {sa_val:<5} {run.get('gpu_id','?'):<4} "
                  f"{time_str:<6} {train_l:<8} {val_l:<8} {win_r2:<8} {win_mse:<8} {ol_r2:<8}")

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
    windowed_r2s = []
    windowed_mses = []

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
                    train_loss = float(train_val[1].strip().split()[0])
                    val_loss = float(val_val[1].strip().split()[0])
                    if train_loss < best_train:
                        best_train = train_loss
                    if val_loss < best_val:
                        best_val = val_loss
                except:
                    pass
            # Parse 2000-step open-loop test metrics
            # Format: Trial 8: MSE=1.2974, R2=0.2110
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
            # Parse 200-step windowed metrics
            # Format: Trial 8: 200-step window MSE=0.1739, R_win=0.8681
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
            # Parse avg windowed line
            # Format: Avg 200-step: MSE=0.1755, R2=0.8742
            elif "Avg 200-step" in line:
                try:
                    mse_part = line.split("MSE=")[1].split(",")[0]
                    r2_part = line.split("R2=")[1].strip()
                    windowed_r2s = []  # Replace per-trial with avg
                    windowed_mses = []
                    windowed_r2s.append(float(r2_part))
                    windowed_mses.append(float(mse_part))
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
        result["mean_test_r2"] = float(np.mean(test_r2s))
    elif len(test_r2s) == 1:
        result["test_r2_val1"] = test_r2s[0]
        result["mean_test_r2"] = test_r2s[0]

    if windowed_r2s:
        result["mean_windowed_r2"] = float(np.mean(windowed_r2s))
        result["mean_windowed_mse"] = float(np.mean(windowed_mses))

    return result


if __name__ == "__main__":
    main()

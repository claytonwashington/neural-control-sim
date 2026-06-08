#!/usr/bin/env python
"""Parallel sweep for Aligned Distillation on bidirectional plant.

Launches 8 configs across GPUs 0-7 on gpu1.
"""
import argparse
import os
import subprocess
import time
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modeling.scripts.preflight_check import add_preflight_args, validate_preflight


CONFIGS = [
    {"alpha_init": 0.9, "past_window": 100, "lr": 5e-4, "hidden": 256},
    {"alpha_init": 0.9, "past_window": 200, "lr": 5e-4, "hidden": 256},
    {"alpha_init": 0.7, "past_window": 100, "lr": 5e-4, "hidden": 256},
    {"alpha_init": 0.7, "past_window": 200, "lr": 5e-4, "hidden": 256},
    {"alpha_init": 0.9, "past_window": 100, "lr": 1e-3, "hidden": 256},
    {"alpha_init": 0.9, "past_window": 200, "lr": 1e-3, "hidden": 256},
    {"alpha_init": 0.5, "past_window": 200, "lr": 5e-4, "hidden": 256},
    {"alpha_init": 0.9, "past_window":  50, "lr": 5e-4, "hidden": 256},
]


def main():
    parser = argparse.ArgumentParser(description="Aligned Distillation sweep")
    parser.add_argument("--data", required=True, help="Training data file")
    parser.add_argument("--acausal-checkpoint", required=True, help="Acausal teacher .pt")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-test-trials", type=int, default=10)
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--device", default="cuda")
    add_preflight_args(parser)
    args = parser.parse_args()
    manifest = validate_preflight(args)
    _token = getattr(args, "preflight_token", None) or os.environ.get("PREFLIGHT_TOKEN", "")

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    n_gpus = len(gpu_ids)
    os.makedirs(args.output_dir, exist_ok=True)

    processes = {}
    results = {}
    t_start = time.time()

    for idx, config in enumerate(CONFIGS):
        gpu_id = gpu_ids[idx % n_gpus]
        while gpu_id in processes:
            # Wait for a GPU to free up
            for gid, (proc, log_f, rid, t0) in list(processes.items()):
                if proc.poll() is not None:
                    log_f.close()
                    elapsed = time.time() - t0
                    print(f"  * Config {rid} finished on GPU {gid} in {elapsed:.1f}s")
                    results[rid] = elapsed
                    del processes[gid]
            if gpu_id in processes:
                time.sleep(5)

        run_id = f"run_{idx:02d}_a{config['alpha_init']}_pw{config['past_window']}_lr{config['lr']}"
        run_dir = os.path.join(args.output_dir, run_id)
        log_path = os.path.join(args.output_dir, f"{run_id}.log")

        cmd = [
            "python", "-m", "modeling.scripts.train_aligned_distill",
            "--data", args.data,
            "--acausal-checkpoint", args.acausal_checkpoint,
            "--n-epochs", str(args.n_epochs),
            "--alpha-schedule", "cosine",
            "--alpha-init", str(config["alpha_init"]),
            "--alpha-final", "0.1",
            "--hidden", str(config["hidden"]),
            "--past-window", str(config["past_window"]),
            "--lr", str(config["lr"]),
            "--device", args.device,
            "--output-dir", run_dir,
            "--n-test-trials", str(args.n_test_trials),
        ]

        env = os.environ.copy()
        env["PREFLIGHT_TOKEN"] = _token
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_f = open(log_path, "w")
        print(f"[{idx+1}/{len(CONFIGS)}] Launching {run_id} on GPU {gpu_id} ...")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, text=True)
        processes[gpu_id] = (proc, log_f, run_id, time.time())

    # Wait for remaining
    while processes:
        for gid, (proc, log_f, rid, t0) in list(processes.items()):
            if proc.poll() is not None:
                log_f.close()
                elapsed = time.time() - t0
                print(f"  * Config {rid} finished on GPU {gid} in {elapsed:.1f}s")
                results[rid] = elapsed
                del processes[gid]
        if processes:
            time.sleep(10)

    total = time.time() - t_start
    print(f"\nSweep complete in {total/60:.1f} minutes!")

    # Parse results
    print(f"\n{'Idx':<4} {'Alpha':<6} {'PW':<4} {'LR':<8} {'GPU':<4} {'Time':<9} {'Train L':<10} {'Val L':<10} {'R2':<8}")
    print("-" * 80)
    for idx, config in enumerate(CONFIGS):
        run_id = f"run_{idx:02d}_a{config['alpha_init']}_pw{config['past_window']}_lr{config['lr']}"
        run_dir = os.path.join(args.output_dir, run_id)
        results_file = os.path.join(run_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                r = json.load(f)
            print(f"{idx:<4} {config['alpha_init']:<6} {config['past_window']:<4} {config['lr']:<8} "
                  f"{gpu_ids[idx % n_gpus]:<4} "
                  f"{results.get(run_id, 0)/60:.1f}m    "
                  f"{r.get('best_train_loss', '?'):<10.5f} "
                  f"{r.get('best_val_loss', '?'):<10.5f} "
                  f"{r.get('r2_200step', r.get('val_r2', '?'))}")
        else:
            print(f"{idx:<4} {config['alpha_init']:<6} {config['past_window']:<4} — RESULTS NOT FOUND")

    # Save summary
    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump({"configs": CONFIGS, "timing": results}, f, indent=2)


if __name__ == "__main__":
    main()

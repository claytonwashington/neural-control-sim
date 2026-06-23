#!/usr/bin/env python3
"""Sweep MPC tuning parameters (mpc_iters × lambda_u) for optoclamp.

Runs multiple optoclamp experiments with different MPC solver settings
to diagnose whether MPC underperformance is due to under-convergence
or over-penalization of control effort.

Usage:
    python -m modeling.scripts.sweep_mpc_tuning \
        --model-checkpoint results/bidir_v2_aligned_distill/run_05_a0.9_pw200_lr0.001/best_model.pt \
        --output-dir results/bidir_v2_mpc_tuning \
        --gpu-ids 0,1,2,3,4,5,6,7
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Sweep MPC tuning params")
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--mpc-horizon", type=int, default=50,
                        help="Fixed horizon (use best from prior sweep)")
    parser.add_argument("--reencode-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mpc-batch-size", type=int, default=1,
                        help="Number of parallel MPC candidates (default: 1)")
    parser.add_argument("--mpc-solver", type=str, default="euler",
                        choices=["euler", "dopri5"],
                        help="ODE solver for MPC rollout")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable online adaptive fine-tuning")
    parser.add_argument("--adapt-lr", type=float, default=1e-5)
    parser.add_argument("--adapt-params", type=str, default="g+decoder")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Sweep grid: mpc_iters × lambda_u
    configs = []
    for iters, lam in itertools.product(
        [30, 100, 200, 500],      # mpc_iters (default was 30)
        [0.001, 0.01, 0.1, 1.0],  # lambda_u (default was 0.01)
    ):
        configs.append({"mpc_iters": iters, "lambda_u": lam})

    print("=" * 70)
    print(f"MPC Tuning Sweep: {len(configs)} configs across {len(gpu_ids)} GPUs")
    print(f"  Horizon: {args.mpc_horizon}, K: {args.reencode_k}")
    print("=" * 70)
    for i, cfg in enumerate(configs):
        print(f"  [{i:2d}] iters={cfg['mpc_iters']:>3d}, lambda_u={cfg['lambda_u']}")
    print()

    processes = {}
    results = {}
    t_start = time.time()

    for idx, config in enumerate(configs):
        gpu_id = gpu_ids[idx % len(gpu_ids)]

        # Wait for GPU to free up
        while gpu_id in processes:
            for gid, (proc, log_f, rid, t0) in list(processes.items()):
                if proc.poll() is not None:
                    log_f.close()
                    elapsed = time.time() - t0
                    print(f"  ✓ {rid} finished on GPU {gid} in {elapsed:.1f}s")
                    results[rid] = elapsed
                    del processes[gid]
            if gpu_id in processes:
                time.sleep(5)

        run_id = f"iters{config['mpc_iters']}_lam{config['lambda_u']}"
        run_dir = os.path.join(args.output_dir, run_id)
        log_path = os.path.join(args.output_dir, f"{run_id}.log")

        cmd = [
            sys.executable, "-m", "modeling.scripts.run_optoclamp",
            "--model-checkpoint", args.model_checkpoint,
            "--output-dir", run_dir,
            "--device", args.device,
            "--mpc-horizons", str(args.mpc_horizon),
            "--reencode-k", str(args.reencode_k),
            "--mpc-iters", str(config["mpc_iters"]),
            "--lambda-u", str(config["lambda_u"]),
            "--seed", str(args.seed),
            "--mpc-batch-size", str(args.mpc_batch_size),
            "--mpc-solver", args.mpc_solver,
        ]
        if args.adaptive:
            cmd.extend(["--adaptive",
                        "--adapt-lr", str(args.adapt_lr),
                        "--adapt-params", args.adapt_params])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Pass through preflight token if set
        if os.environ.get("PREFLIGHT_TOKEN"):
            env["PREFLIGHT_TOKEN"] = os.environ["PREFLIGHT_TOKEN"]

        log_f = open(log_path, "w")
        print(f"[{idx+1}/{len(configs)}] Launching {run_id} on GPU {gpu_id} ...")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                env=env, text=True)
        processes[gpu_id] = (proc, log_f, run_id, time.time())

    # Wait for remaining
    while processes:
        for gid, (proc, log_f, rid, t0) in list(processes.items()):
            if proc.poll() is not None:
                log_f.close()
                elapsed = time.time() - t0
                print(f"  ✓ {rid} finished on GPU {gid} in {elapsed:.1f}s")
                results[rid] = elapsed
                del processes[gid]
        if processes:
            time.sleep(5)

    total = time.time() - t_start
    print(f"\nAll {len(configs)} configs complete in {total:.0f}s ({total/60:.1f} min)")

    # Aggregate results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'iters':>6} {'lambda_u':>10} {'50% RMSE':>10} {'75% RMSE':>10} "
          f"{'125% RMSE':>10} {'75% Settle':>10}")
    print("-" * 70)

    summary = []
    for config in configs:
        run_id = f"iters{config['mpc_iters']}_lam{config['lambda_u']}"
        run_dir = os.path.join(args.output_dir, run_id)
        results_file = os.path.join(run_dir, "optoclamp_results.json")

        if os.path.exists(results_file):
            with open(results_file) as f:
                data = json.load(f)

            # Find NODE_MPC results (should be only one horizon)
            mpc_key = [k for k in data if k.startswith("NODE_MPC")][0] if any(
                k.startswith("NODE_MPC") for k in data) else None

            if mpc_key:
                r = data[mpc_key]
                rmse_50 = r.get("50pct", {}).get("metrics", {}).get("tracking_rmse", float("nan"))
                rmse_75 = r.get("75pct", {}).get("metrics", {}).get("tracking_rmse", float("nan"))
                rmse_125 = r.get("125pct", {}).get("metrics", {}).get("tracking_rmse", float("nan"))
                settle_75 = r.get("75pct", {}).get("metrics", {}).get("settling_time_ms", float("inf"))

                print(f"{config['mpc_iters']:>6} {config['lambda_u']:>10.3f} "
                      f"{rmse_50:>10.2f} {rmse_75:>10.2f} "
                      f"{rmse_125:>10.2f} {settle_75:>10.0f}")

                summary.append({
                    "mpc_iters": config["mpc_iters"],
                    "lambda_u": config["lambda_u"],
                    "rmse_50": rmse_50,
                    "rmse_75": rmse_75,
                    "rmse_125": rmse_125,
                    "settling_75": settle_75,
                })
        else:
            print(f"{config['mpc_iters']:>6} {config['lambda_u']:>10.3f}   MISSING")

    # Save summary
    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Find best
    if summary:
        best = min(summary, key=lambda x: x["rmse_75"])
        print(f"\nBest (by 75% RMSE): iters={best['mpc_iters']}, "
              f"lambda_u={best['lambda_u']}, RMSE={best['rmse_75']:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the optoclamp rate-clamping experiment with multiple controllers.

Recapitulates the optoclamp experiment (Newman et al. 2015):
  1. Baseline: 200ms no control → measure spontaneous rate
  2. Clamp: 1000ms → hold population rate at target (fraction of baseline)
  3. Recovery: 200ms no control

Controllers compared:
  - PI (proportional-integral, no model)
  - Neural ODE MPC (model-predictive control with learned dynamics)

Usage:
    python -m modeling.scripts.run_optoclamp \
        --model-checkpoint results/causal_z64_pw200_h128_lr3e4/model.pt \
        --output-dir results/optoclamp \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brian2 import ms, Hz, mwatt, mm

from cleo.ioproc import LatencyIOProcessor, exp_firing_rate_estimate

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modeling.plant import build_plant


# ─── Recording-only IOProcessor for baseline measurement ───────────────────

class BaselineRecorder(LatencyIOProcessor):
    """Record firing rates without any stimulation."""

    def __init__(self, sample_period_ms=1.0, tau_rate_ms=20.0,
                 probe_name="probe", mua_name="mua", light_name="fibers"):
        super().__init__(sample_period=sample_period_ms * ms)
        self.tau_rate = tau_rate_ms * ms
        self.probe_name = probe_name
        self.mua_name = mua_name
        self.light_name = light_name
        self._rates = None
        self.rate_log = []

    def process(self, state_dict, t_samp):
        i_chan, t_spikes, counts = state_dict[self.probe_name][self.mua_name]
        counts = np.asarray(counts, dtype=float)
        if self._rates is None:
            self._rates = np.zeros(len(counts)) * Hz
        self._rates = exp_firing_rate_estimate(
            counts, self.sample_period, self._rates, self.tau_rate
        )
        self.rate_log.append(np.array(self._rates, dtype=np.float64))
        # Zero stimulation
        return {self.light_name: np.zeros(2) * mwatt / mm**2}, t_samp

    def _base_reset(self):
        super()._base_reset()
        self._rates = None
        self.rate_log.clear()


# ─── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(rates: np.ndarray, target: float,
                    clamp_start: int, clamp_end: int, dt_ms: float = 1.0):
    """Compute control performance metrics.

    Parameters
    ----------
    rates : ndarray, shape (n_steps, n_channels)
        Firing rate time series
    target : float
        Target population mean rate in Hz
    clamp_start, clamp_end : int
        Step indices of the clamping period
    dt_ms : float
        Timestep in ms

    Returns
    -------
    dict with tracking_rmse, settling_time_ms, steady_state_error,
    overshoot, control_quality (1 - RMSE/baseline_std)
    """
    clamp_rates = rates[clamp_start:clamp_end]  # (n_steps, n_channels)
    mean_rates = clamp_rates.mean(axis=1)  # population mean

    # Tracking RMSE
    tracking_rmse = np.sqrt(np.mean((mean_rates - target) ** 2))

    # Settling time: first time rate enters and stays within 10% of target
    tolerance = 0.1 * abs(target) if target != 0 else 1.0
    in_band = np.abs(mean_rates - target) < tolerance
    settling_idx = None
    for i in range(len(in_band)):
        if np.all(in_band[i : min(i + 50, len(in_band))]):  # stay 50ms
            settling_idx = i
            break
    settling_time_ms = settling_idx * dt_ms if settling_idx is not None else float("inf")

    # Steady-state error (last 200ms of clamp)
    ss_rates = mean_rates[-200:]
    steady_state_error = np.mean(np.abs(ss_rates - target))

    # Overshoot
    if target > mean_rates[0]:
        overshoot = max(0, np.max(mean_rates) - target)
    else:
        overshoot = max(0, target - np.min(mean_rates))

    # Baseline variability
    baseline_std = np.std(rates[:clamp_start].mean(axis=1))

    return {
        "tracking_rmse": float(tracking_rmse),
        "settling_time_ms": float(settling_time_ms),
        "steady_state_error": float(steady_state_error),
        "overshoot": float(overshoot),
        "baseline_std": float(baseline_std),
    }


# ─── Main experiment ──────────────────────────────────────────────────────

def run_optoclamp(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 0: Measure baseline spontaneous rate ──────────────────────
    print("=" * 60)
    print("Phase 0: Measuring baseline spontaneous rate")
    print("=" * 60)

    sim, devices = build_plant(seed=args.seed)
    recorder = BaselineRecorder(sample_period_ms=1.0)
    sim.set_io_processor(recorder)
    sim.run(1000 * ms)  # 1 second baseline

    baseline_rates = np.array(recorder.rate_log)  # (1000, n_channels)
    mean_baseline = baseline_rates.mean(axis=1).mean()
    print(f"  Baseline population mean rate: {mean_baseline:.1f} Hz")

    # Target levels: fractions of baseline
    target_fractions = [0.5, 0.75, 1.25]
    targets = [mean_baseline * f for f in target_fractions]
    print(f"  Target rates: {[f'{t:.1f} Hz ({f*100:.0f}%)' for t, f in zip(targets, target_fractions)]}")

    all_results = {}
    # Per (controller, target) time series, persisted so the (expensive) closed-loop
    # run never has to be repeated just to re-plot. mean_rate: (n_steps,);
    # u: (n_steps, n_u) optogenetic input.
    trajectories: dict[tuple[str, str], dict] = {}

    # ── Phase 1: PI Controller ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 1: PI Controller")
    print("=" * 60)

    # Sweep PI gains
    pi_configs = [
        {"Kp": 0.1, "Ki": 0.001},
        {"Kp": 0.5, "Ki": 0.005},
        {"Kp": 1.0, "Ki": 0.01},
        {"Kp": 2.0, "Ki": 0.02},
    ]

    from modeling.controllers.pi_controller import PIController

    best_pi = None
    best_pi_rmse = float("inf")

    for pi_cfg in pi_configs:
        print(f"\n  PI: Kp={pi_cfg['Kp']}, Ki={pi_cfg['Ki']}")
        target = targets[1]  # 75% for tuning

        sim, devices = build_plant(seed=args.seed)
        ctrl = PIController(
            target_rate=target,
            Kp=pi_cfg["Kp"],
            Ki=pi_cfg["Ki"],
            n_u=2,
            sample_period_ms=1.0,
            warmup_steps=200,
        )
        sim.set_io_processor(ctrl)

        # Baseline → Clamp → Recovery
        sim.run(200 * ms)   # baseline (warmup)
        sim.run(1000 * ms)  # clamping
        sim.run(200 * ms)   # recovery

        rates = np.array(ctrl.rate_log)
        metrics = compute_metrics(rates, target, clamp_start=200, clamp_end=1200)
        print(f"    RMSE={metrics['tracking_rmse']:.2f}, "
              f"Settling={metrics['settling_time_ms']:.0f}ms, "
              f"SS_err={metrics['steady_state_error']:.2f}")

        if metrics["tracking_rmse"] < best_pi_rmse:
            best_pi_rmse = metrics["tracking_rmse"]
            best_pi = pi_cfg

    print(f"\n  Best PI: Kp={best_pi['Kp']}, Ki={best_pi['Ki']}, RMSE={best_pi_rmse:.2f}")

    # Run best PI on all targets
    pi_results = {}
    for frac, target in zip(target_fractions, targets):
        print(f"\n  PI @ {frac*100:.0f}% target ({target:.1f} Hz)...")
        sim, devices = build_plant(seed=args.seed)
        ctrl = PIController(
            target_rate=target,
            Kp=best_pi["Kp"],
            Ki=best_pi["Ki"],
            n_u=2,
            warmup_steps=200,
        )
        sim.set_io_processor(ctrl)
        sim.run(200 * ms)
        sim.run(1000 * ms)
        sim.run(200 * ms)

        results = ctrl.get_results()
        metrics = compute_metrics(results["rates"], target, 200, 1200)
        tgt_key = f"{int(frac*100)}pct"
        pi_results[tgt_key] = {
            "target": target,
            "metrics": metrics,
        }
        trajectories[("PI", tgt_key)] = _extract_trajectory(results, target)
        print(f"    RMSE={metrics['tracking_rmse']:.2f}, "
              f"Settling={metrics['settling_time_ms']:.0f}ms")

    all_results["PI"] = {
        "config": best_pi,
        "targets": pi_results,
    }

    # ── Phase 2: Neural ODE MPC ─────────────────────────────────────────
    if args.model_checkpoint:
        print("\n" + "=" * 60)
        print("Phase 2: Neural ODE MPC")
        print("=" * 60)

        from modeling.controllers.node_mpc import NeuralODEMPC

        mpc_results = {}
        for frac, target in zip(target_fractions, targets):
            print(f"\n  MPC @ {frac*100:.0f}% target ({target:.1f} Hz)...")

            # Build target rate vector (broadcast to all channels)
            target_rate_vec = np.full(50, target)

            sim, devices = build_plant(seed=args.seed)
            ctrl = NeuralODEMPC(
                checkpoint_path=args.model_checkpoint,
                target_rate=target_rate_vec,
                sample_period_ms=1.0,
                mpc_horizon=args.mpc_horizon,
                mpc_iters=args.mpc_iters,
                compute_delay_ms=args.mpc_delay,
                u_max=50.0,
                device=args.device,
                warmup_steps=200,
            )
            sim.set_io_processor(ctrl)

            t0 = time.time()
            sim.run(200 * ms)   # baseline (warmup / fill encoder buffer)
            sim.run(1000 * ms)  # clamping
            sim.run(200 * ms)   # recovery
            elapsed = time.time() - t0

            results = ctrl.get_results()
            metrics = compute_metrics(results["rates"], target, 200, 1200)
            metrics["wall_time_s"] = elapsed

            tgt_key = f"{int(frac*100)}pct"
            mpc_results[tgt_key] = {
                "target": target,
                "metrics": metrics,
            }
            trajectories[("NODE_MPC", tgt_key)] = _extract_trajectory(results, target)
            print(f"    RMSE={metrics['tracking_rmse']:.2f}, "
                  f"Settling={metrics['settling_time_ms']:.0f}ms, "
                  f"Wall time={elapsed:.1f}s")

        all_results["NODE_MPC"] = {
            "config": {
                "horizon": args.mpc_horizon,
                "iters": args.mpc_iters,
                "delay_ms": args.mpc_delay,
            },
            "targets": mpc_results,
        }

    # ── Save results + trajectories ─────────────────────────────────────
    results_path = os.path.join(args.output_dir, "optoclamp_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    _save_trajectories(trajectories, args.output_dir)

    # ── Generate plots (from the persisted trajectories) ────────────────
    _plot_comparison(all_results, trajectories, targets, target_fractions, args.output_dir)
    _plot_control_signals(trajectories, targets, target_fractions, args.output_dir)
    print(f"Plots saved to {args.output_dir}/")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Controller':<15} {'Target':<10} {'RMSE':<10} {'Settling':<12} {'SS Error':<10}")
    print("-" * 57)
    for ctrl_name, ctrl_data in all_results.items():
        for tgt_key, tgt_data in ctrl_data["targets"].items():
            m = tgt_data["metrics"]
            print(f"{ctrl_name:<15} {tgt_key:<10} {m['tracking_rmse']:<10.2f} "
                  f"{m['settling_time_ms']:<12.0f} {m['steady_state_error']:<10.2f}")


CTRL_COLORS = {"PI": "#3b82f6", "NODE_MPC": "#ef4444", "LDS_LQR": "#10b981"}
PHASES = (200, 1200, 1400)  # baseline_end, clamp_end, recovery_end (ms; dt=1ms)


def _extract_trajectory(results, target):
    """Pull the persistable time series out of a controller's get_results()."""
    rates = results.get("rates", np.array([]))
    mean_rate = rates.mean(axis=1) if getattr(rates, "size", 0) else np.array([])
    u = results.get("u", np.array([]))
    return {
        "target": float(target),
        "mean_rate": np.asarray(mean_rate, dtype=np.float32),
        "u": np.asarray(u, dtype=np.float32),
    }


def _save_trajectories(trajectories, output_dir, dt_ms=1.0, phases=PHASES):
    """Persist every (controller, target) time series to one .npz, so figures can
    be regenerated later WITHOUT re-running the (slow) closed-loop experiment."""
    if not trajectories:
        return
    arrays = {"dt_ms": np.asarray(dt_ms, dtype=np.float32),
              "phases": np.asarray(phases, dtype=np.int32)}
    meta = {}
    for (ctrl, tgt), d in trajectories.items():
        key = f"{ctrl}__{tgt}"
        arrays[f"{key}__mean_rate"] = d["mean_rate"]
        arrays[f"{key}__u"] = d["u"]
        meta[key] = {"controller": ctrl, "target": tgt, "target_hz": d["target"]}
    arrays["meta_json"] = np.asarray(json.dumps(meta))
    path = os.path.join(output_dir, "optoclamp_trajectories.npz")
    np.savez_compressed(path, **arrays)
    print(f"Trajectories saved to {path} ({len(trajectories)} runs)")


def load_trajectories(npz_path):
    """Inverse of _save_trajectories -> (trajectories, dt_ms, phases)."""
    d = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(d["meta_json"]))
    trajectories = {
        (m["controller"], m["target"]): {
            "target": m["target_hz"],
            "mean_rate": d[f"{key}__mean_rate"],
            "u": d[f"{key}__u"],
        }
        for key, m in meta.items()
    }
    return trajectories, float(d["dt_ms"]), tuple(int(p) for p in d["phases"])


def _plot_comparison(all_results, trajectories, targets, fracs, output_dir,
                     dt_ms=1.0, phases=PHASES):
    """Rate-tracking trajectories (controlled population rate vs. target) per
    target, plus the RMSE bar chart."""
    fig, axes = plt.subplots(len(fracs), 1, figsize=(12, 3.4 * len(fracs)), sharex=True)
    if len(fracs) == 1:
        axes = [axes]
    b_end, c_end, r_end = phases
    for ax, frac, target in zip(axes, fracs, targets):
        ax.axhline(target, color="gray", ls="--", alpha=0.6, label="Target")
        ax.axvspan(0, b_end, alpha=0.06, color="blue")
        ax.axvspan(b_end, c_end, alpha=0.06, color="green")
        ax.axvspan(c_end, r_end, alpha=0.06, color="orange")
        tgt_key = f"{int(frac*100)}pct"
        for ctrl in CTRL_COLORS:
            tr = trajectories.get((ctrl, tgt_key))
            if tr is None or np.size(tr["mean_rate"]) == 0:
                continue
            mr = tr["mean_rate"]
            t = np.arange(len(mr)) * dt_ms
            rmse = (all_results.get(ctrl, {}).get("targets", {}).get(tgt_key, {})
                    .get("metrics", {}).get("tracking_rmse"))
            lbl = ctrl + (f" (RMSE={rmse:.1f})" if isinstance(rmse, (int, float)) else "")
            ax.plot(t, mr, color=CTRL_COLORS[ctrl], lw=1.1, alpha=0.9, label=lbl)
        ax.set_ylabel("Pop. mean rate (Hz)")
        ax.set_title(f"Target: {frac*100:.0f}% of baseline ({target:.1f} Hz)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (ms)  ·  baseline | clamp | recovery")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optoclamp_comparison.png"), dpi=150)
    plt.close()

    # RMSE bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fracs))
    width = 0.25
    for i, (ctrl_name, ctrl_data) in enumerate(all_results.items()):
        rmses = [ctrl_data["targets"].get(f"{int(f*100)}pct", {})
                 .get("metrics", {}).get("tracking_rmse", 0) for f in fracs]
        ax.bar(x + i * width, rmses, width, label=ctrl_name,
               color=CTRL_COLORS.get(ctrl_name, "gray"))
    ax.set_xlabel("Target level")
    ax.set_ylabel("Tracking RMSE (Hz)")
    ax.set_title("Optoclamp: controller comparison")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optoclamp_rmse_comparison.png"), dpi=150)
    plt.close()


def _plot_control_signals(trajectories, targets, fracs, output_dir,
                          dt_ms=1.0, phases=PHASES):
    """Optogenetic control input over time per target (one line per fiber)."""
    if not trajectories:
        return
    fig, axes = plt.subplots(len(fracs), 1, figsize=(12, 3.0 * len(fracs)), sharex=True)
    if len(fracs) == 1:
        axes = [axes]
    b_end, c_end, _ = phases
    for ax, frac in zip(axes, fracs):
        ax.axvspan(b_end, c_end, alpha=0.06, color="green")
        tgt_key = f"{int(frac*100)}pct"
        for ctrl in CTRL_COLORS:
            tr = trajectories.get((ctrl, tgt_key))
            if tr is None or np.size(tr["u"]) == 0:
                continue
            u = np.atleast_2d(tr["u"])
            if u.shape[0] != len(tr["mean_rate"]) and u.shape[1] == len(tr["mean_rate"]):
                u = u.T  # ensure (n_steps, n_u)
            t = np.arange(u.shape[0]) * dt_ms
            for j in range(u.shape[1]):
                ax.plot(t, u[:, j], color=CTRL_COLORS[ctrl], lw=0.9, alpha=0.85,
                        ls=("-" if j == 0 else "--"), label=f"{ctrl} u{j}")
        ax.set_ylabel("Input (mW/mm²)")
        ax.set_title(f"Control signal — {frac*100:.0f}% target")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optoclamp_control_signals.png"), dpi=150)
    plt.close()


def replot_from_npz(npz_path, output_dir=None):
    """Regenerate all optoclamp figures from a persisted trajectories .npz — no
    closed-loop re-run needed (RMSE labels read from optoclamp_results.json)."""
    output_dir = output_dir or os.path.dirname(npz_path)
    trajectories, dt_ms, phases = load_trajectories(npz_path)
    tgt_by_key = {tgt: tr["target"] for (ctrl, tgt), tr in trajectories.items()}
    fracs = sorted({int(k.replace("pct", "")) / 100 for k in tgt_by_key})
    targets = [tgt_by_key[f"{int(f*100)}pct"] for f in fracs]
    rj = os.path.join(output_dir, "optoclamp_results.json")
    all_results = json.load(open(rj)) if os.path.exists(rj) else {}
    _plot_comparison(all_results, trajectories, targets, fracs, output_dir, dt_ms, phases)
    _plot_control_signals(trajectories, targets, fracs, output_dir, dt_ms, phases)
    print(f"Regenerated optoclamp figures in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optoclamp experiment")
    parser.add_argument("--model-checkpoint", type=str, default=None,
                        help="Path to trained causal CA-NODE .pt checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/optoclamp")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mpc-horizon", type=int, default=20)
    parser.add_argument("--mpc-iters", type=int, default=30)
    parser.add_argument("--mpc-delay", type=float, default=5.0,
                        help="MPC compute delay in ms")
    parser.add_argument("--replot-from", type=str, default=None,
                        help="Path to optoclamp_trajectories.npz: regenerate figures "
                             "from persisted trajectories without re-running the sim.")
    args = parser.parse_args()
    if args.replot_from:
        replot_from_npz(args.replot_from)
    else:
        run_optoclamp(args)

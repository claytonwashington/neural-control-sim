#!/usr/bin/env python3
"""Plant dose-response characterization — open-loop sweep for v2 plant.

Uses build_plant_v2 (ChR2-H134R + eNpHR3.0) with fiber_exc/fiber_inh names.

Usage:
    python -m modeling.scripts.dose_response_v2 \
        --output-dir results/dose_response_v2 --seed 42
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

import brian2.only as b2
from brian2 import ms, second, Hz, mwatt, mm

import cleo
from cleo.ioproc import LatencyIOProcessor, exp_firing_rate_estimate

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modeling.plant import build_plant_v2


class ConstantStimulator(LatencyIOProcessor):
    """Apply constant light intensities and record firing rates."""

    def __init__(
        self,
        u_exc: float = 0.0,
        u_inh: float = 0.0,
        sample_period_ms: float = 1.0,
        tau_rate_ms: float = 20.0,
        probe_name: str = "probe",
        mua_name: str = "mua",
        light_name_exc: str = "fiber_exc",
        light_name_inh: str = "fiber_inh",
    ):
        super().__init__(sample_period=sample_period_ms * ms)
        self.u_exc = u_exc
        self.u_inh = u_inh
        self.tau_rate = tau_rate_ms * ms
        self.probe_name = probe_name
        self.mua_name = mua_name
        self.light_name_exc = light_name_exc
        self.light_name_inh = light_name_inh
        self._rates = None
        self.rate_log: list[np.ndarray] = []

    def process(self, state_dict, t_samp):
        i_chan, t_spikes, counts = state_dict[self.probe_name][self.mua_name]
        counts = np.asarray(counts, dtype=float)
        if self._rates is None:
            self._rates = np.zeros(len(counts)) * Hz
        self._rates = exp_firing_rate_estimate(
            counts, self.sample_period, self._rates, self.tau_rate
        )
        self.rate_log.append(np.array(self._rates, dtype=np.float64))
        return {
            self.light_name_exc: np.array([self.u_exc]) * mwatt / mm**2,
            self.light_name_inh: np.array([self.u_inh]) * mwatt / mm**2,
        }, t_samp

    def _base_reset(self):
        super()._base_reset()
        self._rates = None
        self.rate_log.clear()


def run_condition(u_exc, u_inh, seed, settle_ms=1000, measure_ms=1000):
    sim, devices = build_plant_v2(seed=seed)
    stim = ConstantStimulator(u_exc=u_exc, u_inh=u_inh)
    sim.set_io_processor(stim)
    sim.run(settle_ms * ms)
    sim.run(measure_ms * ms)
    all_rates = np.array(stim.rate_log)
    measure_rates = all_rates[settle_ms:]
    pop_mean_rates = measure_rates.mean(axis=1)
    return {
        "u_exc": u_exc,
        "u_inh": u_inh,
        "mean_rate": float(pop_mean_rates.mean()),
        "std_rate": float(pop_mean_rates.std()),
    }


def run_sweep(args):
    os.makedirs(args.output_dir, exist_ok=True)
    irradiances = [0, 1, 2, 5, 10, 20, 30, 50]
    results = {"conditions": [], "meta": {"seed": args.seed, "plant": "v2_ChR2H134R_eNpHR3"}}
    t0 = time.time()

    print("=" * 60)
    print("Plant V2 (ChR2-H134R + eNpHR3.0) Dose-Response")
    print("=" * 60)

    # Inhibition only
    print("\nSweep 1: Inhibition only (exc=0)")
    print("-" * 40)
    for u_inh in irradiances:
        print(f"  u_exc=0, u_inh={u_inh} ... ", end="", flush=True)
        r = run_condition(0, u_inh, args.seed)
        r["group"] = "inh_only"
        results["conditions"].append(r)
        print(f"rate = {r['mean_rate']:.1f} ± {r['std_rate']:.1f} Hz")

    # Excitation only
    print("\nSweep 2: Excitation only (inh=0)")
    print("-" * 40)
    for u_exc in irradiances:
        if u_exc == 0:
            continue
        print(f"  u_exc={u_exc}, u_inh=0 ... ", end="", flush=True)
        r = run_condition(u_exc, 0, args.seed)
        r["group"] = "exc_only"
        results["conditions"].append(r)
        print(f"rate = {r['mean_rate']:.1f} ± {r['std_rate']:.1f} Hz")

    # Combined
    print("\nSweep 3: Combined (exc=5, vary inh)")
    print("-" * 40)
    for u_inh in [0, 5, 10, 20, 30, 50]:
        print(f"  u_exc=5, u_inh={u_inh} ... ", end="", flush=True)
        r = run_condition(5, u_inh, args.seed)
        r["group"] = "combined_exc5"
        results["conditions"].append(r)
        print(f"rate = {r['mean_rate']:.1f} ± {r['std_rate']:.1f} Hz")

    elapsed = time.time() - t0
    results["meta"]["total_time_s"] = elapsed

    # Save
    with open(os.path.join(args.output_dir, "dose_response.json"), "w") as f:
        json.dump(results, f, indent=2)

    baseline = [c for c in results["conditions"] if c["u_exc"] == 0 and c["u_inh"] == 0][0]["mean_rate"]
    results["meta"]["baseline_rate"] = baseline

    # Plot
    _plot(results, baseline, args.output_dir)

    # Summary
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Baseline: {baseline:.1f} Hz")
    print(f"50% target: {baseline * 0.5:.1f} Hz")
    print(f"75% target: {baseline * 0.75:.1f} Hz")
    inh_only = [c for c in results["conditions"] if c["group"] == "inh_only"]
    min_r = min(c["mean_rate"] for c in inh_only)
    print(f"Min rate (inh only): {min_r:.1f} Hz")
    print(f"  → {'CAN' if min_r < baseline * 0.5 else 'CANNOT'} reach 50% target")
    print(f"  → {'CAN' if min_r < baseline * 0.75 else 'CANNOT'} reach 75% target")


def _plot(results, baseline, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fracs = [0.5, 0.75, 1.0, 1.25]
    colors = ["#ef4444", "#f59e0b", "#6b7280", "#10b981"]
    labels = ["50%", "75%", "100%", "125%"]

    for ax in axes:
        for f, c, l in zip(fracs, colors, labels):
            ax.axhline(baseline * f, color=c, ls="--", alpha=0.6, lw=1.5, label=f"{l} target")
        ax.grid(alpha=0.3)
        ax.set_xlim(-1, 52)

    # Panel 1: Inhibition
    ax = axes[0]
    d = sorted([c for c in results["conditions"] if c["group"] == "inh_only"], key=lambda c: c["u_inh"])
    ax.errorbar([c["u_inh"] for c in d], [c["mean_rate"] for c in d],
                yerr=[c["std_rate"] for c in d], marker="o", color="#3b82f6", lw=2, ms=8, capsize=4, label="Rate")
    ax.set_xlabel("Inhibitory irradiance (mW/mm²)")
    ax.set_ylabel("Population mean rate (Hz)")
    ax.set_title("Inhibition Only (eNpHR3.0)", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: Excitation
    ax = axes[1]
    d = sorted([c for c in results["conditions"] if c["group"] == "exc_only" or (c["u_exc"] == 0 and c["u_inh"] == 0)],
               key=lambda c: c["u_exc"])
    ax.errorbar([c["u_exc"] for c in d], [c["mean_rate"] for c in d],
                yerr=[c["std_rate"] for c in d], marker="s", color="#ef4444", lw=2, ms=8, capsize=4, label="Rate")
    ax.set_xlabel("Excitatory irradiance (mW/mm²)")
    ax.set_ylabel("Population mean rate (Hz)")
    ax.set_title("Excitation Only (ChR2-H134R)", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 3: Combined
    ax = axes[2]
    d = sorted([c for c in results["conditions"] if c["group"] == "combined_exc5"], key=lambda c: c["u_inh"])
    ax.errorbar([c["u_inh"] for c in d], [c["mean_rate"] for c in d],
                yerr=[c["std_rate"] for c in d], marker="D", color="#8b5cf6", lw=2, ms=8, capsize=4, label="u_exc=5")
    ax.set_xlabel("Inhibitory irradiance (mW/mm²)")
    ax.set_ylabel("Population mean rate (Hz)")
    ax.set_title("Combined (ChR2=5, vary eNpHR3.0)", fontweight="bold")
    ax.legend(fontsize=8)

    plt.suptitle(f"Plant V2 Dose-Response — Baseline: {baseline:.1f} Hz\n"
                 f"ChR2(H134R) E=0mV + eNpHR3.0 E=-400mV (pump)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dose_response_v2.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/dose_response_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_sweep(args)

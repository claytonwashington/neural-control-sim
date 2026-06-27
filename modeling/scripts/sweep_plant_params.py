"""Calibration sweep for build_plant_v3 (AdEx). Run via:
   cd <adex-plant worktree> && PYTHONPATH=<worktree> python -m modeling.scripts.sweep_plant_params
Measures the spike-count distribution under OU light drive across (I_bg, ou_mu, b)
to find a de-saturated, realistic operating point.
"""
import os, json, itertools, contextlib
import multiprocessing as mp
import numpy as np

DUR = 6.0          # seconds per driven run
BIN_S = 0.01       # 10 ms bins (matches dataset)
N_WORKERS = 12


def run_config(cfg):
    import os, warnings, contextlib, io
    warnings.filterwarnings("ignore")
    import brian2 as b2
    b2.prefs.codegen.runtime.cython.cache_dir = f"/tmp/b2cache_{os.getpid()}"
    try:
        from modeling.plant import build_plant_v3
        from modeling.datagen import generate_dataset

        sim, d = build_plant_v3(
            seed=42, runtime_seed=cfg["idx"] + 1,
            I_bg_pA=cfg["I_bg_pA"], b_pA=cfg["b_pA"],
        )
        ng = d["ng"]
        spikemon = b2.SpikeMonitor(ng)
        sim.network.add(spikemon)

        with contextlib.redirect_stdout(io.StringIO()):
            generate_dataset(
                sim, d, duration_s=DUR, sample_period_ms=1.0, tau_smooth_ms=20.0,
                ou_tau=0.05, ou_sigma=cfg["ou_sigma"], ou_mu=cfg["ou_mu"], seed=42,
            )

        ti = np.asarray(spikemon.t / b2.second)
        ii = np.asarray(spikemon.i)
        n = len(ng)
        T = int(DUR / BIN_S)
        counts = np.zeros((n, T), dtype=np.int32)
        if ti.size:
            bidx = np.clip((ti / BIN_S).astype(int), 0, T - 1)
            np.add.at(counts, (ii, bidx), 1)
        flat = counts.ravel().astype(float)
        m = flat.mean()
        third = max(T // 3, 1)
        early = counts[:, :third].mean() / BIN_S
        late = counts[:, -third:].mean() / BIN_S
        return {
            **cfg,
            "mean_rate_Hz": round(m / BIN_S, 2),
            "pct_zero": round((flat == 0).mean() * 100, 1),
            "pct_railed_ge5": round((flat >= 5).mean() * 100, 3),
            "fano": round(flat.var() / m, 2) if m > 0 else 0.0,
            "max_count": int(flat.max()),
            "adapt_late_over_early": round(late / early, 2) if early > 0 else None,
            "total_spikes": int(flat.sum()),
        }
    except Exception as e:
        import traceback
        return {**cfg, "ERROR": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-400:]}


if __name__ == "__main__":
    grid = []
    for i, (ibg, mu, b) in enumerate(itertools.product(
            [600, 650, 700], [0, 20, 40], [80])):
        grid.append({"idx": i, "I_bg_pA": float(ibg), "ou_mu": float(mu),
                     "b_pA": float(b), "ou_sigma": round(0.5 * mu, 1)})
    print(f"Sweeping {len(grid)} configs, {N_WORKERS} workers, {DUR}s each...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        results = pool.map(run_config, grid)

    json.dump(results, open("/tmp/sweep_results.json", "w"), indent=1)
    errs = [r for r in results if "ERROR" in r]
    ok = [r for r in results if "ERROR" not in r]
    print(f"\n=== {len(ok)} ok, {len(errs)} errored ===")
    if errs:
        print("FIRST ERROR:", errs[0].get("ERROR"), errs[0].get("tb", ""))
    print(f"\n{'idx':>3} {'Ibg':>4} {'ou_mu':>6} {'b':>5} {'rate_Hz':>8} {'%zero':>6} {'%railed':>8} {'fano':>5} {'maxc':>5} {'adapt':>6}")
    for r in sorted(ok, key=lambda r: (r["I_bg_pA"], r["ou_mu"], r["b_pA"])):
        print(f"{r['idx']:>3} {r['I_bg_pA']:>4.0f} {r['ou_mu']:>6.0f} {r['b_pA']:>5.0f} "
              f"{r['mean_rate_Hz']:>8.2f} {r['pct_zero']:>6.1f} {r['pct_railed_ge5']:>8.3f} "
              f"{r['fano']:>5.2f} {r['max_count']:>5d} {str(r['adapt_late_over_early']):>6}")
    print("\nTargets: rate ~5-30 Hz, %railed <5, fano ~1-1.5, adapt <1 (firing decays).")
    print("Wrote /tmp/sweep_results.json")

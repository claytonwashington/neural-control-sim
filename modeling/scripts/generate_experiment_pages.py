#!/usr/bin/env python
"""Generate a standalone, self-contained HTML page per experiment.

Each page (``results/experiments/<slug>.html``) embeds — as base64, so the file
works when opened directly — the experiment's metadata, hypothesis, leaderboard
metrics, the plant it ran on, and the validation plots produced by
``validation_plots.py`` (channel heatmaps, channel traces, PC projections).
Leaderboard rows in the main dashboard link here via the shared ``slugify``.

Experiments without a saved checkpoint get a metadata-only page (plus any
pre-existing prediction image referenced in the registry).

Run::

    python -m modeling.scripts.generate_experiment_pages
"""

from __future__ import annotations

import json
from html import escape as _esc

from modeling.scripts.dashboard_common import (
    EXPERIMENTS_DIR,
    PLANTS_BY_ID,
    PLANTS_DIR,
    RESULTS,
    SHARED_RESULTS,
    all_experiments,
    no_plots_reason,
    png_to_data_uri,
    slugify,
)

VAL_FILES = [
    ("val_channel_heatmap.png", "True vs. predicted firing rates — all channels",
     "Ground-truth, model-predicted, and absolute-error firing-rate heatmaps across every "
     "channel for the held-out test trial."),
    ("val_pc_projection.png", "Top principal components",
     "True (white) and predicted (orange) trajectories projected onto the leading PCs of the "
     "true data, with per-PC R² and variance explained."),
    ("val_channel_traces.png", "Top-variance channel traces",
     "Per-channel true-vs-predicted overlays for the highest-variance channels."),
]

STYLE = """
  :root{
    --bg:#0a0e1a; --panel:#111827; --card:#141a2e; --border:#2a3055;
    --text:#e5e7eb; --muted:#9ca3af; --green:#10b981; --cyan:#22d3ee;
    --indigo:#6366f1; --amber:#f59e0b;
  }
  *{box-sizing:border-box} html,body{margin:0}
  body{background:var(--bg); color:var(--text); font-family:-apple-system,
    BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
    line-height:1.5; padding:0 0 60px}
  a{color:var(--cyan); text-decoration:none} a:hover{text-decoration:underline}
  .wrap{max-width:1080px; margin:0 auto; padding:0 22px}
  header{border-bottom:1px solid var(--border); padding:22px 0 18px; margin-bottom:24px;
    background:linear-gradient(180deg,#0d1326,transparent)}
  h1{font-size:1.5rem; margin:6px 0 2px}
  .back{font-size:.85rem; color:var(--muted)}
  .pill{display:inline-block; font-size:.7rem; padding:2px 9px; border-radius:999px;
    border:1px solid var(--border); background:var(--panel); color:var(--muted); margin-right:6px}
  .grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:18px 0}
  .stat{background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px 16px}
  .stat .k{font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em}
  .stat .v{font-size:1.25rem; font-weight:700; margin-top:4px}
  .card{background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:18px 20px; margin:18px 0}
  .card h3{margin:0 0 8px; font-size:1rem}
  .muted{color:var(--muted)} .small{font-size:.85rem}
  .imgcard{background:var(--card); border:1px solid var(--border); border-radius:14px;
    overflow:hidden; margin:22px 0}
  .imgcard .h{padding:12px 18px; border-bottom:1px solid var(--border); font-weight:600}
  .imgcard img{display:block; width:100%; background:var(--bg); cursor:zoom-in}
  .imgcard .cap{padding:9px 18px; font-size:.8rem; color:var(--muted)}
  .plantrow{display:flex; gap:18px; align-items:center; flex-wrap:wrap}
  .plantrow img{width:240px; max-width:100%; border-radius:10px; border:1px solid var(--border)}
  .lb{position:fixed; inset:0; background:rgba(0,0,0,.92); display:none;
    align-items:center; justify-content:center; z-index:50; cursor:zoom-out}
  .lb img{max-width:96%; max-height:96%}
  .ctbl{border-collapse:collapse; width:100%; font-size:.82rem; margin-top:10px}
  .ctbl th,.ctbl td{border:1px solid var(--border); padding:5px 10px; text-align:left}
  .ctbl th{color:var(--muted); font-weight:600}
"""

LIGHTBOX_JS = """
  function zoom(img){var l=document.getElementById('lb');
    document.getElementById('lbimg').src=img.src; l.style.display='flex';}
  function closeLb(){document.getElementById('lb').style.display='none';}
"""


def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _load_leaderboard() -> dict:
    for root in (RESULTS, SHARED_RESULTS):
        p = root / "leaderboard.json"
        if p.exists():
            with open(p) as f:
                return {e.get("model", ""): e for e in json.load(f)}
    return {}


def _img(path, title, caption):
    return f"""    <div class="imgcard">
      <div class="h">{title}</div>
      <img src="{png_to_data_uri(path)}" alt="{title}" onclick="zoom(this)">
      <div class="cap">{caption}</div>
    </div>"""


# Control experiments (closed-loop, e.g. optoclamp) need a different "validation"
# view: how well the controller tracks firing-rate targets, not true-vs-inferred.
_CTRL_METRICS = [
    ("tracking_rmse", "Tracking RMSE (Hz)"),
    ("settling_time_ms", "Settling (ms)"),
    ("steady_state_error", "SS error (Hz)"),
    ("overshoot", "Overshoot (Hz)"),
]
_CTRL_PLOTS = [
    ("optoclamp_comparison.png", "Tracking — controlled rate vs. target"),
    ("optoclamp_control_signals.png", "Control signal (optogenetic input) over time"),
    ("optoclamp_rmse_comparison.png", "Tracking RMSE by controller × target"),
]


def _control_view(entry: dict) -> str:
    """Closed-loop control view (tracking plots + metrics table) for a control
    experiment, or "" if it isn't one / has no control results."""
    rd = entry.get("results_dir") or ((entry.get("checkpoint_dir") or "").split("/")[0] or None)
    if not rd:
        return ""
    base = next((root / rd for root in (RESULTS, SHARED_RESULTS)
                 if (root / rd / "optoclamp_results.json").exists()), None)
    if base is None:
        return ""
    with open(base / "optoclamp_results.json") as f:
        res = json.load(f)

    controllers = list(res.keys())
    targets = []
    for c in controllers:
        for t in res[c].get("targets", {}):
            if t not in targets:
                targets.append(t)

    def _fmt_m(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else _esc(str(v))

    rows = []
    for t in targets:
        for c in controllers:
            tg = res[c].get("targets", {}).get(t, {})
            tgt_hz = tg.get("target")
            m = tg.get("metrics", {})
            cells = "".join(f"<td>{_fmt_m(m.get(k))}</td>" for k, _ in _CTRL_METRICS)
            tlabel = f"{_esc(t)}" + (f" ({tgt_hz:.0f} Hz)" if isinstance(tgt_hz, (int, float)) else "")
            rows.append(f"<tr><td>{tlabel}</td><td>{_esc(c)}</td>{cells}</tr>")
    header = "".join(f"<th>{lbl}</th>" for _, lbl in _CTRL_METRICS)
    table = (f'<table class="ctbl"><thead><tr><th>Target</th><th>Controller</th>'
             f'{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>')

    imgs = []
    for png, title in _CTRL_PLOTS:
        p = base / png
        if p.exists():
            imgs.append(_img(p, title, ""))

    return (
        '  <div class="card"><h3>🎛️ Closed-loop control performance</h3>'
        '<p class="muted small">This is a control experiment, so "validation" means how well '
        'the controller tracks firing-rate targets (lower RMSE / SS-error is better), not '
        'true-vs-inferred rates. Per-step trajectories aren\'t persisted, so these are the '
        f"run's saved figures and recorded metrics.</p>{table}</div>\n" + "\n".join(imgs))


def build_page(entry: dict, lb: dict) -> str:
    name = entry["name"]
    ckpt_dir = entry.get("checkpoint_dir")
    lbrow = lb.get(name, {})
    r2 = lbrow.get("r2", entry.get("best_r2"))
    mse = lbrow.get("mse")
    plant = PLANTS_BY_ID.get(entry.get("plant_id", ""), {})

    # validation assets + metrics
    val_dir = (RESULTS / ckpt_dir) if ckpt_dir else None
    metrics = {}
    if val_dir and (val_dir / "val_metrics.json").exists():
        with open(val_dir / "val_metrics.json") as f:
            metrics = json.load(f)

    img_blocks = []
    if val_dir:
        for fname, title, cap in VAL_FILES:
            p = val_dir / fname
            if p.exists():
                img_blocks.append(_img(p, title, cap))
    if not img_blocks and entry.get("existing_plot"):
        for root in (RESULTS, SHARED_RESULTS):
            p = root / entry["existing_plot"]
            if p.exists():
                img_blocks.append(_img(p, "Prediction traces",
                                       "Pre-existing prediction figure (no saved checkpoint to "
                                       "regenerate full validation plots)."))
                break

    # stat tiles
    pc_r2 = metrics.get("r2_per_pc_top4") or []
    ch_r2 = [v for v in (metrics.get("r2_per_channel") or []) if isinstance(v, (int, float))]
    mean_ch = sum(ch_r2) / len(ch_r2) if ch_r2 else None
    stats = [
        ("Leaderboard R²", _fmt(r2, 4)),
        ("MSE", _fmt(mse, 4) if mse is not None else "—"),
        ("Test-trial R²", _fmt(metrics.get("r2_overall"), 4) if metrics else "—"),
        ("Mean per-channel R²", _fmt(mean_ch, 3) if mean_ch is not None else "—"),
        ("PC1 R²", _fmt(pc_r2[0], 3) if pc_r2 else "—"),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in stats
    )

    # plant card
    plant_html = ""
    if plant:
        pimg = PLANTS_DIR / f"{plant['id']}_setup.png"
        thumb = (f'<img src="{png_to_data_uri(pimg)}" alt="{_esc(plant["title"], quote=True)}" onclick="zoom(this)">'
                 if pimg.exists() else "")
        plant_html = f"""  <div class="card">
    <h3>🧠 Plant — {_esc(plant['title'])}</h3>
    <div class="plantrow">
      {thumb}
      <div class="small muted" style="flex:1; min-width:260px">
        <p>{_esc(plant['desc'])}</p>
        <p><strong>Neurons:</strong> {_esc(plant['neurons'])}<br>
           <strong>Opsins:</strong> {_esc(plant['opsins'])}<br>
           <strong>Inputs:</strong> {_esc(plant['inputs'])}</p>
        <p><a href="../dashboard.html">↗ View all plants on the dashboard</a></p>
      </div>
    </div>
  </div>"""

    hypothesis = entry.get("hypothesis") or entry.get("note") or ""
    idea = entry.get("idea_id")
    pills = []
    if entry.get("model_type"):
        pills.append(f'<span class="pill">{_esc(str(entry["model_type"]))}</span>')
    if entry.get("data_file"):
        pills.append(f'<span class="pill">{_esc(str(entry["data_file"]))}</span>')
    if idea is not None:
        pills.append(f'<span class="pill">idea #{idea}</span>')
    if not ckpt_dir:
        pills.append('<span class="pill">metadata only</span>')

    twin_subtext = ("True vs. inferred firing rates on the held-out test trial "
                    "(40/10 split, seed 42).")
    control_html = _control_view(entry)
    if img_blocks:
        body_imgs, val_subtext = "\n".join(img_blocks), twin_subtext
    elif control_html:
        body_imgs, val_subtext = control_html, "Closed-loop control tracking performance."
    else:
        reason = _esc(no_plots_reason(entry) or "Validation plots are not available.")
        body_imgs = (
            f'<div class="card muted small"><strong>No validation plots.</strong> '
            f'{reason} Aggregate leaderboard metrics are shown above.</div>')
        val_subtext = twin_subtext

    hyp_html = (f"""  <div class="card"><h3>Hypothesis / Notes</h3>
    <p class="muted">{_esc(hypothesis)}</p></div>""" if hypothesis else "")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)} — Experiment</title>
<style>{STYLE}</style></head>
<body>
  <header><div class="wrap">
    <div class="back"><a href="../dashboard.html">← Back to dashboard</a></div>
    <h1>{_esc(name)}</h1>
    <div>{''.join(pills)}</div>
  </div></header>
  <div class="wrap">
    <div class="grid">{stat_html}</div>
{hyp_html}
{plant_html}
    <h2 style="font-size:1.1rem; margin:26px 0 4px">Validation</h2>
    <p class="muted small">{val_subtext}</p>
{body_imgs}
  </div>
  <div class="lb" id="lb" onclick="closeLb()"><img id="lbimg" src=""></div>
  <script>{LIGHTBOX_JS}</script>
</body></html>"""


def main():
    print("=== Generating Experiment Pages ===")
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    lb = _load_leaderboard()
    n = 0
    index = []
    for entry in all_experiments():
        slug = slugify(entry["name"])
        html = build_page(entry, lb)
        out = EXPERIMENTS_DIR / f"{slug}.html"
        with open(out, "w") as f:
            f.write(html)
        has_val = bool(entry.get("checkpoint_dir") and
                       (RESULTS / entry["checkpoint_dir"] / "val_metrics.json").exists())
        index.append({"name": entry["name"], "slug": slug, "has_validation": has_val})
        print(f"  wrote {out.name}  ({'validation' if has_val else 'metadata-only'})")
        n += 1
    with open(EXPERIMENTS_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Done: {n} experiment page(s) -> {EXPERIMENTS_DIR}")


if __name__ == "__main__":
    main()

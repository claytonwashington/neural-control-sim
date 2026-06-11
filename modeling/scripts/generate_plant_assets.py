#!/usr/bin/env python
"""Render 3D visualizations for every plant design on the Plants page.

For each entry in ``dashboard_common.PLANTS`` this builds the Cleo simulator via
its registered builder, renders it with ``cleo.viz.plot`` (dark theme matching
the dashboard), and writes:

  * ``results/plants/<id>_setup.png``  — the 3D render
  * ``results/plants/plants.json``     — metadata consumed by the Plants tab

Run (from the worktree, dtmodeling env)::

    python -m modeling.scripts.generate_plant_assets
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import modeling.plant as plant_mod  # noqa: E402
from modeling.scripts.dashboard_common import COLORS, PLANTS, PLANTS_DIR  # noqa: E402


def _style_axes3d(ax):
    """Apply the dark dashboard theme to a 3D axis."""
    ax.set_facecolor(COLORS["panel"])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(COLORS["border"])
    ax.tick_params(colors="#6b7280", labelsize=7)
    ax.xaxis.label.set_color(COLORS["muted"])
    ax.yaxis.label.set_color(COLORS["muted"])
    ax.zaxis.label.set_color(COLORS["muted"])
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor("#1a1f35")
        leg.get_frame().set_edgecolor(COLORS["border"])
        for text in leg.get_texts():
            text.set_color(COLORS["text"])


def render_plant(entry: dict) -> str:
    """Build and render one plant; return the output PNG path (str)."""
    import cleo.viz
    from brian2 import um

    builder = getattr(plant_mod, entry["builder"])
    print(f"  Building plant '{entry['id']}' via {entry['builder']}() ...")
    sim, devices = builder()

    fig, ax = cleo.viz.plot(
        sim=sim,
        axis_scale_unit=um,
        figsize=(10, 8),
        scatterargs={"s": 2, "alpha": 0.4},
    )
    ax.set_title(
        f"{entry['title']}\n{entry['neurons']} · {entry['probe']}",
        color=COLORS["text"],
        fontsize=11,
        fontweight="bold",
        pad=15,
    )
    fig.patch.set_facecolor(COLORS["bg"])
    _style_axes3d(ax)
    fig.tight_layout()

    PLANTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLANTS_DIR / f"{entry['id']}_setup.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")
    return str(out_path)


def main():
    print("=== Generating Plant Assets ===")
    PLANTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = []
    for entry in PLANTS:
        try:
            render_plant(entry)
            png_rel = f"plants/{entry['id']}_setup.png"
            status = "ok"
        except Exception as e:  # pragma: no cover - rendering is environment-sensitive
            print(f"    [WARN] render failed for '{entry['id']}': {e}")
            png_rel, status = None, f"error: {e}"
        item = {k: v for k, v in entry.items() if k != "builder"}
        item["image"] = png_rel
        item["status"] = status
        meta.append(item)

    plants_json = PLANTS_DIR / "plants.json"
    with open(plants_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote metadata: {plants_json}")
    print("Done.")


if __name__ == "__main__":
    main()

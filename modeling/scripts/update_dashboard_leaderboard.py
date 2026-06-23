#!/usr/bin/env python3
import os
import re
import json
from datetime import datetime

import argparse

from html import escape as _esc

from modeling.scripts.dashboard_common import EXPERIMENTS_DIR, slugify

def _update_leaderboard_from_results(repo_root, results_dir):
    try:
        return _update_leaderboard_from_results_ORIG(repo_root, results_dir)
    except Exception as e:
        print(f"WARNING: Leaderboard update skipped: {e}")

def _update_leaderboard_from_results_ORIG(repo_root, results_dir):
    """Add entries from a results directory to leaderboard.json."""
    import glob
    results_dir_abs = os.path.join(repo_root, results_dir) if not os.path.isabs(results_dir) else results_dir
    leaderboard_path = os.path.join(repo_root, "results/leaderboard.json")

    # Load existing leaderboard
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f:
            leaderboard = json.load(f)
    else:
        leaderboard = []

    # Load MANIFEST for experiment info
    manifest_path = os.path.join(results_dir_abs, "MANIFEST.json")
    exp_name = "Unknown"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        exp_name = manifest.get("experiment_name", "Unknown")

    # Find best result across all runs
    best_r2 = -1
    best_mse = float("inf")
    best_config = ""
    for rpath in glob.glob(os.path.join(results_dir_abs, "run_*/results.json")):
        with open(rpath) as f:
            r = json.load(f)
        r2 = r.get("r2", r.get("r2_200step", r.get("val_r2", -1)))
        mse = r.get("mse", r.get("best_val_loss", float("inf")))
        if r2 > best_r2:
            best_r2 = r2
            best_mse = mse
            best_config = os.path.basename(os.path.dirname(rpath))

    # Also check for single-run results
    single_results = os.path.join(results_dir_abs, "results.json")
    if os.path.exists(single_results):
        with open(single_results) as f:
            r = json.load(f)
        r2 = r.get("r2", r.get("r2_200step", -1))
        mse = r.get("mse", r.get("best_val_loss", float("inf")))
        if r2 > best_r2:
            best_r2 = r2
            best_mse = mse
            best_config = "single run"

    if best_r2 <= 0:
        print(f"WARNING: No valid results found in {results_dir}")
        return

    # Remove old entries with the same model name (avoid duplicates)
    leaderboard = [e for e in leaderboard if e.get("model") != exp_name]

    # Determine verdict
    baseline_r2 = 0.9009  # CA-NODE baseline
    if best_r2 > baseline_r2:
        verdict = "NEW BEST" if best_r2 > max((e.get("r2", 0) for e in leaderboard), default=0) else "BEATS BASELINE"
        verdict_color = "#4caf50"
    else:
        verdict = "BELOW BASELINE"
        verdict_color = "#ff9800"

    entry = {
        "model": exp_name,
        "type": best_config,
        "r2": round(best_r2, 4),
        "mse": round(best_mse, 4),
        "verdict": verdict,
        "verdict_color": verdict_color,
        "is_best_row": verdict == "NEW BEST",
        "r2_style": "font-weight:bold;color:#4caf50" if best_r2 > baseline_r2 else "",
    }
    leaderboard.append(entry)
    leaderboard.sort(key=lambda x: x.get("r2", -999), reverse=True)

    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Updated leaderboard: {exp_name} R2={best_r2:.4f} ({best_config})")



def main():
    parser = argparse.ArgumentParser(description="Update dashboard leaderboard")
    parser.add_argument("--results-dir", default=None, help="Results directory (optional)")
    args = parser.parse_args()

    # Detect repo root dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # If --results-dir provided, update leaderboard.json first
    if args.results_dir:
        _update_leaderboard_from_results(repo_root, args.results_dir)
    
    leaderboard_path = os.path.join(repo_root, "results/leaderboard.json")
    dashboard_path = os.path.join(repo_root, "results/dashboard.html")

    # leaderboard.json is git-ignored and may live only in the shared checkout.
    if not os.path.exists(leaderboard_path):
        from modeling.scripts.dashboard_common import SHARED_RESULTS
        shared_lb = str(SHARED_RESULTS / "leaderboard.json")
        if os.path.exists(shared_lb):
            leaderboard_path = shared_lb

    if not os.path.exists(leaderboard_path):
        print(f"Error: {leaderboard_path} not found.")
        return
        
    if not os.path.exists(dashboard_path):
        print(f"Error: {dashboard_path} not found.")
        return

    with open(leaderboard_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Sort data by R^2 descending
    data.sort(key=lambda x: x.get("r2", -999.0), reverse=True)
    
    # Generate HTML rows
    html_lines = []
    html_lines.append("          <tbody>")
    
    for item in data:
        model = item.get("model", "")
        m_type = item.get("type", "")
        r2 = item.get("r2", 0.0)
        mse = item.get("mse", 0.0)
        verdict = item.get("verdict", "")
        verdict_color = item.get("verdict_color", "")
        is_best_row = item.get("is_best_row", False)
        r2_style = item.get("r2_style", "")
        
        # Format R^2 span
        if r2_style:
            r2_span = f'<span style="{r2_style}">{r2:.4f}</span>'
        else:
            r2_span = f'<span>{r2:.4f}</span>'
            
        # Format MSE
        if isinstance(mse, (int, float)):
            if mse >= 1000:
                mse_str = f"{mse:,.0f}"
            elif hasattr(mse, "is_integer") and mse.is_integer():
                mse_str = f"{int(mse)}"
            else:
                mse_str = f"{mse:.4f}"
        else:
            mse_str = str(mse)
            
        # Verdict column (verdict text escaped; color is controlled)
        if verdict_color:
            verdict_td = f'<td style="color:{verdict_color}">{_esc(verdict)}</td>'
        else:
            verdict_td = f'<td>{_esc(verdict)}</td>'
            
        # Row class
        if is_best_row:
            row_tr = '            <tr class="best-row">'
        else:
            row_tr = '            <tr>'
            
        html_lines.append(row_tr)
        # Link the model name to its standalone experiment page when one exists.
        slug = slugify(model)
        model_e = _esc(model)
        if (EXPERIMENTS_DIR / f"{slug}.html").exists():
            label = (f'<a href="experiments/{slug}.html" target="_blank" '
                     f'style="color:var(--cyan);text-decoration:none">{model_e} '
                     f'<span style="opacity:.6">↗</span></a>')
        else:
            label = model_e
        html_lines.append(f'              <td class="td-label">{label}</td>')
        html_lines.append(f'              <td>{_esc(m_type)}</td>')
        html_lines.append(f'              <td><div class="r2-cell">{r2_span}</div></td>')
        html_lines.append(f'              <td>{mse_str}</td>')
        html_lines.append(f'              {verdict_td}')
        html_lines.append('            </tr>')
        
    html_lines.append("          </tbody>")
    new_tbody = "\n".join(html_lines)
    
    # Read dashboard
    with open(dashboard_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    start_tag = "<!-- LEADERBOARD_ROWS_START -->"
    end_tag = "<!-- LEADERBOARD_ROWS_END -->"
    
    start_idx = content.find(start_tag)
    end_idx = content.find(end_tag)
    
    if start_idx == -1 or end_idx == -1:
        print("Error: Could not find leaderboard tags in dashboard.html.")
        return
        
    # Reconstruct the file contents
    new_content = (
        content[:start_idx + len(start_tag)] +
        "\n" + new_tbody + "\n          " +
        content[end_idx:]
    )
    
    # Update timestamp
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M EDT')
    new_content = re.sub(r'Last updated: [\d\-]+ [\d:]+ EDT', f'Last updated: {now_str}', new_content)
    
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Successfully updated dashboard leaderboard and timestamp!")

if __name__ == "__main__":
    main()

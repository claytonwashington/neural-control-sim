#!/usr/bin/env python3
import os
import re
import json
from datetime import datetime

import argparse

def main():
    parser = argparse.ArgumentParser(description="Update dashboard leaderboard")
    parser.add_argument("--results-dir", default=None, help="Results directory (optional)")
    args = parser.parse_args()

    # Detect repo root dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    leaderboard_path = os.path.join(repo_root, "results/leaderboard.json")
    dashboard_path = os.path.join(repo_root, "results/dashboard.html")
    
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
            
        # Verdict column
        if verdict_color:
            verdict_td = f'<td style="color:{verdict_color}">{verdict}</td>'
        else:
            verdict_td = f'<td>{verdict}</td>'
            
        # Row class
        if is_best_row:
            row_tr = '            <tr class="best-row">'
        else:
            row_tr = '            <tr>'
            
        html_lines.append(row_tr)
        html_lines.append(f'              <td class="td-label">{model}</td>')
        html_lines.append(f'              <td>{m_type}</td>')
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

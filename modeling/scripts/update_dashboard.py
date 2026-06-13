#!/usr/bin/env python
"""Embed prediction plots and dataset info into dashboard.html.

This script:
1. Reads existing dashboard.html
2. Converts PNG files to base64 data URIs 
3. Inserts dataset description section
4. Inserts plant visualization
5. Inserts prediction plots into relevant tabs
6. Writes updated dashboard.html
"""

import base64
import os
import re
from datetime import datetime


def png_to_data_uri(filepath):
    """Convert PNG file to base64 data URI."""
    with open(filepath, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f'data:image/png;base64,{data}'


def make_image_card(title, data_uri, caption=''):
    """Create an image card matching the dashboard glassmorphism style."""
    return f'''      <div class="image-card" style="margin-top: 20px;">
        <div class="img-header">{title}</div>
        <img src="{data_uri}" alt="{title}" onclick="openLightbox(this)" style="cursor:zoom-in; background: #0a0e1a;">
        {f'<div style="padding: 8px 18px; font-size: 0.8rem; color: var(--text-secondary);">{caption}</div>' if caption else ''}
      </div>'''


# ===========================================================================
# Plants tab (idempotent, marker-guarded)
# ===========================================================================
PLANTS_BTN_START = '<!-- PLANTS_TAB_BTN_START -->'
PLANTS_BTN_END = '<!-- PLANTS_TAB_BTN_END -->'
PLANTS_START = '<!-- PLANTS_TAB_START -->'
PLANTS_END = '<!-- PLANTS_TAB_END -->'


def _upsert(html, start, end, payload, anchor, before=False):
    """Insert or replace ``start+payload+end`` (idempotent).

    If the markers already exist, replace what's between them. Otherwise insert
    the whole block immediately after ``anchor`` (or before it if ``before``).
    """
    block = f'{start}\n{payload}\n{end}'
    if start in html and end in html:
        pre, rest = html.split(start, 1)
        _, post = rest.split(end, 1)
        return pre + block + post
    idx = html.find(anchor)
    if idx == -1:
        print(f'  ⚠️  anchor not found, skipping inject: {anchor[:40]!r}')
        return html
    if before:
        return html[:idx] + block + '\n\n  ' + html[idx:]
    idx += len(anchor)
    return html[:idx] + '\n  ' + block + html[idx:]


def build_plants_tab():
    """Return (button_html, content_html) for the Plants tab from plants.json."""
    import json
    from html import escape as _esc
    from modeling.scripts.dashboard_common import (
        PLANTS_DIR, EXPERIMENTS_DIR, all_experiments, slugify,
    )

    plants_json = PLANTS_DIR / 'plants.json'
    if not plants_json.exists():
        print(f'  ⚠️  {plants_json} not found — run generate_plant_assets first')
        return None, None
    with open(plants_json) as f:
        plants = json.load(f)

    # Map plant_id -> [(name, slug, has_page), ...]
    exps_by_plant = {}
    for e in all_experiments():
        slug = slugify(e['name'])
        has_page = (EXPERIMENTS_DIR / f'{slug}.html').exists()
        exps_by_plant.setdefault(e.get('plant_id'), []).append((e['name'], slug, has_page))

    cards = []
    for p in plants:
        img_html = ''
        png = PLANTS_DIR / f"{p['id']}_setup.png"
        if png.exists():
            img_html = (f'<img src="{png_to_data_uri(png)}" alt="{_esc(p["title"], quote=True)}" '
                        f'onclick="openLightbox(this)" '
                        f'style="max-width:560px;width:100%;cursor:zoom-in;background:#0a0e1a;border-radius:8px;">')
        datasets = ''.join(
            f'<code style="background:var(--bg-secondary);padding:2px 6px;border-radius:4px;'
            f'font-size:0.72rem;margin-right:6px;">{_esc(d)}</code>' for d in p.get('datasets', [])
        )
        exp_links = []
        for name, slug, has_page in exps_by_plant.get(p['id'], []):
            if has_page:
                exp_links.append(f'<a href="experiments/{slug}.html" target="_blank" '
                                 f'style="color:var(--cyan);">{_esc(name)}</a>')
            else:
                exp_links.append(f'<span style="color:var(--text-muted);">{_esc(name)}</span>')
        exp_html = ' · '.join(exp_links) if exp_links else '<span style="color:var(--text-muted);">—</span>'

        cards.append(f'''
      <div class="section">
        <div style="background: var(--gradient-card); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden;">
          <div style="padding: 14px 18px; border-bottom: 1px solid var(--border); font-size: 0.95rem; font-weight: 700;">🧠 {_esc(p['title'])}</div>
          <div style="display:flex; gap:20px; padding:18px; flex-wrap:wrap;">
            <div style="flex:0 0 auto;">{img_html}</div>
            <div style="flex:1; min-width:280px; font-size:0.82rem; color:var(--text-secondary);">
              <p>{_esc(p['desc'])}</p>
              <p style="margin-top:10px;"><strong style="color:var(--text-primary);">Neurons:</strong> {_esc(p['neurons'])}<br>
                 <strong style="color:var(--text-primary);">Opsins:</strong> {_esc(p['opsins'])}<br>
                 <strong style="color:var(--text-primary);">Inputs:</strong> {_esc(p['inputs'])}<br>
                 <strong style="color:var(--text-primary);">Probe:</strong> {_esc(p['probe'])}</p>
              <p style="margin-top:10px;"><strong style="color:var(--text-primary);">Datasets:</strong><br>{datasets}</p>
              <p style="margin-top:10px;"><strong style="color:var(--text-primary);">Experiments:</strong> {exp_html}</p>
            </div>
          </div>
        </div>
      </div>''')

    button = ('  <button class="tab-btn" onclick="showTab(\'plants\')" '
              'style="background: linear-gradient(135deg, #0ea5e9, #6366f1);">🧠 Plants</button>')
    content = ('  <!-- ========== Tab: Plants ========== -->\n'
               '  <div id="tab-plants" class="tab-content">\n'
               '    <div class="section">\n'
               '      <div class="section-header"><h2>Simulated Plants</h2>'
               '<span class="badge">Cleo</span></div>\n'
               '      <p style="color:var(--text-secondary); font-size:0.85rem; margin-bottom:8px;">'
               'The digital-twin datasets come from these Cleo simulations. Click an image to enlarge.</p>\n'
               '    </div>\n'
               + '\n'.join(cards) +
               '\n  </div>')
    return button, content


def inject_plants_tab(html):
    """Idempotently add the Plants tab button + content to the dashboard HTML."""
    button, content = build_plants_tab()
    if button is None:
        return html
    html = _upsert(html, PLANTS_BTN_START, PLANTS_BTN_END, button, '<div class="tabs">')
    # Insert the content just before the first tab-content block.
    html = _upsert(html, PLANTS_START, PLANTS_END, content,
                   '<!-- ========== Tab: Latent NODE', before=True)
    return html


def plants_only():
    """Inject just the Plants tab into the existing dashboard (idempotent)."""
    from modeling.scripts.dashboard_common import RESULTS
    dashboard_path = str(RESULTS / 'dashboard.html')
    with open(dashboard_path) as f:
        html = f.read()
    html = inject_plants_tab(html)
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M EDT')
    html = re.sub(r'Last updated: [\d\-]+ [\d:]+ EDT', f'Last updated: {now_str}', html)
    with open(dashboard_path, 'w') as f:
        f.write(html)
    print(f'✅ Plants tab injected into {dashboard_path}')


def main():
    os.chdir('/snel/home/cbwash2/cleo')
    
    dashboard_path = 'results/dashboard.html'
    
    with open(dashboard_path, 'r') as f:
        html = f.read()
    
    # ====================================================================
    # 1. DATASET DESCRIPTION — insert after metrics-row, before tabs
    # ====================================================================
    dataset_section = '''
  <!-- Dataset Description -->
  <div class="section" style="margin-bottom: 24px;">
    <div style="background: var(--gradient-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px 28px;">
      <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 1.3rem;">🧬</span>
        <h3 style="font-size: 1rem; font-weight: 700;">Dataset &amp; Plant</h3>
        <span class="badge" style="font-size: 0.6rem;">Cleo Simulation</span>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 0.82rem; color: var(--text-secondary);">
        <div>
          <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">📊 Data</div>
          <p>50 trials × 30s × 1kHz × 50 channels</p>
          <p>40 train / 10 test (seed=42)</p>
          <p style="margin-top: 4px;"><code style="background: var(--bg-secondary); padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">data/training_trials.h5</code></p>
        </div>
        <div>
          <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">🔬 Plant</div>
          <p>800 excitatory + 200 inhibitory LIF neurons</p>
          <p>2 OpticFiber inputs (OU noise, 473nm)</p>
          <p>50-channel MultiUnitActivity probe</p>
        </div>
      </div>
      <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 0.78rem; color: var(--text-muted);">
        <strong>Architecture:</strong> <code>dx/dt = f(x) + g(x)u</code> (Control-Affine Neural ODE) · 500×500×500 μm³ volume · Recurrent E/I connectivity (p=0.1) · τ_m=20ms
      </div>
    </div>
  </div>
'''
    
    # Insert before the tabs div
    tabs_marker = '  <!-- Tabs -->'
    if tabs_marker in html:
        html = html.replace(tabs_marker, dataset_section + '\n' + tabs_marker)
        print('  ✅ Inserted dataset description section')
    else:
        print('  ⚠️  Could not find tabs marker')
    
    # ====================================================================
    # 2. PLANT VISUALIZATION — removed. The Plants tab (inject_plants_tab /
    #    update_dashboard.py --plants-only) is now the single source of truth
    #    for plant visualizations; the old static plant_setup.png card was
    #    redundant and could show an outdated plant.
    # ====================================================================

    # ====================================================================
    # 3. PREDICTION PLOTS — embed in relevant tabs
    # ====================================================================
    
    # 3a. CA-NODE baseline prediction in the comparison tab or skip tab
    canode_candidates = [
        'results/sweep_skip_40_10/run_00_h128_l2_lr0.0001_dopri5_noskip_compiled/canode_prediction_dashboard.png',
        'results/sweep_skip_40_10/run_00_h128_l2_lr0.0001_dopri5_noskip_compiled/canode_prediction.png',
        'results/sweep_spectral_40_10/run_00_h128_l2_lr0.0001_dopri5_sa0.0_noskip_compiled/canode_prediction.png',
    ]
    
    canode_img = None
    for c in canode_candidates:
        if os.path.exists(c):
            canode_img = c
            break
    
    if canode_img:
        canode_uri = png_to_data_uri(canode_img)
        canode_card = make_image_card(
            '📈 CA-NODE Baseline — Predicted vs Actual Firing Rates (200ms windows)',
            canode_uri,
            'Channel-level CA-NODE (h=128, l=2, lr=1e-4, dopri5) on test trial. R²=0.9009.'
        )
        # Insert in the Skip Connections tab after the table
        skip_tab_marker = '<!-- ========== Tab: Latent NODE (NEW BEST) ========== -->'
        comparison_marker = '<!-- ========== Tab: Model Comparison ========== -->'
        if comparison_marker in html:
            # Add before the comparison tab
            comparison_pred_section = f'''
      {canode_card}
'''
            # Insert prediction card just before the closing of comparison section's table
            # Find the comparison tab content and add after the table
            comp_start = html.find(comparison_marker)
            # Find the first </table> after the comparison section starts
            table_end = html.find('</table>', comp_start)
            if table_end != -1:
                wrapper_end = html.find('</div>', table_end) 
                if wrapper_end != -1:
                    insert_point = wrapper_end + len('</div>')
                    html = html[:insert_point] + '\n' + comparison_pred_section + html[insert_point:]
                    print(f'  ✅ Embedded CA-NODE prediction plot in Comparison tab ({os.path.getsize(canode_img)//1024} KB)')
    
    # 3b. Causal prediction plot
    causal_path = 'results/causal_z64_pw200_h128_lr3e4/causal_prediction.png'
    if os.path.exists(causal_path):
        causal_uri = png_to_data_uri(causal_path)
        causal_card = make_image_card(
            '📈 Causal Latent CA-NODE — Predicted vs Actual Firing Rates',
            causal_uri,
            'Causal encoder (200ms past) → latent z=64 → ODE → decode. R²=0.8252 on test trial.'
        )
        # Insert in the Causal Latent tab
        causal_tab_marker = '<!-- ========== Tab: Causal Latent CA-NODE (40/10) ========== -->'
        if causal_tab_marker in html:
            # Find the end of the causal latent table
            causal_start = html.find(causal_tab_marker)
            # Find last </table> in this tab section (before the next tab)
            next_tab = html.find('<!-- ==========', causal_start + 10)
            causal_section = html[causal_start:next_tab] if next_tab != -1 else html[causal_start:]
            
            # Find the closing </div> of table-wrapper within the causal section
            tw_end_offset = causal_section.rfind('</div>')
            if tw_end_offset != -1:
                # Insert before the last closing divs of the section
                # Find the second to last </div></div> pattern (end of section div)
                insert_pos = causal_start + causal_section.rfind('</div>\n  </div>')
                html = html[:insert_pos] + '\n' + causal_card + '\n' + html[insert_pos:]
                print(f'  ✅ Embedded causal prediction plot ({os.path.getsize(causal_path)//1024} KB)')
    
    # 3c. Latent NODE prediction plot (if generated)
    latent_path = 'results/sweep_latent_node_40_10/run_05_z64_h256_l2_lr0.0005_dopri5/latent_node_prediction.png'
    if os.path.exists(latent_path):
        latent_uri = png_to_data_uri(latent_path)
        latent_card = make_image_card(
            '🏆 Latent NODE — Predicted vs Actual Firing Rates (Best Model)',
            latent_uri,
            'Non-causal encoder (full trial) → latent z=64 → ODE → decode. R²=0.9387 on test trial.'
        )
        latent_tab_marker = '<!-- ========== Tab: Latent NODE (NEW BEST) ========== -->'
        if latent_tab_marker in html:
            latent_start = html.find(latent_tab_marker)
            next_tab = html.find('<!-- ==========', latent_start + 10)
            latent_section = html[latent_start:next_tab] if next_tab != -1 else html[latent_start:]
            insert_pos = latent_start + latent_section.rfind('</div>\n  </div>')
            html = html[:insert_pos] + '\n' + latent_card + '\n' + html[insert_pos:]
            print(f'  ✅ Embedded latent NODE prediction plot ({os.path.getsize(latent_path)//1024} KB)')
    
    # ====================================================================
    # 4. UPDATE TIMESTAMP
    # ====================================================================
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M EDT')
    html = re.sub(r'Last updated: [\d\-]+ [\d:]+ EDT', f'Last updated: {now_str}', html)
    print(f'  ✅ Updated timestamp to {now_str}')
    
    # ====================================================================
    # 5. WRITE UPDATED DASHBOARD
    # ====================================================================
    with open(dashboard_path, 'w') as f:
        f.write(html)
    
    print(f'\n✅ Dashboard updated: {dashboard_path}')
    print(f'   File size: {os.path.getsize(dashboard_path) // 1024} KB')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Update dashboard.html')
    ap.add_argument('--plants-only', action='store_true',
                    help='Only (re)inject the Plants tab — idempotent, safe to re-run')
    args = ap.parse_args()
    if args.plants_only:
        plants_only()
    else:
        main()

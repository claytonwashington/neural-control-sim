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
    # 2. PLANT VISUALIZATION — add to dataset section or as separate
    # ====================================================================
    plant_path = 'results/plant_setup.png'
    if os.path.exists(plant_path):
        plant_uri = png_to_data_uri(plant_path)
        plant_card = f'''
  <!-- Plant Visualization -->
  <div class="section" style="margin-bottom: 24px;">
    <div style="background: var(--gradient-card); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden;">
      <div style="padding: 14px 18px; border-bottom: 1px solid var(--border); font-size: 0.85rem; font-weight: 600;">🧠 3D Plant Setup — 800E + 200I Neurons, 2 Fibers, 50-ch Probe</div>
      <div style="display: flex; justify-content: center; padding: 8px; background: #0a0e1a;">
        <img src="{plant_uri}" alt="Plant Setup" style="max-width: 700px; width: 100%; cursor: zoom-in;" onclick="openLightbox(this)">
      </div>
    </div>
  </div>
'''
        # Insert after dataset section, before tabs
        html = html.replace(tabs_marker, plant_card + '\n' + tabs_marker)
        print(f'  ✅ Embedded plant visualization ({os.path.getsize(plant_path)//1024} KB)')
    else:
        print(f'  ⚠️  Plant image not found: {plant_path}')
    
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
    main()

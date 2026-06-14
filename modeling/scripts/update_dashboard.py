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




if __name__ == '__main__':
    # The Plants tab is the only thing this script injects; the dashboard
    # pipeline (build_dashboard.py / preflight) invokes it as `--plants-only`.
    # (The old in-place `main()` that embedded the dataset section, a static
    # plant image, and prediction plots was non-idempotent and is gone — those
    # are owned by generate_*/the pipeline now.)
    plants_only()

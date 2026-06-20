# Experiment Lifecycle & Logging Audit Instructions

## Overview

Every experiment must follow a strict lifecycle: **Preflight → Run → Complete → Dashboard**. 
This document describes how to audit that all steps were properly executed.

---

## 1. Preflight Checklist

Before any training/eval run, `preflight start` must be called. Verify:

```bash
# Check MANIFEST.json exists in results dir
cat results/<exp_dir>/MANIFEST.json | python3 -m json.tool

# Verify ideas/modeling.md has the experiment entry
grep "### Experiment <N>" ideas/modeling.md

# Verify branches.md has the cross-reference
grep "<exp_dir>" branches.md
```

**What to look for:**
- [ ] `MANIFEST.json` exists with `preflight_token`, `experiment_name`, `branch`, `data`, `gpu_ids`
- [ ] `ideas/modeling.md` has entry with status `🔄 IN PROGRESS`
- [ ] `branches.md` has a row linking branch → worktree → experiment → results dir

---

## 2. Completion Checklist

After an experiment finishes, `preflight complete` must be called:

```bash
cd /mnt/cbwash2/cleo-worktrees/<worktree>

python -m modeling.scripts.preflight complete \
  --results-dir results/<exp_dir> \
  --status completed \
  --best-r2 <value> \
  --notes "Brief summary of findings"
```

**Status options:** `completed` (beat baseline), `failed`, `baseline` (did not beat)

**Verify after completion:**
```bash
# ideas/modeling.md should show ✅ COMPLETE or ❌ FAILED
grep -A3 "### Experiment <N>" ideas/modeling.md

# branches.md should show result in status column
grep "<exp_dir>" branches.md

# MANIFEST.json should have completion_time and status
cat results/<exp_dir>/MANIFEST.json | python3 -m json.tool | grep -E "status|completion|best_r2"
```

**What to look for:**
- [ ] `ideas/modeling.md` status changed from 🔄 to ✅/❌
- [ ] `branches.md` status column updated with R² and verdict
- [ ] `MANIFEST.json` has `completion_time`, `status`, `best_r2`
- [ ] Git commit exists: `preflight(complete): Exp <N>`

---

## 3. Leaderboard Update

The leaderboard (`/mnt/cbwash2/cleo/results/leaderboard.json`) tracks the best model 
from each experiment type. It is used by `results/dashboard.html`.

**When to update:** After any experiment that produces a new best-in-class model.

**How to verify:**
```bash
# Check if experiment appears in leaderboard
cat /mnt/cbwash2/cleo/results/leaderboard.json | python3 -m json.tool | grep -i "<exp_name>"

# The leaderboard should contain entries for:
# - Acausal models (best R²)
# - Causal models (best R²)  
# - Distilled models
# - EnKF-enhanced models
# - Closed-loop control results (RMSE)
```

**What to look for:**
- [ ] New best models are added to leaderboard.json
- [ ] Old entries are marked with `"is_best_row": false`
- [ ] Dashboard HTML regenerated if leaderboard changed

---

## 4. Dashboard Regeneration

The dashboard is a static HTML file at `results/dashboard.html`.

```bash
# Check dashboard last modified time
ls -la results/dashboard.html

# Verify it references latest experiments  
grep -i "Exp 28\|Exp 29\|bidir.*v2\|H134R" results/dashboard.html
```

**What to look for:**
- [ ] Dashboard updated after leaderboard changes
- [ ] All completed experiments visible in the results table
- [ ] Plots/figures from latest experiments embedded or linked

---

## 5. Full Audit Command

Run this one-liner to audit all experiments at once:

```bash
cd /mnt/cbwash2/cleo-worktrees/bidir-v2-plant

echo "=== EXPERIMENT STATUS AUDIT ==="
echo ""
echo "--- ideas/modeling.md ---"
grep -E "### Experiment [0-9]+" ideas/modeling.md | while read line; do
  exp_num=$(echo "$line" | grep -oE "[0-9]+")
  status=$(grep -A1 "### Experiment ${exp_num}\." ideas/modeling.md | grep "Status" | head -1)
  echo "  Exp ${exp_num}: ${status}"
done

echo ""
echo "--- MANIFEST.json checks ---"
for d in results/*/; do
  if [ -f "${d}MANIFEST.json" ]; then
    name=$(python3 -c "import json; print(json.load(open('${d}MANIFEST.json')).get('experiment_name','?'))" 2>/dev/null)
    status=$(python3 -c "import json; print(json.load(open('${d}MANIFEST.json')).get('status','RUNNING'))" 2>/dev/null)
    echo "  ${d}: ${name} [${status}]"
  fi
done

echo ""
echo "--- Leaderboard entries ---"
python3 -c "
import json
lb = json.load(open('/mnt/cbwash2/cleo/results/leaderboard.json'))
for e in lb:
    best = '🏆' if e.get('is_best_row') else '  '
    print(f\"  {best} {e['model']}: R²={e.get('r2','N/A')}\")
" 2>/dev/null || echo "  (no leaderboard found)"

echo ""
echo "--- Dashboard last updated ---"
ls -la results/dashboard.html 2>/dev/null || echo "  (no dashboard)"
```

---

## 6. Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Exp still shows 🔄 IN PROGRESS | `preflight complete` never called | Run completion command |
| Missing from leaderboard | Leaderboard not updated after completion | Manually add entry to leaderboard.json |
| Dashboard stale | HTML not regenerated | Re-run dashboard generator script |
| MANIFEST.json missing status | Old preflight version | Manually add `"status": "completed"` |
| branches.md missing result | `preflight complete` failed | Manually update the row |

---

## 7. Bidir V2 Experiments Quick Reference

| Exp | Name | Results Dir | Expected Completion Fields |
|-----|------|-------------|---------------------------|
| 28 | Acausal NODE | `bidir_v2_acausal/` | `best_r2=0.956` |
| 29 | Aligned Distillation | `bidir_v2_aligned_distill/` | `best_r2=0.917` |
| 30 | Periodic Re-Encoding | `bidir_v2_periodic_reencode/` | `best_r2=0.935` (K=1) |
| 31 | EnKF Eval | `bidir_v2_enkf/` | `best_r2=0.9998` |
| 32 | Optoclamp MPC vs PI | `bidir_v2_optoclamp/` | `best_rmse=10.7` (PI) |

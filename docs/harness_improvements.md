# Harness Vulnerabilities & Improvement Plan

Audit of the experiment harness (`modeling/scripts/preflight.py` + the dashboard
pipeline: `validation_plots.py`, `generate_experiment_pages.py`,
`update_dashboard*.py`, `build_dashboard.py`, `dashboard_common.py`).
Findings grouped by severity.

**Status:** Phase 1 ✅ (M4, H3, L4, M5), Phase 2 ✅ (H1, H2, M2, M3), and
**M1 ✅ (Option D)** — version `dashboard.html` (the base template the pipeline
mutates) + the JSON facts; the regenerable `experiments/*.html` pages and
`plants/*.png` (the bulk of the bloat) are git-ignored and rebuilt via
`build_dashboard.py`. **L1, L2, L3, L5 ✅** (guarded `torch.load`; accurate
"why no plots" reason on pages; results.json R² sanity check; legacy
`update_dashboard.main()` removed). All findings now resolved.

_Last updated: 2026-06-13._

## High severity

### H1 — `preflight complete` has no mutual exclusion (concurrent completions corrupt state)
`_do_complete` → `_require_dashboard_update` → `_commit_completion` →
`_require_git_push` → `_cleanup_worktree` is a long multi-step mutation of a
**single shared working tree** (`/snel/home/cbwash2/cleo`) with no lock. Two
agents completing at once interleave: git index races, one's dashboard
regeneration clobbers the other's, `git add`/commit/merge collide. With multiple
agents (the reason worktrees exist) this is a live hazard.
**Fix:** wrap `complete` in an exclusive file lock (`fcntl.flock` on
`.git/preflight.lock`); refuse or queue if held.

### H2 — Push gate has no fetch/rebase/retry → fails or diverges under concurrent push
`_require_git_push` computes `origin/modeling-dev..HEAD` and does a bare
`git push`. If origin advanced since, the push is rejected non-fast-forward →
preflight aborts *after* committing locally and marking the experiment complete.
Result: locally "done", remotely not, worktree not cleaned, no recovery path.
**Fix:** `git pull --rebase origin modeling-dev` (or fetch+rebase) before push,
with bounded retry; only mark complete after a successful push, or make the tail
idempotent (see M3).

### H3 — Unescaped experiment metadata injected into HTML/markdown
`generate_experiment_pages.py` and `update_dashboard*.py` interpolate `name`,
`hypothesis`, `notes`, `type` straight into HTML/attributes with f-strings —
**zero escaping**. A name/notes containing `<`, `&`, `"`, or `|` corrupts the
page, the leaderboard row, or the `branches.md` table. `--notes` is free-text, so
this will happen eventually.
**Fix:** `html.escape()` all interpolated text in the page/dashboard generators;
sanitize `--notes`/names before they hit markdown tables in preflight.

## Medium severity

### M1 — Git bloat: every `complete` force-commits the full dashboard + base64 pages
`_commit_completion` force-adds `dashboard.html` (~2.4 MB single blob),
`experiments/` (~2.5 MB each), and `plants/`. `dashboard.html` changes on every
completion, so each experiment adds a fresh multi-MB blob to history; the repo
balloons and clones slow over time.
**Fix options:** treat the dashboard as a build artifact (regenerate on
`modeling-dev` post-merge, don't version it); or move heavy artifacts to git-LFS;
or publish to a dedicated `dashboard-artifacts` branch / GitHub Pages rather than
`modeling-dev` history. _Needs a design decision — changes where the dashboard
"lives"._

### M2 — "Gates before state changes" invariant now violated
`_do_complete`'s docstring promises all gates fire before mutations, but
`_require_dashboard_update` (GATE 2) now writes `val_*.png`, pages,
`dashboard.html`, `leaderboard.json`. If a later step fails, those are left
modified/uncommitted and collide with the next completion.
**Fix:** move dashboard regeneration after the idea/branches/manifest commit, or
stage+commit it atomically with rollback; document GATE 2 as a mutating step.

### M3 — No idempotency / resume for a half-finished `complete`
If `complete` dies after the commit (e.g., push rejected), re-running it crashes
(`git commit` with nothing staged → `check=True`) or double-appends. No
`--resume`/idempotent path.
**Fix:** make each step idempotent (tolerate "nothing to commit", detect an
already-completed manifest, skip an already-merged worktree).

### M4 — "Run from main checkout" rule is documented but not enforced
Nothing checks that `complete` runs on `modeling-dev` in the main checkout. Run
from a stale worktree → executes that worktree's old `preflight.py` (the
`cleosim.pth` makes this silent) and `_cleanup_worktree`'s `git merge <branch>`
runs against the wrong HEAD.
**Fix:** at the top of `_do_complete`, assert `get_repo_root()` is
`/snel/home/cbwash2/cleo` and the current branch is `modeling-dev`; hard-error
with the correct command otherwise.

### M5 — No tests for the harness; `cleosim.pth` hides breakage
Zero tests cover preflight or the dashboard pipeline. A real break (the
`dashboard_common` import that fails under `python script.py`) was nearly shipped
— the global `cleosim.pth` masked it by always resolving `modeling` to main, so
worktree "isolation" is leaky and regressions are invisible until production.
**Fix:** add a smoke test — a `complete`-style path on a fixture results dir plus
a `build_dashboard` run on a tiny fixture — in CI/`pytest`. Invoke dashboard
steps with explicit `sys.executable -m` + `PYTHONPATH` everywhere (some legacy
code still `os.chdir`'s to main).

## Low severity / hardening

- **L1 — `torch.load(weights_only=False)`** (`validation_plots.py:103`, asset
  script): arbitrary-pickle execution on load. Flip to `weights_only=True`
  (+ allowlist) to remove the footgun.
- **L2 — Silent wrong-model default:** missing `model_type` defaults to
  `latent_canode`; a real `canode` then fails to load → metadata-only with no
  explanation. Emit a clear warning instead of a silent skip.
- **L3 — `_require_eval_results` doesn't validate content:** a `results.json`
  with NaN/garbage R² passes and poisons the leaderboard. Add a finite/plausible
  range check.
- **L4 — `--notes`/names unsanitized into markdown tables** (`branches.md`): a
  `|` breaks the table (markdown side of H3).
- **L5 — Legacy `update_dashboard.main()`** still `os.chdir`'s to main and
  re-inserts non-idempotent sections; remove the legacy insertion code now that
  the pipeline owns it.

## Suggested plan (ordered by value/effort)

**Phase 1 — cheap, high-value:** ✅ done
1. ✅ **M4** branch/cwd guard in `_do_complete` (`_assert_main_checkout`).
2. ✅ **H3/L4** `html.escape()` in the generators + `_md_inline` markdown sanitization.
3. ✅ **M5** harness smoke test (`tests/test_harness_smoke.py`).

**Phase 2 — concurrency correctness:** ✅ done
4. ✅ **H1** `flock` around `complete` (`_completion_lock`, on the shared git dir).
5. ✅ **H2 + M3** `_require_git_push` fetch/rebase + retry; `_commit_completion`
   tolerates empty staging (idempotent re-run).
6. ✅ **M2** `_do_complete` reordered — dashboard regen runs after the
   idea/branches/manifest mutations, immediately before the atomic commit.

**Phase 3 — scaling/strategic:** _pending_
7. **M1** artifact strategy (build-artifact vs LFS vs pages) — biggest long-term
   win; needs a design decision.
8. **L1/L2/L3/L5** hardening cleanups.

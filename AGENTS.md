# Cleo: Closed-Loop, Electrophysiology, and Optophysiology Simulation Testbed

Cleo is a Python framework built on top of [Brian 2](https://brian2.readthedocs.io/en/stable/) for simulating realistic closed-loop neuroscience experiments: electrode recordings, optogenetic stimulation, calcium imaging, and real-time control.

## Agent Coordination

> [!IMPORTANT]
> Before starting any work, check these files:
> - `task.md` — active task list with links to detailed subtasks
> - `branches.md` — which agent owns which branch
> - `tasks/` — detailed checklists and implementation plans per task


### Source of Truth

> [!IMPORTANT]
> The **remote NAS** at `/snel/home/cbwash2/cleo/` is the **canonical source of truth** for all code.
> The local path `/Users/claywashington/code/gpu2/cleo/` is a **sync mirror only** — do not edit
> files locally and expect them to persist. All edits must be made on the remote via `ssh gpu2`.

### Workspace Isolation (Git Worktrees)

To prevent file and execution collisions when multiple agents run experiments concurrently, each agent must operate in a dedicated **Git Worktree** checked out to their active branch.

1. **Create the Worktree**: From the main/shared repository directory (`/snel/home/cbwash2/cleo`), create a new worktree directory and branch:
   ```bash
   git worktree add /snel/home/cbwash2/cleo-worktrees/<branch-name> -b feature/<branch-name>
   ```
2. **Register the Worktree**: Update [branches.md](branches.md) and [task.md](task.md) to document the new worktree path and branch assignment.
3. **Shift Workspace**: Conduct all subsequent file edits, command runs (such as sweeps or tests), and git commits inside the dedicated `/snel/home/cbwash2/cleo-worktrees/<branch-name>` directory.

## Environment

- **Python**: >= 3.10
- **Conda env**: `cleo` (core library), `dtmodeling` (modeling/ML work with PyTorch)
- **Activate**: `conda activate cleo` or `conda activate dtmodeling`
- **Infrastructure**: See [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md) for full details on machines, GPUs, MCP servers, and NAS storage.
  - **gpu1**: 8× RTX 2080 Ti (11GB), 48 CPUs. MCP: `gpu1` (filesystem), `gpu1_shell` (commands).
  - **gpu2**: 8× A100 80GB, 128 CPUs. MCP: `gpu2_shell` (active), or SSH.
  - **NAS**: Home dirs shared at `/snel/home/cbwash2/` — code, envs, data visible on both machines.
- **Brian2 limitation**: Single-threaded. For parallel simulations, use `multiprocessing.Pool` with `spawn` context and `brian2.start_scope()` per worker.
- **Running commands via MCP**: Use `bash -lc '...'` wrapper for conda activation:
  ```
  call_mcp_tool(ServerName="gpu1_shell", ToolName="run_command",
                Arguments={"command": "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate dtmodeling && python script.py' 2>&1"})
  ```

## Core Architecture

- **`CLSimulator`**: Orchestrates Brian `Network`, `InterfaceDevice`s, and `IOProcessor`. Uses a Brian `NetworkOperation` for the closed-loop control loop.
- **`InterfaceDevice`**: Base for all simulated hardware.
  - **`Recorder`**: Reads from the network (spikes, LFP, calcium signals).
  - **`Stimulator`**: Applies signals back (light, current).
- **`IOProcessor`** / **`LatencyIOProcessor`**: User-defined control logic with latency modeling.
- **`DeviceInteractionRegistry`**: Manages light-opsin/indicator many-to-many interactions.

## Key Modules

| Module | Purpose |
|--------|---------|
| `cleo.ephys` | Electrophysiology: probes, spike detection, LFP (TKLFP, RWSLFP) |
| `cleo.imaging` | 2P calcium imaging: Scope, GECI sensors (GCaMP, jGCaMP, OGB-1) |
| `cleo.light` | Light sources: OpticFiber (1P), GaussianEllipsoid (2P) |
| `cleo.opto` | Opsin kinetics: 4-state Markov, proportional current |
| `cleo.ioproc` | IO processing with latency and sampling schedules |
| `cleo.coords` | 3D coordinate assignment for neuron groups |
| `cleo.viz` | 3D visualization and video animation |
| `modeling/` | **NOT part of published package.** Digital twin modeling code (plant, data gen, system ID). See `tasks/digital_twin_phase1_plan.md` for details. |

## External Dependencies (tutorials)

Some tutorials require additional packages beyond core Cleo:

- **`ldsctrlest`** (v0.9.0): C++ library with Python bindings for LDS estimation and control. Built from source at `~/lds-ctrl-est` (CMake + vcpkg). Used by: `docs/tutorials/optimal_ctrl.ipynb`
- **`cvxpy`**: Convex optimization. Used by: `optimal_ctrl.ipynb` (MPC section)
- **`lqmpc`**: Linear-quadratic MPC. Used by: `optimal_ctrl.ipynb`
- **`seaborn`**: Plotting. Used by various notebooks.

## Development & Testing

- **Testing**: `conda run -n cleo pytest` (parallel: `pytest -n auto`)
- **Linting**: `ruff`
- **Docs**: Sphinx + ReadTheDocs
- **Tutorials tested with**: `nbmake`

## Conventions

- New recording/stimulation features: subclass `Recorder` or `Stimulator`
- Type hints use `jaxtyping` for arrays
- Multi-device interactions: use `DeviceInteractionRegistry`
- Visualization: use `cleo.viz`
- All Python commands should run within the appropriate conda env (`cleo` or `dtmodeling`)
- **Reproducibility & Data Split**: Always use a 40/10 train/test split (10 test trials out of 50 total trials) when training digital twin models (CA-NODE, GRU, N4SID). Set and pass a fixed random seed (default: `42`, configured centrally in [config.py](file:///snel/home/cbwash2/cleo-worktrees/extended-training/modeling/config.py)) to all random number generators to ensure complete reproducibility of train/val dataset splits and network parameter initialization.
- **Always explain actions and rationale beforehand**: Under no circumstances should you call any tool or execute any shell command without first outputting a message explaining what you are doing, why you are doing it, and what you expect to achieve. Do not perform actions silently.
- **Always use tmux for long-running jobs**: Any model training, evaluation, or benchmarking runs MUST be executed inside a `tmux` session (e.g., using `tmux new-session -d -s <session_name>`). This ensures the processes survive network disconnection and can be monitored easily. This applies to both the primary agent and any subagents spawned. If you delegate tasks to subagents, ensure their prompts explicitly instruct them to run commands inside a `tmux` session.
- **Log all experiment results via the dashboard pipeline**: All model training results, hyperparameter sweeps, and comparison benchmarks MUST be captured by the dashboard pipeline (`preflight complete` / `build_dashboard.py`). See the "Experiment / Sweep" workflow below.

## Results Dashboard

The project maintains a living results dashboard at **`results/dashboard.html`**. This is how we operate:

1. **What's versioned vs. regenerated** (artifact strategy, Option D): we version `results/dashboard.html` (the hand-authored base the pipeline mutates in place) plus the small JSON facts — `leaderboard.json`, `val_metrics.json`, `plants/plants.json`, `experiments/index.json`. The **heavy, fully-regenerable renders are git-ignored**: the per-experiment `results/experiments/*.html` pages and `results/plants/*.png` images. Rebuild them with `conda run -n dtmodeling python -m modeling.scripts.build_dashboard` (`--skip-validation` to reuse plots). A fresh clone has the dashboard + facts but must run `build_dashboard` once to materialize the experiment pages / plant images.
2. **Open locally**: `open results/dashboard.html` (macOS) or view in any browser. After a fresh checkout, run `build_dashboard` first so the linked experiment pages exist. Images are embedded or referenced via relative paths from `results/`.
3. **Structure**: The dashboard has tabs for sweep results, model comparisons, training curves, a chronological experiment log, and a **🧠 Plants** tab (3D renders + metadata for each plant design, the single source of truth for plant visualizations).
4. **Per-experiment validation pages**: Each leaderboard model links (cyan ↗) to a standalone, self-contained page at `results/experiments/<slug>.html` showing **true-vs-inferred firing rates across all channels (heatmaps + top-variance traces) and projected onto the top principal components**, plus `val_metrics.json` (overall / per-channel / per-PC R²). Models with a saved `model.pt` get full plots; the rest are metadata-only.
5. **Auto-generated — do NOT hand-edit**: The Plants tab, validation plots, experiment pages, and leaderboard rows are produced by the dashboard pipeline (`modeling/scripts/build_dashboard.py`, and automatically by `preflight complete` — see the Experiment Lifecycle below). To refresh everything manually: `conda run -n dtmodeling python -m modeling.scripts.build_dashboard` (`--skip-validation` to reuse existing plots). Adding a **new plant design** = add a builder in `modeling/plant.py` + an entry in `modeling/scripts/dashboard_common.py::PLANTS`, then run `build_dashboard` once.
6. **Updating** (when editing by hand is unavoidable): add rows to the sweep table, image cards for curves/prediction plots, a dated log entry, and update the summary cards if a record is broken.
7. **Convention**: Save all plots as `.png` files in `results/` subdirectories. Use relative paths in the HTML.
8. **Sweep scripts**: Training scripts should save a `sweep_summary.json` alongside logs. The dashboard can embed this data directly.

## Agent Workflows

### Before Starting Work
1. Read `task.md` and `branches.md`
2. Create or claim a branch; register it in `branches.md`
3. Create a subtask file in `tasks/` if the task is non-trivial
4. Link the subtask from `task.md`

### Bug Fixing
1. Use codebase investigation to find the source
2. Create minimal test case in `tests/`
3. Fix and format with `ruff`
4. Validate: `conda run -n cleo pytest`

### Feature Addition
1. Identify the right base class and integration points
2. Build following existing patterns in `cleo/`
3. Add unit tests in `tests/` and tutorial in `docs/tutorials/` if appropriate
4. Validate: `conda run -n cleo pytest`
5. Update docstrings

### Experiment / Sweep
1. Write or update the training script with CLI arguments for all hyperparameters
2. Run inside a `tmux` session on the appropriate GPU machine
3. Save model checkpoints, training curves, and prediction plots to `results/<experiment_name>/`
4. Save a `sweep_summary.json` with structured metrics
5. **Update `results/dashboard.html`** with the new results: table rows, plots, log entries, and summary metrics
6. Sync the dashboard to the remote machine via `scp`



### Experiment Lifecycle Protocol

Every experiment follows a strict lifecycle enforced by code:
**Idea → Preflight → Execute → Complete**

#### 1. Ideas Directory (`ideas/`)

All experiments must originate from an entry in one of these files:
- `ideas/modeling.md` — modeling architecture & training experiments
- `ideas/control.md` — controller & closed-loop experiments
- `ideas/project.md` — broader project ideas (organoids, multi-area, etc.)
- `ideas/future.md` — speculative/research directions
- `ideas/architecture_specs.md` — model architecture specifications

Each entry has a status: `Not started` → `🔄 IN PROGRESS` → `✅ COMPLETE` or `❌ DID NOT BEAT BASELINE`.
**No experiment starts without an idea entry. No experiment finishes without annotating its idea entry.**

#### 2. Preflight (`preflight start`)

Before launching ANY training run:

```bash
python -m modeling.scripts.preflight start \
  --idea-file ideas/modeling.md \
  --idea-id 23 \
  --experiment-name "<descriptive name>" \
  --branch feature/<branch-name> \
  --hypothesis "<what you're testing and why>" \
  --data <path-to-data-file> \
  --machine <gpu1|gpu2> \
  --gpu-ids <comma-separated GPU indices> \
  --results-dir results/<unique-dir-name> \
  --model-type <canode|latent_canode|latent_node|gru|n4sid|other> \
  [--past-window 200] \
  [--worktree-path /snel/home/cbwash2/cleo-worktrees/<name>]
```

This enforces:
- The idea entry exists in the specified ideas file and is not already completed
- The results directory does NOT already contain files (prevents overwrites)
- `branches.md` and the idea file mutually cross-reference each other
- A `MANIFEST.json` is written with a preflight token, the `data_file`, and the
  **`model_type`** (this drives the validation-plot inference path at completion —
  set it correctly so `preflight complete` can render the validation plots)
- Everything is committed atomically

Training scripts (`fit_*.py`, `sweep_*.py`) **refuse to start** without a valid `--preflight-token`.

#### 3. Execution

Run training in tmux. The `--preflight-token` argument is required:
```bash
python -m modeling.scripts.fit_latent_canode \
  --preflight-token <TOKEN_FROM_PREFLIGHT> \
  --output-dir results/<dir> ...
```

#### 4. Completion (`preflight complete`)

After training finishes, close the loop:

> **Run `preflight complete` from the MAIN checkout (`/snel/home/cbwash2/cleo`,
> on `modeling-dev`), under the `dtmodeling` env** — NOT from inside the
> experiment's worktree:
> ```bash
> cd /snel/home/cbwash2/cleo
> conda run -n dtmodeling python -m modeling.scripts.preflight complete ...
> ```
> Why: completion merges the feature branch into `modeling-dev`, pushes, and
> regenerates the dashboard — all of which must run against the main checkout. It
> also means completion always executes `modeling-dev`'s copy of the harness
> code, so every experiment is bound by the current rules at merge time
> regardless of when its worktree was branched. (Running it from a stale worktree
> would execute that worktree's old `preflight.py` and break the merge step.) The
> validation step needs torch, hence `dtmodeling`.

```bash
python -m modeling.scripts.preflight complete \
  --results-dir results/<dir> \
  --status completed|failed|baseline \
  --best-r2 0.864 \
  --notes "Closes 35% of causal-acausal gap"
```

This updates:
- The idea entry with `✅ COMPLETE` or `❌` status and results notes
- `branches.md` status column
- `MANIFEST.json` with completion timestamp and metrics
- **Regenerates the dashboard for this experiment**: validation plots
  (`val_*.png` + `val_metrics.json`), its `results/experiments/<slug>.html` page,
  the Plants tab, and the leaderboard row + link — then force-adds and commits
  those artifacts. Validation is best-effort (GRU/N4SID/no-checkpoint → the page
  is metadata-only, completion is not blocked); the page + leaderboard row are
  guaranteed.
- Commits everything atomically

**No experiment is considered finished until `preflight complete` has been run.**





### Autonomous Experiment Workflow

Agents running experiments must follow this exact sequence. Every step is code-enforced.

#### 1. Select Experiment
Pick the next `Not started` entry from `ideas/modeling.md`, `ideas/control.md`, `ideas/project.md`, `ideas/future.md`, or `ideas/architecture_specs.md`.

#### 2. Check GPU Availability
```bash
python -m modeling.scripts.preflight gpu-status
```
This shows which GPUs are free and suggests `--gpu-ids`.

#### 3. Run Preflight
```bash
python -m modeling.scripts.preflight start \
  --idea-file ideas/modeling.md --idea-id 23 \
  --experiment-name "Descriptive Name" \
  --branch feature/branch-name \
  --hypothesis "What you're testing" \
  --data data/training_trials_bidir_v2.h5 \
  --machine gpu1 --gpu-ids 0,1,2,3 \
  --results-dir results/unique_dir_name \
  --model-type latent_canode \
  --worktree-path /snel/home/cbwash2/cleo-worktrees/name
```
Save the printed token for the next step.

#### 4. Launch Training in tmux
All sweep scripts **refuse to run outside tmux**. Use the session name from preflight:
```bash
tmux new-session -d -s exp_23_branch_name \
  'cd /snel/home/cbwash2/cleo-worktrees/name && \
   conda activate dtmodeling && \
   python -m modeling.scripts.sweep_... \
     --preflight-token <TOKEN> ...'
```

#### 5. Monitor Training
```bash
python -m modeling.scripts.preflight monitor \
  --results-dir results/unique_dir_name \
  --poll-interval 60 --timeout 14400
```
This blocks until all runs produce `results.json`, then prints the best result and the exact `preflight complete` command to run.

#### 6. Complete Experiment (Postflight)
Run from the **main checkout** (`/snel/home/cbwash2/cleo`, on `modeling-dev`), under `dtmodeling` — never from the experiment's worktree:
```bash
cd /snel/home/cbwash2/cleo
conda run -n dtmodeling python -m modeling.scripts.preflight complete \
  --results-dir results/unique_dir_name \
  --status completed --best-r2 0.864 \
  --notes "Summary of findings"
```
It enforces these gates:
1. **Eval results exist** — `results.json` must be present (hard)
2. **Dashboard regenerated** — validation plots (best-effort) → experiment page (hard) → Plants tab → leaderboard row (hard); artifacts are force-added and committed
3. **Git push** — auto-pushes to `origin/modeling-dev` (hard)
4. **Worktree cleanup** — merges branch, removes worktree, deletes branch

#### 7. Repeat
Select the next experiment and go to step 2.

### Worktree Lifecycle

- **Creation**: `preflight start --worktree-path ...` creates worktrees
- **Deletion**: `preflight complete` auto-merges and removes worktrees
- **No manual worktrees**: All worktrees must be created via preflight
- **Stale worktrees**: Any worktree not tied to an active experiment should be cleaned up

### Prohibited Patterns

1. **Never read from local file mirrors** (e.g., ~/code/gpu2/cleo/). These are stale. Always use the MCP tools (gpu1, gpu2) or SSH to read files from the actual machines.
2. **Never run training outside tmux**. Sweep scripts enforce this and will refuse to start.
3. **Never reuse a results directory**. Preflight enforces this — each experiment gets a unique results dir.
### Results Digestion Protocol

Every new set of experiment results **MUST** be digested through a structured git commit that updates all tracking artifacts. No results are considered "landed" until this process completes.

> **For preflight-managed experiments, `preflight complete` already does steps 2
> and 5 for you** — it regenerates the leaderboard row, validation plots, and the
> experiment page, and force-adds + commits them. The manual steps below apply to
> results that did **not** go through preflight, or to hand-tweaks (Key Findings,
> Experiment Log narrative) that the pipeline doesn't author. Do not hand-edit the
> auto-generated leaderboard rows / Plants tab / experiment pages.

#### Required Steps

1. **Parse results** from the sweep log/JSON:
   - Extract R², MSE, training time, and any hyperparameters
   - Verify results are plausible (sanity check against known baselines)
   - Flag any suspicious patterns (e.g., identical metrics across configs that should differ)

2. **Update `results/dashboard.html`**:
   - Add/update rows in the relevant sweep tab (sorted by R²)
   - Update the Model Comparison tab if rankings changed
   - Update summary metric cards (Best R², configs tested, etc.)
   - Update the Experiment Log tab with a dated entry
   - Update the Key Findings tab if new conclusions emerged
   - Set the "Last updated" timestamp

3. **Update `ideas/modeling.md`**:
   - Mark the tested idea with status: ✅ TESTED, ❌ Did not beat baseline, or 🔄 IN PROGRESS
   - Include the R² result and comparison to baseline
   - Add any new insights under the idea's notes

4. **Update `task.md`** (local artifact or repo-level):
   - Mark completed items as `[x]`
   - Add new items if the results suggest follow-up experiments

5. **Commit atomically**:
   ```bash
   git add -f results/dashboard.html ideas/modeling.md task.md
   git commit -m 'results(<sweep_name>): <brief summary with best R²>'
   ```
   Example: `results(latent-node): 8/8 complete, best R²=0.9387 (z=64, h256, lr=5e-4)`

6. **Verification checklist** (include in commit message body):
   - [ ] Dashboard has no ⏳ placeholders for completed runs
   - [ ] Model Comparison tab reflects current leaderboard
   - [ ] modeling_ideas.md status markers are accurate
   - [ ] No stale numbers from old train/test splits

#### For Autonomous Agents

When a subagent completes a sweep or training run, it MUST:
1. Parse its own results
2. Produce a structured JSON summary (saved to `results/<sweep_name>/sweep_summary.json`)
3. Report results back to the parent agent with: model name, R², MSE, key hyperparameters
4. The **parent agent** (or a dedicated results-digestion agent) then performs steps 2-6 above

**Never leave results un-digested.** If a sweep finishes but the dashboard hasn't been updated, the results effectively don't exist for the project.

#### Standardization Rules

- **All results MUST use the 40/10 train/test split** on `data/training_trials.h5` with `seed=42`
- Results on other splits (e.g., old 48/2) must be clearly marked as non-comparable and should not appear in the main leaderboard
- The CA-NODE baseline (R²=0.9009) is the reference point for all comparisons

#### Required Metrics Per Model

Every results digestion must capture and report:
1. **R²** (200ms windowed) — primary metric
2. **MSE** (200ms windowed) — secondary metric
3. **Causality** — is the model causal (real-time deployable) or acausal (offline only)?
4. **Inference time** — wall-clock time for a single 200ms prediction step (ms). Mark as "acausal" for models that require the full trial.
5. **Training time** — total elapsed training time
6. **Number of parameters** — for fair capacity comparisons

Training scripts SHOULD measure and log inference time. Add this to `sweep_summary.json`:
```python
# After training, measure inference time
import time
model.eval()
with torch.no_grad():
    dummy_x = torch.randn(1, n_x).to(device)
    dummy_u = torch.randn(1, 200, n_u).to(device)
    # Warmup
    for _ in range(10):
        _ = model(dummy_x, dummy_u)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = model(dummy_x, dummy_u)
    torch.cuda.synchronize()
    inference_time_ms = (time.time() - t0) / 100 * 1000
```


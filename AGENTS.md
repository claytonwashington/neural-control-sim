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
- **Log all experiment results in the HTML dashboard**: All model training results, hyperparameter sweeps, and comparison benchmarks MUST be logged in [`results/dashboard.html`](results/dashboard.html). This is the single source of truth for experiment tracking. See the "Experiment / Sweep" workflow below for details.

## Results Dashboard

The project maintains a living results dashboard at **`results/dashboard.html`**. This is how we operate:

1. **Single source of truth**: All experiment metrics, sweep tables, training curves, and key findings go here.
2. **Open locally**: `open results/dashboard.html` (macOS) or view in any browser. Images are referenced via relative paths from `results/`.
3. **Structure**: The dashboard has tabs for sweep results, model comparisons, training curves, and a chronological experiment log.
4. **Updating**: When you complete an experiment or sweep, update the dashboard by:
   - Adding rows to the sweep results table (sorted by primary metric)
   - Adding image cards for new training curves / prediction plots
   - Appending a log entry with date, summary, and tags
   - Updating the summary metric cards at the top if records are broken
5. **Convention**: Save all plots as `.png` files in `results/` subdirectories. Use relative paths in the HTML.
6. **Sweep scripts**: Training scripts should save a `sweep_summary.json` alongside logs. The dashboard can embed this data directly.

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


### Results Digestion Protocol

Every new set of experiment results **MUST** be digested through a structured git commit that updates all tracking artifacts. No results are considered "landed" until this process completes.

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

3. **Update `docs/modeling_ideas.md`**:
   - Mark the tested idea with status: ✅ TESTED, ❌ Did not beat baseline, or 🔄 IN PROGRESS
   - Include the R² result and comparison to baseline
   - Add any new insights under the idea's notes

4. **Update `task.md`** (local artifact or repo-level):
   - Mark completed items as `[x]`
   - Add new items if the results suggest follow-up experiments

5. **Commit atomically**:
   ```bash
   git add -f results/dashboard.html docs/modeling_ideas.md task.md
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

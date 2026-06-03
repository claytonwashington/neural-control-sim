# Cleo: Closed-Loop, Electrophysiology, and Optophysiology Simulation Testbed

Cleo is a Python framework built on top of [Brian 2](https://brian2.readthedocs.io/en/stable/) for simulating realistic closed-loop neuroscience experiments: electrode recordings, optogenetic stimulation, calcium imaging, and real-time control.

## Agent Coordination

> [!IMPORTANT]
> Before starting any work, check these files:
> - `task.md` — active task list with links to detailed subtasks
> - `branches.md` — which agent owns which branch
> - `tasks/` — detailed checklists and implementation plans per task

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

# Cleo: Closed-Loop, Electrophysiology, and Optophysiology Simulation Testbed

Cleo is a Python framework built on top of [Brian 2](https://brian2.readthedocs.io/en/stable/) for simulating realistic closed-loop neuroscience experiments: electrode recordings, optogenetic stimulation, calcium imaging, and real-time control.

## Agent Coordination

> [!IMPORTANT]
> Before starting any work, check these files:
> - `task.md` — active task list with links to detailed subtasks
> - `branches.md` — which agent owns which branch
> - `tasks/` — detailed checklists and implementation plans per task

## Environment

- **Python**: >= 3.10
- **Conda env**: `cleo` (core library), `dtmodeling` (modeling/ML work with PyTorch)
- **Activate**: `conda activate cleo` or `conda activate dtmodeling`
- **Machine**: gpu2 (128 CPUs, 7 GPUs). Home dir is NAS-shared with gpu1 at `/snel/home/cbwash2`.
- **Brian2 limitation**: Single-threaded. For parallel simulations, use `multiprocessing.Pool` with `spawn` context and `brian2.start_scope()` per worker.

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

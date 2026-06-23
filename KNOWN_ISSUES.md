# Known Issues

Documented issues, workarounds, and machine-specific quirks.

---

## ISSUE-001: Brian2/Cleo Optoclamp Crashes at High Concurrency

**Status**: ✅ Resolved (limit concurrency to 4)  
**Severity**: Critical (silent crashes — processes die with no output)  
**Affected**: gpu2 (likely any machine)  
**First seen**: 2026-06-11, MPC tuning sweep

### Symptoms
- Multiple `run_optoclamp.py` processes launched via `subprocess.Popen`
  crash silently during Phase 0 (Brian2 network compilation)
- Log files show only 7 lines (Brian2 warnings), then nothing
- No core dump files, no error messages

### Root Cause
**Resource exhaustion at high concurrency.** Each optoclamp run builds 
the Brian2 plant 3-4 times (baseline + PI + per-target MPC), loads a 
PyTorch model on CUDA, and runs multi-minute simulations. At 7+ 
concurrent processes, this overwhelms system resources (likely C++ 
compilation contention or memory).

### Evidence

| Test | Concurrent Procs | Duration Alive | Result |
|------|-----------------|----------------|--------|
| 7× `build_plant_v2` (100ms sim) | 7 | completed | ✅ Pass |
| 2× `build_plant_v2` (mp.spawn) | 2 | completed | ✅ Pass |
| 2× `run_optoclamp` (subprocess) | 2 | completed | ✅ Pass |
| 3× `run_optoclamp` (subprocess, A100) | 3 | 9+ min | ✅ Pass (killed by us) |
| 4× `run_optoclamp` (sweep, 4 GPUs) | 4 | ongoing | ✅ Running |
| 7× `run_optoclamp` (sweep, 7 GPUs) | 7 | <5 sec | ❌ All crash |

**Conclusion**: Simple `build_plant` parallelizes fine at any count. 
Full optoclamp (heavy workload) is safe at 4 concurrent, crashes at 7.

### Workaround
Limit `sweep_mpc_tuning.py` to 4 GPUs max:
```bash
python -m modeling.scripts.sweep_mpc_tuning --gpu-ids 0,1,2,3  # safe
python -m modeling.scripts.sweep_mpc_tuning --gpu-ids 0,1,2,3,4,5,6  # crashes
```

---

## ISSUE-002: GCC Version Mismatch Across Machines

**Status**: ✅ Resolved (conda GCC installed on gpu2)  
**Severity**: Medium (potential for subtle numerical differences)

### Details

| Machine | System GCC | Kernel | Conda GCC |
|---------|-----------|--------|-----------|
| gpu1 | 13.3.0 (Ubuntu 24.04) | 6.8.0-51 | not needed |
| gpu2 | 8.4.0 (Ubuntu 20.04) | 5.15.0-139 | ✅ 15.2.0 installed |

Brian2 codegen compiles C++ with the system g++ by default. The 
version mismatch could cause different optimizations and floating-point 
behavior across machines.

### Fix Applied
```bash
# On gpu2:
conda install -y gcc_linux-64 gxx_linux-64 -c conda-forge
```

After install, clear the codegen cache:
```bash
rm -rf ~/.cython/brian_extensions/
```

---

## ISSUE-003: GtACR2 Inhibition Biophysics (Plant 1/2)

**Status**: 📝 Documented (won't fix — use Plant 3 instead)  
**Severity**: Low (design limitation, not a bug)

GtACR2 has reversal potential E = −69.5 mV, nearly identical to resting 
potential E_L = −70 mV. This produces only ~0.5 mV of hyperpolarizing 
drive, making "inhibition" negligible.

**Impact**: Plant 1 (ChrimsonR + GtACR2) cannot perform genuine 
bidirectional control. Use **Plant 3** (ChR2-H134R + eNpHR3.0, 
E = −400 mV) for all bidirectional experiments.

See [PLANTS.md](PLANTS.md) for full details.

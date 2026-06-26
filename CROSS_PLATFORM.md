# Cross-Platform AI Agent Coordination

Two AI tools work on this project concurrently. Each should read this file before starting work to understand the other's context and avoid duplicating effort.

## Active Tools

### Google Antigravity (Gemini)
- **Session title**: "Synchronizing Remote Filesystem Access"
- **Session DB**: `~/.gemini/antigravity/conversations/a1052a22-0cd1-43d8-9688-fee4545876eb.db` (on Mac)
- **Brain/artifacts**: `~/.gemini/antigravity/brain/a1052a22-0cd1-43d8-9688-fee4545876eb/`
- **Implementation plan**: `~/.gemini/antigravity/brain/a1052a22-0cd1-43d8-9688-fee4545876eb/implementation_plan.md`
- **Access method**: MCP servers (gpu1_shell, gpu2_shell) configured in `~/.gemini/antigravity/mcp_config.json`
- **Strengths**: Has full conversation history with experiment evolution from Exp 1 through Exp 62. Wrote the preflight harness, dashboard, and most modeling scripts.

### Claude Code
- **Session search**: Use `mcp__ccd_session_mgmt__search_session_transcripts` to find sessions, or check `~/.claude/projects/-Users-claywashington-code-gpu2/`
- **Memory**: `~/.claude/projects/-Users-claywashington-code-gpu2/memory/`
- **Access method**: Direct SSH to gpu1/gpu2 (no MCP servers needed)
- **Working directory**: `/Users/claywashington/code/gpu2` (local Mac, stale mirror — do NOT edit locally)

## Coordination Protocol

1. **Check before acting**: Before starting work on any experiment, check `branches.md` and `preflight status` to see if the other tool is mid-task.
2. **Shared state files**: Both tools read/write these canonical files:
   - `branches.md` — who owns which branch/worktree
   - `task.md` — active task list
   - `ideas/modeling.md` — experiment ideas and statuses
   - `MANIFEST.json` files in each `results/*/` directory
3. **Reading the other tool's context**:
   - **Antigravity reading Claude**: Check Claude Code memory at `~/.claude/projects/-Users-claywashington-code-gpu2/memory/MEMORY.md`
   - **Claude reading Antigravity**: Read `~/.gemini/antigravity/brain/a1052a22-*/implementation_plan.md` and decode conversation from the SQLite DB (`steps` table, protobuf-encoded payloads, `strings` to extract text)
4. **Don't duplicate**: If one tool is running experiments, the other should work on different tasks (infra, analysis, different experiments).
5. **Always push**: Commit and push ALL changes to origin immediately so the other tool can see them.

## Current State (updated 2026-06-26 09:44 ET)

### Running Experiments
| Exp | Name | Machine | Status | Branch/Worktree |
|-----|------|---------|--------|----------------|
| 62 | Acausal VAE CA-NODE (LFADS-style) | gpu1:GPU0 | 🔄 RUNNING (epoch 0/500) | `feature/acausal-vae-canode` / `acausal-vae-canode` |
| 46 | LFADS No Controller + ext_input | gpu2 | 🔄 RELAUNCHED | `feature/lfads-nocon-gc1` / `lfads-nocon-gc1` |
| 47 | LFADS No Controller, No ext_input | gpu2 | 🔄 RELAUNCHED | `feature/lfads-nocon-noext-gc1` / `lfads-nocon-noext-gc1` |

### Recent Results
| Exp | Name | Result | Notes |
|-----|------|--------|-------|
| 44 | Poisson NLL Spiking CA-NODE (bias init) | ❌ R²=-0.04 | Deterministic autoencoder, no KL loss |
| 61 | Causal VAE CA-NODE | ❌ R²=-0.13 | VAE loss didn't help in causal mode |

### Key Findings
- **Causal encoding of sparse spikes doesn't work** — Exps 37, 38, 44, 61 all failed (R² < 0)
- **Acausal (LFADS-style) encoding is required** — that's what Exp 62 tests
- **Preflight now supports OR-logic dependencies**: `--depends-on "46|47|62"` unblocks when any succeeds
- **ExperimentStatus enum added** to preflight.py

### Worktree Map
```
/mnt/cbwash2/cleo                    → modeling-dev (main checkout)
/mnt/cbwash2/cleo-worktrees/
  acausal-vae-canode/                → feature/acausal-vae-canode (Exp 62) ← NEW
  bidir-v2-plant/                    → feature/bidir-v2-plant (Exps 28-44, 61)
  lfads-nocon-gc1/                   → feature/lfads-nocon-gc1 (Exp 46)
  lfads-nocon-noext-gc1/             → feature/lfads-nocon-noext-gc1 (Exp 47)
  bidir-optoclamp/                   → feature/bidir-optoclamp (Exp 27, old)
  extended-training/                 → feature/extended-training (old)
  hybrid-distillation/               → feature/hybrid-distillation (old)
  idea-enforcement/                  → feature/idea-enforcement (old)
  mpc-latency/                       → feature/mpc-latency (old)
  periodic-reencode/                 → feature/periodic-reencode (old)
  residual-v2/                       → feature/residual-v2 (old)
```

### Next Actions (from Antigravity plan v4)
- Wait for Exp 62 to finish (acausal VAE) — if R² > 0.5, proceed to Aligned Distillation
- Wait for LFADS Exps 46/47 to finish
- LFP data gen script ready but not launched yet
- Spiking control (Exps 52/53) deferred until a spiking model succeeds

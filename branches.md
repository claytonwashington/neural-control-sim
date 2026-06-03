# Active Git Branches & Agents

This file tracks which branch is being used by which agent for what task. **All agents must check this file before starting work, register their active branch here, and keep it updated.**

| Branch Name | Agent / Worktree Path | Task/Goal Description | Status / PR Link | Last Updated |
| ----------- | --------------------- | --------------------- | ---------------- | ------------ |
| `main`      | N/A | Production branch | Active | 2026-05-26 |
| `feature/digital-twin-phase1` | Antigravity | Phase 1 digital twin: Cleo plant, OU data generation, N4SID + control-affine NODE system ID | Completed | 2026-05-28 |
| `feature/skip-connections` | Antigravity (Instance 1) <br> `/snel/home/cbwash2/cleo-worktrees/skip-connections` | Simpler skip connections: linear-only (no MLP) with stronger regularization | Completed (Sweep Done) | 2026-06-03 |
| `feature/spectral-loss` | Antigravity (Instance 2) <br> `/snel/home/cbwash2/cleo-worktrees/spectral-loss` | Frequency-Aware (Spectral) Loss for CA-NODE | Active | 2026-06-03 |
| `feature/multi-rate-integration` | Antigravity (Instance 3) <br> `/snel/home/cbwash2/cleo-worktrees/multi-rate-integration` | Multi-Scale / Multi-Rate Integration for CA-NODE | Active (Sweeping) | 2026-06-03 |
| `feature/extended-training` | Antigravity (Instance 4) <br> `/snel/home/cbwash2/cleo-worktrees/extended-training` | Extended training sweep with cosine decay and learning rate warmup | Active | 2026-06-03 |

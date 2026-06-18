# Compute Infrastructure

## Machines

Both machines share `/home/cbwash2/` via NAS — same conda envs, same code, same data.

### gpu1
- **Hostname:** gpu1 (gpu1.bme.emory.edu)
- **GPUs:** 8× NVIDIA GeForce RTX 2080 Ti (11 GB each)
- **CPUs:** 48 cores
- **Access:** MCP servers `gpu1` (filesystem) and `gpu1_shell` (command execution)
- **Notes:** Good for multi-GPU training or parallel CPU work. Each GPU is 11GB so large models may need model parallelism or smaller batch sizes.

### gpu2
- **Hostname:** gpu2 (gpu2.bme.emory.edu)
- **GPUs:** 8× NVIDIA A100 80GB (or similar large-memory cards)
- **CPUs:** 128 cores
- **Access:** SSH (`ssh gpu2`) or MCP server `gpu2_shell` (active)
- **Notes:** Best for large models, big batch sizes. 80GB per GPU is generous.

## Shared Storage (NAS)

- Home directories are mounted via NAS at `/home/cbwash2/` (also `/snel/home/cbwash2/`). **Primary project location: `/mnt/cbwash2/cleo`**
- Code, conda envs, data files are all shared between gpu1 and gpu2
- **Implication:** A conda env created on one machine works on both
- **Implication:** Files written on one machine are immediately visible on the other

## Key Conda Environment

### `dtmodeling` (Python 3.11)
- **Purpose:** Digital twin modeling pipeline
- **Location:** `~/miniconda3/envs/dtmodeling/`
- **Activation:** `source ~/miniconda3/etc/profile.d/conda.sh && conda activate dtmodeling`
- **Key packages:**
  - PyTorch 2.5.1+cu121
  - torchdiffeq 0.2.5
  - cleo (editable install from ~/cleo)
  - brian2 2.9.0
  - h5py 3.16.0
  - scipy, matplotlib, numpy

## MCP Server Configuration

Config file: `~/.gemini/antigravity/mcp_config.json`

| Server | Type | Machine | Purpose |
|--------|------|---------|--------|
| `gpu1` | filesystem | gpu1 | Read/write files under /mnt/cbwash2 and /snel/home/cbwash2 |
| `gpu1_shell` | shell | gpu1 | Execute commands on gpu1 |
| `gpu2_shell` | shell | gpu2 | Execute commands on gpu2 |

## Running Commands via MCP

To run a command on gpu1 using the shell MCP server:
```
call_mcp_tool(ServerName="gpu1_shell", ToolName="run_command", 
              Arguments={"command": "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate dtmodeling && python script.py'"})
```

**Important:** Use `bash -lc '...'` wrapper for commands that need conda activation.

## Long-Running Jobs (Mandatory tmux Policy)

> [!IMPORTANT]
> **Always** run model training and evaluation runs inside a `tmux` session. Do not run them directly in your interactive shell or background them with `&` without tmux, as they will terminate if your SSH connection drops.
> Using `tmux` ensures your jobs survive disconnects and continue running in the background.

To start a training job in a detached tmux session:
```bash
tmux new-session -d -s jobname "bash -lc 'conda activate dtmodeling && cd /mnt/cbwash2/cleo && python script.py 2>&1 | tee logfile.log'"
```

Useful commands for managing tmux sessions:
- List active sessions: `tmux list-sessions`
- Inspect session output/logs: `tmux capture-pane -t jobname -p`
- Attach to a session: `tmux attach -t jobname`
- Kill a session: `tmux kill-session -t jobname`

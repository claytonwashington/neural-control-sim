#!/bin/bash
# Sequential launch: wait for Exp 40, then run Exp 42, then Exp 43
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lfads-torch-cuda12
cd /mnt/cbwash2/cleo-worktrees/bidir-v2-plant

echo "$(date): Waiting for Exp 40 (lfads_noext) to finish..."
# Wait for the lfads_noext tmux session to end
while tmux has-session -t lfads_noext 2>/dev/null; do
    sleep 60
done
echo "$(date): Exp 40 finished. Waiting 30s for cleanup..."
sleep 30
ray stop 2>/dev/null || true
sleep 10

echo "$(date): === LAUNCHING EXP 42: LFADS No Controller + ext_input ==="
python scripts/run_lfads_nocon_pbt.py \
  --model-config cleo_spiking_nocon_ext \
  --datamodule-config cleo_spiking \
  --tag nocon_ext \
  --num-workers 12 \
  --max-epochs 2000 \
  2>&1 | tee results/lfads_nocon_ext.log

echo "$(date): Exp 42 done. Waiting 30s..."
ray stop 2>/dev/null || true
sleep 30

echo "$(date): === LAUNCHING EXP 43: LFADS No Controller + no ext_input ==="
python scripts/run_lfads_nocon_pbt.py \
  --model-config cleo_spiking_nocon_noext \
  --datamodule-config cleo_spiking_noext \
  --tag nocon_noext \
  --num-workers 12 \
  --max-epochs 2000 \
  2>&1 | tee results/lfads_nocon_noext.log

echo "$(date): All LFADS no-controller experiments complete!"

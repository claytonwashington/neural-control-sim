#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dtmodeling
cd /mnt/cbwash2/cleo-worktrees/bidir-v2-plant

export PREFLIGHT_TOKEN=a69fabfd774736ce4178d2a9e1dea84b51e591ec807a0f30c580b5d2d8c032d9
export PYTHONUNBUFFERED=1

for z in 32 64; do
  for h in 128 256; do
    for lr in 5e-4 1e-3; do
      outdir="results/spiking_canode_p0/z${z}_h${h}_lr${lr}"
      mkdir -p "${outdir}"
      echo "============================================================"
      echo "=== z=${z} h=${h} lr=${lr} ==="
      echo "============================================================"
      python -u -m modeling.scripts.train_spiking_canode \
        --data data/spiking_plant3.h5 \
        --placement 0 \
        --z-dim ${z} \
        --hidden ${h} \
        --lr ${lr} \
        --epochs 200 \
        --output-dir ${outdir} \
        --device cuda \
        2>&1 | tee "${outdir}.log"
    done
  done
done

echo "SWEEP COMPLETE"

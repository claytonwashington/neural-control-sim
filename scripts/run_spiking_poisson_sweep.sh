#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dtmodeling
cd /mnt/cbwash2/cleo-worktrees/bidir-v2-plant

DATA=data/spiking_plant3.h5
BASE=results/spiking_canode_poisson_p0
TOKEN=exp38_poisson_spiking_sweep

for z in 32 64; do
  for h in 128 256; do
    for lr in 5e-4 1e-3; do
      NAME="z${z}_h${h}_lr${lr}"
      OUTDIR="${BASE}/${NAME}"
      echo "============================================================"
      echo "=== z=${z} h=${h} lr=${lr} ==="
      echo "============================================================"
      python -m modeling.scripts.train_spiking_canode_poisson \
        --data $DATA --placement 0 \
        --z-dim $z --hidden $h --lr $lr \
        --epochs 200 --batch-size 64 \
        --output-dir $OUTDIR \
        --device cuda \
        --preflight-token $TOKEN
    done
  done
done

echo "All 8 configs complete!"

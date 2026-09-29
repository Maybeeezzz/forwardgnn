#!/usr/bin/env bash
# Original nodeclass-bp.sh protocol, restricted to GCN for the fused SpMM study.
set -euo pipefail
scriptDir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${scriptDir}/../../src"

for dataset in CitationFull-CiteSeer CitationFull-Cora_ML CitationFull-PubMed Amazon-Photo GitHub; do
  for num_layers in 1 2 3 4; do
    python train_backprop_sparse.py \
      --exp-setting bp-sparse-results --task node-class --model GNN-GCN \
      --dataset "${dataset}" --num-layers "${num_layers}" \
      --num-runs 5 --seed 100 --epochs 1000 --num-hidden 128 \
      --lr 0.001 --val-every 2 --patience 100 --graph-backend sparse-tensor "$@"
  done
done

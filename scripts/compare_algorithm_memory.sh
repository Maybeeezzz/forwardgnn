#!/usr/bin/env bash
# Run from the forwardgnn repository root in the CUDA environment.
set -euo pipefail
python src/train_forward.py \
  --task node-class --model GNN_SingleForward-SAGE --dataset Amazon-Photo \
  --num-layers 2 --num-hidden 128 --num-runs 1 --seed 100 \
  --epochs 3 --lr 0.001 --patience -1 --val-every 1 \
  --append-label none --aug-edge-direction bidirection --temperature 1.0 \
  --gpu 0 --exp-setting algorithm-memory-sf --overwrite-result \
  --memory-analysis --memory-analysis-algorithm \
  --memory-analysis-rules configs/memory/sf.json \
  --memory-analysis-dir results/algorithm-memory-sf
python src/train_backprop.py \
  --task node-class --model GNN-SAGE --dataset Amazon-Photo \
  --num-layers 2 --num-hidden 128 --num-runs 1 --seed 100 \
  --epochs 3 --lr 0.001 --patience -1 --val-every 1 \
  --gpu 0 --exp-setting algorithm-memory-bp --overwrite-result \
  --memory-analysis --memory-analysis-algorithm \
  --memory-analysis-rules configs/memory/bp.json \
  --memory-analysis-dir results/algorithm-memory-bp
python src/memory_algorithm_summary.py \
  --sf results/algorithm-memory-sf --bp results/algorithm-memory-bp \
  --output results/algorithm-memory-comparison

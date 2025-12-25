#!/bin/bash
set -e

echo "=============================="
echo " bsuite — memory_len "
echo "=============================="

pip install -r requirements.txt

python train_meta.py \
  --config configs/bsuite_memory.yaml

python evaluate.py \
  --env bsuite/memory_len

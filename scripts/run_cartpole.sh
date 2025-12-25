#!/bin/bash
set -e

echo "=============================="
echo " CartPole-v1 — PPO + AMRL "
echo "=============================="

pip install -r requirements.txt

python train_meta.py \
  --config configs/cartpole_ppo.yaml

python evaluate.py \
  --env CartPole-v1

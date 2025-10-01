#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=[h200train]
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=berman.ed@northeastern.edu

module load python/3.13.5

eval "$(poetry env activate)"

python3 -m train.train \
  --dataset COCO-SP \
  --architecture GCN \
  --num_layers 12 \
  --skip_connections False \
  --activation_function ReLU \
  --batch_size 64 \
  --batch_norm BatchNorm \
  --num_attention_heads 2 \
  --dropout_rate 0.1 \
  --hidden_size 128 \
  --edge_aggregator GATED \
  --optimizer Adam \
  --lr 0.001 \
  --epochs 750 \
  --weight_decay 0.0 \
  --window_size 4 \
  --receptive_field 5 \
  --save_dir output

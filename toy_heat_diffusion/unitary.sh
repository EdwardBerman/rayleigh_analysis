#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=-m
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li.tao@northeastern.edu

python3 -m toy_heat_diffusion.train --data_dir toy_heat_diffusion/data --start_time 0.0 --train_steps 5 --eval_steps 2 --model unitary  --layers 12 --hidden 128 --epochs 750

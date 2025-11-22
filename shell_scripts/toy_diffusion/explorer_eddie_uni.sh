#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=[heatgraph]
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

python3 -m toy_heat_diffusion.train --data_dir toy_heat_diffusion/data --start_time 0.0 --train_steps 3 --eval_steps 1 --model lie_unitary --layers 12 --hidden 128 --epochs 1000 --act Identity --lr 1e-4


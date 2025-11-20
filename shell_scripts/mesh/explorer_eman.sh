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


python3 -m external.custom_hermes.train dataset=heat backbone=eman

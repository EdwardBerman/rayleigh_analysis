#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=[prep]
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err


module load python/3.13.5

eval "$(poetry env activate)"

python3 -m data_preprocessing.weatherbench_healpix.preprocess_healpix

#!/bin/bash
#SBATCH -J preprocess_weatherbench
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=47:59:59
#SBATCH --partition=short
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

module load python/3.13.5

eval "$(poetry env activate)"

python3 -m data_preprocessing.weatherbench_healpix.preprocess_healpix

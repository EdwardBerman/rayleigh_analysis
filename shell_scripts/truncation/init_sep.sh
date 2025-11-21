#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=[heatgraph]
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li.tao@northeastern.edu

python3 -m truncation.taylor_series_truncation_heat --architecture Uni --data_dir toy_heat_diffusion/data
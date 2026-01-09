#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=eval_wb
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=berman.ed@northeastern.edu

module load python/3.13.5

source weatherbench2/bin/activate

python3 -m external.custom_hermes.eval_weatherbench backbone=hermes paths.forecast=./rollouts/temperature/850/forecasts_EMAN_20260102_200954_633210.zarr
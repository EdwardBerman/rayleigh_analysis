#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=eval_wb
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=berman.ed@northeastern.edu

module load python/3.13.5

eval "$(poetry env activate)"

python3 -m external.custom_hermes.eval_rollout_weatherbench dataset=weatherbench backbone=hermes model_save_path=model_checkpoints/weatherbench_hermes_seed1_model_temp.pt

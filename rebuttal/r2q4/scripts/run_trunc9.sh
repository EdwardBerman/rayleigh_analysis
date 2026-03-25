#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=truncation_reb
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
python3 -m toy_heat_diffusion.train \
    --data_dir toy_heat_diffusion/data \
    --train_steps 3 \
    --eval_steps 1 \
    --model lie_unitary \
    --layers 12 \
    --act Identity \
    --hidden 1 \
    --epochs 200 \
    --lr 0.0001 \
    --batch_size 64 \
    --dropout 0.0 \
    --truncation_level 9 \
    --save_dir rebuttal/r2q4/lie_unitary_trunc9_trial0 \
    --entity_name rayleigh_analysis_gnn \
    --project_name truncation_rebuttal

python3 -m toy_heat_diffusion.train \
    --data_dir toy_heat_diffusion/data \
    --train_steps 3 \
    --eval_steps 1 \
    --model lie_unitary \
    --layers 12 \
    --act Identity \
    --hidden 1 \
    --epochs 200 \
    --lr 0.0001 \
    --batch_size 64 \
    --dropout 0.0 \
    --truncation_level 9 \
    --save_dir rebuttal/r2q4/lie_unitary_trunc9_trial1 \
    --entity_name rayleigh_analysis_gnn \
    --project_name truncation_rebuttal

python3 -m toy_heat_diffusion.train \
    --data_dir toy_heat_diffusion/data \
    --train_steps 3 \
    --eval_steps 1 \
    --model lie_unitary \
    --layers 12 \
    --act Identity \
    --hidden 1 \
    --epochs 200 \
    --lr 0.0001 \
    --batch_size 64 \
    --dropout 0.0 \
    --truncation_level 9 \
    --save_dir rebuttal/r2q4/lie_unitary_trunc9_trial2 \
    --entity_name rayleigh_analysis_gnn \
    --project_name truncation_rebuttal

python3 -m toy_heat_diffusion.train \
    --data_dir toy_heat_diffusion/data \
    --train_steps 3 \
    --eval_steps 1 \
    --model lie_unitary \
    --layers 12 \
    --act Identity \
    --hidden 1 \
    --epochs 200 \
    --lr 0.0001 \
    --batch_size 64 \
    --dropout 0.0 \
    --truncation_level 9 \
    --save_dir rebuttal/r2q4/lie_unitary_trunc9_trial3 \
    --entity_name rayleigh_analysis_gnn \
    --project_name truncation_rebuttal

python3 -m toy_heat_diffusion.train \
    --data_dir toy_heat_diffusion/data \
    --train_steps 3 \
    --eval_steps 1 \
    --model lie_unitary \
    --layers 12 \
    --act Identity \
    --hidden 1 \
    --epochs 200 \
    --lr 0.0001 \
    --batch_size 64 \
    --dropout 0.0 \
    --truncation_level 9 \
    --save_dir rebuttal/r2q4/lie_unitary_trunc9_trial4 \
    --entity_name rayleigh_analysis_gnn \
    --project_name truncation_rebuttal


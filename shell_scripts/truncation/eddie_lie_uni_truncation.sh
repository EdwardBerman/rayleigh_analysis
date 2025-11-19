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

for trunc in 1 2 4 8 16 32 64 128
do
    echo "Running with truncation=$trunc"
    python3 -m experiments.unitary_or_bust.taylor_series_truncation_lrgb \
        --architecture LieUni \
        --truncation $trunc \
        --epochs 100 \
        --optimizer Muon \
        --num_layers 12 \
        --batch_size 64 \
        --dropout_rate 0.1 \
        --lr 0.001 \
        --weight_decay 0.01 \
        --verbose --project lieuni_identity_noedgeagg
done

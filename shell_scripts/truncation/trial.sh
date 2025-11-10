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
#SBATCH --mail-user=li.tao@northeastern.edu

for trunc in 1 2
do
    echo "Running with truncation=$trunc"
    python3 -m experiments.unitary_or_bust.taylor_series_truncation \
        --architecture Uni \
        --truncation $trunc \
        --epochs 1 \
        --verbose --toy
done

for trunc in 1 2 
do
    echo "Running with truncation=$trunc"
    python3 -m experiments.unitary_or_bust.taylor_series_truncation \
        --architecture LieUni \
        --truncation $trunc \
        --epochs 1 \
        --verbose --toy
done
#!/bin/bash


for trunc in 1 2 4 8 16 32 64 128
do
    echo "Running with truncation=$trunc"
    python3 -m flops.flops_counter --architecture Uni --data_dir toy_heat_diffusion/data
        --truncation $trunc \
        --verbose --act ReLU --edge_agg GINE \
        --project uni_gine_relu
done


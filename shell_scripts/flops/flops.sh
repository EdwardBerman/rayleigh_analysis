#!/bin/bash


python3 -m flops.flops_counter --architecture Uni --data_dir toy_heat_diffusion/data
    --verbose --act ReLU --edge_agg GINE \
    --project uni_gine_relu


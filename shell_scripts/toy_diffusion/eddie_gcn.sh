#!/bin/bash

python3 -m toy_heat_diffusion.train --data_dir toy_heat_diffusion/data --start_time 0.0 --train_steps 3 --eval_steps 1 --model gcn --layers 12 --hidden 128 --epochs 50


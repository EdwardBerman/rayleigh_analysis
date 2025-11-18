python3 -m toy_heat_diffusion.train --data_dir toy_heat_diffusion/data --start_time 0.0 --train_steps 5 --eval_steps 2 --model separable_unitary --layers 2 --hidden 16 --epochs 1
python3 -m toy_heat_diffusion.train --data_dir toy_heat_diffusion/data --start_time 0.0 --train_steps 5 --eval_steps 2 --model lie_unitary --layers 2 --hidden 16 --epochs 1
python3 -m toy_heat_diffusion.train --data_dir toy_heat_diffusion/data --start_time 0.0 --train_steps 5 --eval_steps 2 --model gcn --layers 2 --hidden 16 --epochs 1

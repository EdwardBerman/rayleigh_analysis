#!/bin/bash

python -m fourierflow.commands train --trial 0 experiments/airfoil/ffno/4_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/airfoil/ffno/8_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/airfoil/ffno/12_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/airfoil/ffno/16_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/airfoil/ffno/20_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/airfoil/ffno/24_layers/config.yaml

python -m fourierflow.commands train --trial 0 experiments/airfoil/geo-fno/4_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/airfoil/geo-fno/8_layers/config.yaml
python -m fourierflow.commands train --trial 0 experiments/airfoil/geo-fno/12_layers/config.yaml

"""
Evaluate Weatherbench with the native scripts provided through Weatherbench 2
https://weatherbench2.readthedocs.io/en/latest/evaluation.html
"""

import apache_beam
import xarray as xr
from weatherbench2 import config
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam
from weatherbench2.metrics import ACC, MSE

# (5114, 41, 1, 64, 32)
# ('time', 'prediction_timedelta', 'level', 'longitude', 'latitude')

climatology = "data/weatherbench/climatology"
eras5 = "data/weatherbench/eras5"

paths = config.Paths(
    forecast=eras5,
    obs=eras5,
    climatology=climatology,
    output_dir="weatherbench_output"
)

selection = config.Selection(
    variables=[
        'geopotential',
    ],
    levels=[500],
    time_slice=slice("2012-01-01", "2022-12-31"),
)

data_config = config.Data(selection=selection, paths=paths)

eval_configs = {
    'deterministic': config.Eval(
        metrics={
            'mse': MSE(),
            'acc': ACC(climatology=climatology)
        },
    )
}

if __name__ == "__main__":
    pass

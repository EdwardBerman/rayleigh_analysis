"""
Evaluates RMSE and ACC for the Weatherbench 2 dataset using the provided evalution framework.
See documentation at: `https://weatherbench2.readthedocs.io/en/latest/evaluation.html`
"""

import os

import numpy as np

import hydra
import xarray as xr
from matplotlib import pyplot as plt
from weatherbench2 import config
from weatherbench2.evaluation import evaluate_in_memory
from weatherbench2.metrics import ACC, MSE

from external.custom_hermes.utils import create_dataset_loaders

from evaluation.plotting_params import set_rc_params


def plot_visuals(output_dir: str, file_name: str, variable: str, level: int):

    results = xr.open_dataset(output_dir + "/" + file_name + ".nc")

    results = xr.concat(
        [
            results,
            results.sel(metric=['mse']).assign_coords(metric=['rmse']) ** 0.5
        ],
        dim='metric'
    )

    # plot the RMSE
    plt.figure(figsize=(6, 4))
    results[variable].sel(
        metric="rmse",
        level=level,
    ).plot()
    plt.title(f"{variable.upper()} RMSE @ {level} hPa")
    plt.tight_layout()

    rmse_path = os.path.join(
        output_dir, f"{variable}_rmse_{level}.png"
    )
    plt.savefig(rmse_path, dpi=200)
    plt.close()

    # plot the ACC
    plt.figure(figsize=(6, 4))
    results[variable].sel(
        metric="acc",
        level=level,
    ).plot()
    plt.title(f"{variable.upper()} ACC @ {level} hPa")
    plt.tight_layout()

    acc_path = os.path.join(
        output_dir, f"{variable}_acc_{level}.png"
    )
    plt.savefig(acc_path, dpi=200)
    plt.close()

    # save acc and rmse as numpy files

    # This will be useful for comparing different performances later
    acc_values = results[variable].sel(
        metric="acc",
        level=level,
    ).values
    rmse_values = results[variable].sel(
        metric="rmse",
        level=level,
    ).values

    acc_npy_path = os.path.join(
        output_dir, f"{variable}_acc_{level}.npy"
    )
    rmse_npy_path = os.path.join(
        output_dir, f"{variable}_rmse_{level}.npy"
    )
    np.save(acc_npy_path, acc_values)
    np.save(rmse_npy_path, rmse_values)


@hydra.main(version_base=None, config_path="./conf", config_name="eval_weatherbench")
def main(cfg):

    # loading the datasets with hydra confs and this function
    test_ds = create_dataset_loaders(cfg, return_datasets=True)['test']

    forecast_path = cfg.paths.forecast
    climatology_path = cfg.paths.climatology
    era5_path = cfg.paths.obs
    output_dir = cfg.paths.output_dir

    climatology = xr.open_zarr(climatology_path)

    paths = config.Paths(
        forecast=forecast_path,
        obs=era5_path,
        climatology=climatology,
        output_dir=output_dir
    )

    selection = config.Selection(
        variables=[
            test_ds.variable,
        ],
        levels=[test_ds.level],
        time_slice=test_ds.time_slice,
    )

    data_config = config.Data(selection=selection, paths=paths)

    # output file will be called `{eval_name}.nc`
    eval_name = f'deterministic_{test_ds.variable}_{test_ds.level}'

    eval_configs = {
        eval_name: config.Eval(
            metrics={
                'mse': MSE(),
                'acc': ACC(climatology=climatology)
            },
        )
    }

    evaluate_in_memory(data_config, eval_configs)
    plot_visuals(output_dir, eval_name, test_ds.variable, test_ds.level)


if __name__ == "__main__":
    set_rc_params(20)
    main()

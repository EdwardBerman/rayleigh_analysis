import hydra
import xarray as xr
from matplotlib import pyplot as plt
from weatherbench2 import config
from weatherbench2.evaluation import evaluate_in_memory
from weatherbench2.metrics import ACC, MSE

from external.custom_hermes.utils import create_dataset_loaders


@hydra.main(version_base=None, config_path="./conf", config_name="eval_weatherbench")
def main(cfg):

    # loading the datasets with hydra confs and this function
    # this is just to get some metadata
    datasets_dict = create_dataset_loaders(cfg, return_datasets=True)
    train = datasets_dict['train']

    if train.task == 'z500':
        variable = 'geopotential'
        level = 500
    else:
        variable = 'temperature'
        level = 850

    forecast_path = "./rollouts/fivetrajectories"
    climatology_path = "./data/weatherbench/climatology"
    era5_path = "./data/weatherbench/eras5"

    climatology = xr.open_zarr(climatology_path)

    paths = config.Paths(
        forecast=forecast_path,
        obs=era5_path,
        climatology=climatology,
        output_dir="weatherbench_output"
    )

    selection = config.Selection(
        variables=[
            variable,
        ],
        levels=[level],
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

    evaluate_in_memory(data_config, eval_configs)


if __name__ == "__main__":

    main()

    results = xr.open_dataset('./weatherbench_output/deterministic.nc')

    results = xr.concat(
        [
            results,
            results.sel(metric=['mse']).assign_coords(metric=['rmse']) ** 0.5
        ],
        dim='metric'
    )

    results['temperature'].sel(
        metric='acc', level=850).plot()

    plt.show()

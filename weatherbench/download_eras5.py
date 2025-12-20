import xarray as xr
from dask.diagnostics import ProgressBar

era5_path = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)

out_path = "data/weatherbench/eras5"

ds = xr.open_zarr(
    era5_path,
    storage_options={"token": "anon"},
    chunks="auto",
)

ds_subset = (
    ds[["geopotential", "temperature"]]
    .sel(
        level=[500, 850],
        time=slice("2018-01-01", "2022-12-31"),
    )
)

with ProgressBar():
    ds_subset.to_zarr(
        out_path,
        mode="w",
        zarr_version=2,
    )

import xarray as xr
from dask.diagnostics import ProgressBar

climatology_path = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr"

climatology = xr.open_zarr(
    climatology_path,
    storage_options={"token": "anon"},
    chunks="auto"
)

climatology = climatology[["geopotential", "temperature"]].sel(level=[
                                                               500, 850])
out_path = "data/weatherbench/climatology"

with ProgressBar():
    climatology.to_zarr(
        out_path,
        mode="w",
        zarr_version=2,
    )

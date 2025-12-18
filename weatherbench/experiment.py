"""Idea: Make this whole setup compatible with the `custom_hermes` code, so that the training is done with the same loop, so that we can swap out the dataset. Likely we will need to write another evaluation rollout script for this specific mesh. But this feels fairly doable."""

import apache_beam
import weatherbench2
import xarray as xr
import numpy as np


# climatology data for evaluating ACC

climatology_path = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_conservative.zarr"

climatology = xr.open_zarr(
    climatology_path,
    storage_options={"token": "anon"},
    chunks="auto"
)
climatology = climatology[["geopotential", "temperature"]].sel(level=[
                                                               500, 850])


def make_windows(ds, K):
    X = ds.isel(time=slice(None, -K))
    Y = xr.concat(
        [ds.isel(time=slice(i+1, i+1-K or None)) for i in range(K)],
        dim="lead_time",
    )
    Y = Y.assign_coords(lead_time=np.arange(1, K+1))
    return X, Y


era5_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

ds = xr.open_zarr(
    era5_path
    storage_options={"token": "anon"},
    chunks="auto",
)

ds = ds[
    ["geopotential", "temperature"]
].sel(
    level=[500, 850],
    time=slice("2018-01-01", "2022-12-31"),
)

# reshuffle the channels so that time goes first
ds = ds.stack(channel=("variable", "level"))
ds = ds.transpose("time", "channel", "latitude", "longitude")

X, Y = make_windows(ds, 28)

breakpoint()

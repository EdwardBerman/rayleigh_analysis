"""
Downloads and preprocesses data from Weatherbench.
Note that this is *not* the most efficient way to do this, likely the optimal way involves streaming directly from the Google Buckets. 
For the purposes of this research we are only using a somewhat small subset of data, and thus download and preprocess things locally for speed.
"""

import os

import numpy as np
import xarray as xr
from tqdm import tqdm


def select_variable(ds, var, level):
    return (
        ds[var]
        .sel(level=level)
        .transpose("time", "latitude", "longitude")
    )


ERA5_PATH = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)

ds = xr.open_zarr(
    ERA5_PATH,
    storage_options={"token": "anon"},
    chunks={"time": 256},
)


out_root = "data/weatherbench"

os.makedirs(f"{out_root}/z500", exist_ok=True)
os.makedirs(f"{out_root}/t850", exist_ok=True)

z500 = select_variable(ds, "geopotential", 500)
t850 = select_variable(ds, "temperature", 850)

years = np.unique(z500.time.dt.year.values)

for year in tqdm(years[:1]):
    z = z500.sel(time=str(year)).values
    t = t850.sel(time=str(year)).values

    z = z.reshape(z.shape[0], -1)
    t = t.reshape(t.shape[0], -1)

    np.save(f"{out_root}/z500/{year}.npy", z.astype(np.float32))
    np.save(f"{out_root}/t850/{year}.npy", t.astype(np.float32))

import apache_beam
import weatherbench2
import xarray as xr

forecast_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr'
obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
climatology_path = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr'

dataset = xr.open_zarr(forecast_path)

ds_subset = (
    dataset[["geopotential"]]
    .sel(
        level=[500],
        time=slice("2012-01-01", "2022-12-31"),
    )
)
ds_subset = ds_subset['geopotential']

breakpoint()
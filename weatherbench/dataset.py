import apache_beam 
import weatherbench2
import xarray as xr

era5_path = "./data/weatherbench"

era5 = xr.open_zarr(era5_path)

breakpoint()


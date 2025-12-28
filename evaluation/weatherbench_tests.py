import apache_beam
import numpy as np
import weatherbench2
import xarray as xr

from external.custom_hermes.dataset.weatherbench import WeatherBench

# forecast_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr'
# obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
# climatology_path = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr'

# dataset = xr.open_zarr(forecast_path)

# ds_subset = (
#     dataset[["geopotential"]]
#     .sel(
#         level=[500],
#         time=slice("2012-01-01", "2022-12-31"),
#     )
# )
# ds_subset = ds_subset['geopotential']

# breakpoint()

era5_path = "./data/weatherbench/eras5"
mesh_path = "./data/weatherbench/earth_mesh.vtp"

train = WeatherBench(era5_path, mesh_path, task="z500",
                     norm=False, rollout_steps=40, split="train")
test = WeatherBench(era5_path, mesh_path, task="z500", norm=False,
                    rollout_steps=40, split="test", x_mean=train.x_mean, x_std=train.x_std)

preds = np.load("rollouts/weatherbench/test/earth/Hermes/predictions.npy")
num_traj, num_nodes, T = preds.shape

breakpoint()

#         # saving information that will be needed to convert predicitons back to .zarr files
#         self.grid_shape = (ds.latitude.size, ds.longitude.size)
#         self.lat = ds.latitude.values
#         self.lon = ds.longitude.values
#         self.time = ds.time.values
#         self.level = level

nlat, nlon = test.grid_shape

preds = preds.reshape(num_traj, nlat, nlon, T)
preds = preds.transpose(0, 3, 2, 1)   # (time, pred_dt, lon, lat)
preds = preds[:, :, 500, :, :]       # add level dim

pred_timedelta = np.arange(T).astype("timedelta6[h]")  # adjust if needed

# (5114, 41, 1, 64, 32)
# (Pdb) ds_subset.dims
# ('time', 'prediction_timedelta', 'level', 'longitude', 'latitude')
# (Pdb)
# ('time', 'prediction_timedelta', 'level', 'longitude', 'latitude')

da = xr.DataArray(
    preds,
    dims=("time", "prediction_timedelta", "level", "longitude", "latitude"),
    coords={
        "time": test.time,
        "prediction_timedelta": pred_timedelta,
        "level": [500],
        "longitude": test.lon,
        "latitude": test.lat,
    },
    name="geopotential",
)

da = da.chunk({
    "time": 4,
    "prediction_timedelta": 1,
    "level": 1,
    "longitude": nlon,
    "latitude": nlat,
})

da.to_zarr("wb_predictions.zarr", mode="w")

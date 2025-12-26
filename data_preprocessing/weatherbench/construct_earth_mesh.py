"""Save PyVista mesh of the globe based on a specific resolution in weatherbench. """

import numpy as np
import pyvista as pv
import xarray as xr


def latlon_to_xyz(lat, lon, radius=1.0):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


era5_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

ds = xr.open_zarr(
    era5_path,
    storage_options={"token": "anon"},
    chunks="auto",
)

ds = ds[
    ["geopotential", "temperature"]
].sel(
    level=[500, 850],
    time=slice("2018-01-01", "2022-12-31"),
)

lats = ds.latitude.values
lons = ds.longitude.values

Lon, Lat = np.meshgrid(lons, lats)

points = latlon_to_xyz(Lat, Lon).reshape(-1, 3)

nlat, nlon = Lat.shape

faces = []


def vid(i, j):
    return i * nlon + j


for i in range(nlat - 1):
    for j in range(nlon):
        j_next = (j + 1) % nlon

        v0 = vid(i, j)
        v1 = vid(i, j_next)
        v2 = vid(i + 1, j)
        v3 = vid(i + 1, j_next)

        faces.append([3, v0, v1, v2])
        faces.append([3, v1, v3, v2])

faces = np.array(faces, dtype=np.int64).flatten()
mesh = pv.PolyData(points, faces)

mesh.save("data/weatherbench/earth_mesh.vtp")

# visualize it!
# plotter = pv.Plotter()
# plotter.add_mesh(
#     mesh,
#     show_edges=True,
#     opacity=1.0,
# )
# plotter.add_axes()
# plotter.show()

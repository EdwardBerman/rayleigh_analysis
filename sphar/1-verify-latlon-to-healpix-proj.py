import cartopy.crs as ccrs
import cartopy.feature as cfeature
import healpy as hp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from external.custom_hermes.dataset.weatherbench import task_to_variable
from external.custom_hermes.dataset.weatherbench_healpix import (
    WeatherbenchHealpix, get_latlons_for_healpix)


def _hp_marker_size(nside: int, region) -> float:

    base = 80_000 / nside**2
    if region is not None:
        lon_span = region[1] - region[0]
        lat_span = region[3] - region[2]
        zoom = (360.0 * 180.0) / (lon_span * lat_span)
        base *= min(zoom, 400)
    return base


def plot_global_latlon_vs_healpix(
    latlon_data, lat, lon, hp_map, nside,
):

    lon2d, lat2d = np.meshgrid(lon, lat)
    hp_lat, hp_lon = get_latlons_for_healpix(nside)

    vmin = float(np.nanpercentile(latlon_data, 2))
    vmax = float(np.nanpercentile(latlon_data, 98))

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.08)

    ax_ll = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax_hp = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
    ax_cb = fig.add_subplot(gs[2])

    for ax in (ax_ll, ax_hp):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

    im = ax_ll.pcolormesh(
        lon2d, lat2d, latlon_data.T,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="nearest",
    )
    ax_ll.set_title("Lat-lon grid")

    ax_hp.scatter(
        hp_lon, hp_lat, c=hp_map,
        transform=ccrs.PlateCarree(),
        s=_hp_marker_size(nside, None),
        marker="s", cmap="RdBu_r",
        vmin=vmin, vmax=vmax, linewidths=0,
    )
    ax_hp.set_title(f"HEALPix (nside={nside})")

    plt.colorbar(im, cax=ax_cb, extend="both")

    return fig


era5_path = "./data/weatherbench/eras5"
mesh_path = "./data/weatherbench/earth_mesh.vtp"

train = WeatherbenchHealpix(era5_path, mesh_path, task="z500",
                            split="train", nside=32, lmax=20)
t = 0
era5 = xr.open_zarr(era5_path)
variable, level = task_to_variable("z500")
ds_t = era5[variable].sel(level=level, time=train.time_slice).isel(time=t)

latlon_data = ds_t.values
lat = ds_t.latitude.values
lon = ds_t.longitude.values
hp_map = train.healpix_vals[t].numpy()

fig = plot_global_latlon_vs_healpix(
    latlon_data, lat, lon, hp_map, nside=32,
)
fig.savefig("sanity_global.png", dpi=150, bbox_inches="tight")

plt.show()

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import healpy as hp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from external.custom_hermes.dataset.weatherbench import task_to_variable
from external.custom_hermes.dataset.weatherbench_healpix import \
    WeatherbenchHealpix


def plot_latlon_vs_healpix(
    latlon_data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    hp_map: np.ndarray,
    nside: int,
    title: str = "",
    region: tuple = None,
    projection=ccrs.PlateCarree(),
    cmap: str = "RdBu_r",
    figsize: tuple = (16, 6),
    vmin: float = None,
    vmax: float = None,
) -> plt.Figure:

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        sub = latlon_data[np.ix_(lat_mask, lon_mask)]
    else:
        sub = latlon_data

    if vmin is None:
        vmin = float(np.nanpercentile(sub, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(sub, 98))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.08)

    ax_ll = fig.add_subplot(gs[0], projection=projection)
    ax_hp = fig.add_subplot(gs[1], projection=projection)
    ax_cb = fig.add_subplot(gs[2])

    for ax in (ax_ll, ax_hp):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        if region is not None:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                          crs=ccrs.PlateCarree())

    lon2d, lat2d = np.meshgrid(lon, lat)

    im = ax_ll.pcolormesh(
        lon2d, lat2d, latlon_data.T,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
    )
    ax_ll.set_title("Lat-lon grid", fontsize=11)

    npix = hp.nside2npix(nside)
    pix_idx = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix_idx)
    hp_lat = 90.0 - np.degrees(theta)
    hp_lon = np.degrees(phi)
    hp_lon[hp_lon > 180] -= 360

    if region is not None:
        mask = (
            (hp_lon >= lon_min) & (hp_lon <= lon_max) &
            (hp_lat >= lat_min) & (hp_lat <= lat_max)
        )
    else:
        mask = np.ones(npix, dtype=bool)

    ax_hp.scatter(
        hp_lon[mask], hp_lat[mask], c=hp_map[mask],
        transform=ccrs.PlateCarree(),
        s=_hp_marker_size(nside, region),
        marker="s",
        cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0,
    )
    ax_hp.set_title(f"HEALPix grid  (nside={nside})", fontsize=11)

    # ── shared colourbar ──────────────────────────────────────────────────────
    plt.colorbar(im, cax=ax_cb, extend="both")

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    return fig


def _hp_marker_size(nside: int, region) -> float:

    base = 80_000 / nside**2
    if region is not None:
        lon_span = region[1] - region[0]
        lat_span = region[3] - region[2]
        zoom = (360.0 * 180.0) / (lon_span * lat_span)
        base *= min(zoom, 400)
    return base


def plot_latlon(
    latlon_data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str = "",
    region: tuple = None,
    projection=ccrs.PlateCarree(),
    cmap: str = "RdBu_r",
    figsize: tuple = (8, 5),
    vmin: float = None,
    vmax: float = None,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={
                           "projection": projection})

    if region is not None:
        ax.set_extent(list(region), crs=ccrs.PlateCarree())

    lon2d, lat2d = np.meshgrid(lon, lat)

    if vmin is None or vmax is None:
        sub = latlon_data
        if region is not None:
            lon_min, lon_max, lat_min, lat_max = region
            sub = latlon_data[
                np.ix_(
                    (lat >= lat_min) & (lat <= lat_max),
                    (lon >= lon_min) & (lon <= lon_max),
                )
            ]
        vmin = vmin or float(np.nanpercentile(sub, 2))
        vmax = vmax or float(np.nanpercentile(sub, 98))

    im = ax.pcolormesh(
        lon2d, lat2d, latlon_data.T,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05, extend="both")

    if title:
        ax.set_title(title, fontsize=12)
    return fig


def plot_healpix(
    hp_map: np.ndarray,
    nside: int,
    title: str = "",
    region: tuple = None,
    projection=ccrs.PlateCarree(),
    cmap: str = "RdBu_r",
    figsize: tuple = (8, 5),
    vmin: float = None,
    vmax: float = None,
) -> plt.Figure:

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    hp_lat = 90.0 - np.degrees(theta)
    hp_lon = np.degrees(phi)
    hp_lon[hp_lon > 180] -= 360

    if region is not None:
        lon_min, lon_max, lat_min, lat_max = region
        mask = (
            (hp_lon >= lon_min) & (hp_lon <= lon_max) &
            (hp_lat >= lat_min) & (hp_lat <= lat_max)
        )
    else:
        mask = np.ones(npix, dtype=bool)

    if vmin is None:
        vmin = float(np.nanpercentile(hp_map[mask], 2))
    if vmax is None:
        vmax = float(np.nanpercentile(hp_map[mask], 98))

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={
                           "projection": projection})

    if region is not None:
        ax.set_extent(list(region), crs=ccrs.PlateCarree())

    sc = ax.scatter(
        hp_lon[mask], hp_lat[mask], c=hp_map[mask],
        transform=ccrs.PlateCarree(),
        s=_hp_marker_size(nside, region),
        marker="s", cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.05, extend="both")

    if title:
        ax.set_title(title, fontsize=12)
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

fig = plot_latlon_vs_healpix(
    latlon_data, lat, lon,
    hp_map, nside=32,
    title="Z500 – lat-lon vs HEALPix (global)",
)
fig.savefig("sanity_global.png", bbox_inches="tight", dpi=150)

region_europe = (-30, 40, 30, 75)
fig = plot_latlon_vs_healpix(
    latlon_data, lat, lon,
    hp_map, nside=32,
    title="Z500 – lat-lon vs HEALPix (Europe)",
    region=region_europe,
)
fig.savefig("sanity_europe.png", bbox_inches="tight", dpi=150)
plt.show()

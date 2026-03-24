"""
The weatherbench dataset, improved in the following ways:
1. Message passing is done on the Healpix grid instead. We start out with the lat-lon grid, computes its spherical harmonics decomposition, and then samples the function at the Healpix grid. To go back to lat-lon, we inverse that process, moving the Healpix grid to spherical harmonic space, and then sampling at points corresponding to the lat-lon grid. 
2. We will take in two states and predict the next, like GraphCast
3. We will predict the *difference* between time step n and time step n + 1, and not directly predict time step n + 1. 
4. Normalize to zero mean and unit variance.
5. Minimize objective function over 12 steps of forecasts
"""

import os
from typing import Callable, Optional

import healpy as hp
import numpy as np
import pyshtools as pysh
import torch
import xarray as xr
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from external.custom_hermes.dataset.clusterize import clusterize
from external.custom_hermes.dataset.heatwave_pde import (compute_adj_mat,
                                                         compute_edges_dense)
from external.custom_hermes.dataset.weatherbench import task_to_variable


def get_latlons_for_healpix(nside: int):
    npix = hp.nside2npix(nside)
    pix_idx = np.arange(npix)

    theta, phi = hp.pix2ang(nside, pix_idx)
    lat = 90.0 - np.degrees(theta)
    lon = np.degrees(phi)
    lon[lon > 180] -= 360

    return lat, lon


def _fit_one_timestep(data_t, lat_flat, lon_flat, lmax):
    cilm, _ = pysh.expand.SHExpandLSQ(
        data_t.T.flatten(), lat_flat, lon_flat, lmax=lmax)
    return pysh.SHCoeffs.from_array(cilm)


def latlon_to_sphar(lat, lon, data, lmax=20, n_jobs=-1):
    lon = lon % 360
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()

    all_coeffs = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_fit_one_timestep)(data[t], lat_flat, lon_flat, lmax)
        for t in range(data.shape[0])
    )

    return all_coeffs


def healpix_to_sphar(hp_lat, hp_lon, data, lmax=20):
    hp_lon = hp_lon % 360
    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, npix)

    all_coeffs = []
    for t in range(data.shape[0]):
        cilm, _ = pysh.expand.SHExpandLSQ(
            data[t], hp_lat, hp_lon, lmax=lmax
        )
        all_coeffs.append(pysh.SHCoeffs.from_array(cilm))

    return all_coeffs


def sphar_to_healpix(all_coeffs, nside=32):

    npix = hp.nside2npix(nside)
    pix_idx = np.arange(npix)

    theta, phi = hp.pix2ang(nside, pix_idx)

    lat_deg = 90.0 - np.degrees(theta)
    lon_deg = np.degrees(phi)

    hp_maps = []
    for coeffs in all_coeffs:
        cilm = coeffs.to_array()
        lmax = coeffs.lmax
        vals = pysh.expand.MakeGridPoint(
            cilm, lat=lat_deg, lon=lon_deg, lmax=lmax)
        hp_maps.append(vals)

    return np.stack(hp_maps, axis=0), lon_deg, lat_deg


def sphar_to_latlon(all_coeffs, lat, lon):
    lat2d, lon2d = np.meshgrid(lat, lon)
    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()

    maps = []
    for coeffs in all_coeffs:
        cilm = coeffs.to_array()
        vals = pysh.expand.MakeGridPoint(
            cilm,
            lat=lat_flat,
            lon=lon_flat,
            lmax=coeffs.lmax
        )
        maps.append(vals.reshape(240, 121))

    return np.stack(maps, axis=0)


class WeatherbenchHealpix(Dataset):

    def __init__(self,
                 eras5_path: str,
                 mesh_path: str,
                 split: str,
                 task: str,
                 lmax: int,  # the number of terms in spherical harmonic expansion
                 nside: int,  # nsides argument for healpix grid
                 norm: bool = True,
                 input_length: int = 2,
                 train_rollout_steps: int = 12,
                 eval_rollout_steps: int = 40,
                 cluster: bool = True,
                 compute_edges: bool = False,
                 compute_adj: bool = False,
                 max_cluster_size: int = 20,
                 x_mean: Optional[torch.Tensor] = None,
                 x_std: Optional[torch.Tensor] = None,
                 pre_transform: Optional[Callable] = None,
                 cache_dir: Optional[str] = None
                 ):

        super().__init__(None, None, pre_transform)

        assert split in [
            'train', 'test'], "Split must be one of train or test."
        assert task in ['z500', 't850'], "Task must be one of z500 or t850."

        self.eras5_path = eras5_path
        self.mesh_path = mesh_path
        self.split = split
        self.task = task
        self.lmax = lmax
        self.nside = nside
        self.norm = norm
        self.input_length = input_length
        self.train_rollout_steps = train_rollout_steps
        self.eval_rollout_steps = eval_rollout_steps
        self.cluster = cluster
        self.compute_edges = compute_edges
        self.compute_adj = compute_adj
        self.x_mean = x_mean
        self.x_std = x_std
        self.pre_transform = pre_transform
        self.max_cluster_size = max_cluster_size

        # saving the time frames used as train and test
        self.train_slice = slice("2013-01-01", "2019-12-31")
        self.test_slice = slice("2020-01-01", "2020-12-31")

        self.cache_dir = cache_dir

        # note that the pos, face and edge_index are *shared* across all data objects
        self._read_data()

    @staticmethod
    def from_cache(
        era5_path: str,
        mesh_path: str,
        cache_dir: str,
        split: str,
        task: str,
        nside: int = 32,
        lmax: int = 20,
        **kwargs,
    ) -> "WeatherbenchHealpix":

        assert os.path.exists(os.path.join(cache_dir, f"{split}.pt")), \
            f"No cache found at {cache_dir}/{split}.pt. Run WeatherbenchHealpix.preprocess_and_save() first."
        assert os.path.exists(os.path.join(cache_dir, "x_mean.npy")), \
            f"No x_mean.npy found at {cache_dir}. Run WeatherbenchHealpix.preprocess_and_save() first."

        return WeatherbenchHealpix(
            eras5_path=era5_path,
            mesh_path=mesh_path,
            split=split,
            task=task,
            nside=nside,
            lmax=lmax,
            cache_dir=cache_dir,
            **kwargs,
        )

    @staticmethod
    def preprocess_and_save(
        era5_path: str,
        save_dir: str,
        task: str,
        nside: int = 32,
        lmax: int = 20,
    ):
        os.makedirs(save_dir, exist_ok=True)

        era5 = xr.open_zarr(era5_path)
        variable, level = task_to_variable(task)
        ds = era5[variable]

        train_slice = slice("2013-01-01", "2019-12-31")
        test_slice = slice("2020-01-01", "2020-12-31")

        for split, time_slice in [("train", train_slice), ("test", test_slice)]:
            print(f"\nProcessing {split}...")
            ds_split = ds.sel(level=level, time=time_slice)

            orig_lat = ds_split.latitude.values
            orig_lon = ds_split.longitude.values

            print("Computing spherical harmonic coefficients...")
            sphar_coeffs = latlon_to_sphar(
                orig_lat, orig_lon, ds_split.values, lmax)

            print("Projecting to HEALPix...")
            hp_maps, _, _ = sphar_to_healpix(sphar_coeffs, nside)
            healpix_vals = torch.from_numpy(hp_maps).float()

            if split == "train":
                x_mean = healpix_vals.mean().item()
                x_std = healpix_vals.std().item()
                np.save(os.path.join(save_dir, "x_mean.npy"), x_mean)
                np.save(os.path.join(save_dir, "x_std.npy"),  x_std)
                print(f"mean={x_mean:.4f}, std={x_std:.4f}")

            torch.save({
                "healpix_vals": healpix_vals,
            }, os.path.join(save_dir, f"{split}.pt"))
            print(
                f"  Saved to {save_dir}/{split}.pt  shape={healpix_vals.shape}")

    def _load_from_cache(self):

        path = os.path.join(self.cache_dir, f"{self.split}.pt")
        assert os.path.exists(path), \
            f"Cache not found at {path}. Run WeatherbenchHealpix.preprocess_and_save() first."

        print(f"Loading preprocessed data from {path}...")
        data = torch.load(path)

        self.healpix_vals = data["healpix_vals"]

        x_mean = float(np.load(os.path.join(self.cache_dir, "x_mean.npy")))
        x_std = float(np.load(os.path.join(self.cache_dir, "x_std.npy")))

        if self.split == "train":
            self.x_mean = x_mean
            self.x_std = x_std
        else:
            if self.x_mean is None:
                self.x_mean = x_mean
            if self.x_std is None:
                self.x_std = x_std

    def _compute_shared_data(self):

        data = Data(pos=self.healpix_pos, face=self.healpix_faces)
        if self.pre_transform is None:
            return data

        print("WARNING: This operation assumes that the mesh is consistent for all all input data, which is the case for Weatherbench.")

        data = self.pre_transform(data)

        if self.compute_edges:
            data = compute_edges_dense(data)
        if self.compute_edges and self.compute_adj:
            data = compute_adj_mat(data)

        if self.cluster:
            pos_np = self.pos.cpu().numpy().astype(np.float32)
            labels_np, centers_np = clusterize(
                pos_np, max_cluster_size=self.max_cluster_size)
            data.cluster_labels = torch.from_numpy(labels_np).long()
            data.cluster_centers = torch.from_numpy(centers_np).float()

        return data

    def _project_to_healpix(self, ds):

        sphar_coeffs = latlon_to_sphar(
            self.orig_lat, self.orig_lon, ds.values, self.lmax)
        hp_maps, _, _ = sphar_to_healpix(sphar_coeffs, self.nside)

        npix = hp.nside2npix(self.nside)
        x, y, z = hp.pix2vec(self.nside, np.arange(
            npix))

        healpix_vals = torch.from_numpy(
            hp_maps).float()           # (time, num_nodes)
        healpix_pos = torch.from_numpy(
            np.stack([x, y, z], axis=1)).float()  # (num_nodes, 3)

        # triangulated edges via Delaunay on the unit sphere
        # convex hull of points on sphere = triangulation
        hull = ConvexHull(healpix_pos)
        healpix_faces = torch.tensor(
            hull.simplices, dtype=torch.long).T  # (3, num_faces)

        return healpix_vals, healpix_pos, healpix_faces

    def _project_to_latlon(self, data):

        sphar_coeffs = healpix_to_sphar(
            self.hp_lat, self.hp_lon, data, self.lmax)
        latlon_map = sphar_to_latlon(
            sphar_coeffs, self.orig_lat, self.orig_lon)

        return latlon_map

    def _read_data(self):
        era5 = xr.open_zarr(self.eras5_path)
        self.variable, self.level = task_to_variable(self.task)
        ds = era5[self.variable]
        self.time_slice = self.train_slice if self.split == "train" else self.test_slice
        ds = ds.sel(level=self.level, time=self.time_slice)

        self.hp_lat, self.hp_lon = get_latlons_for_healpix(self.nside)
        self.orig_lat = ds.latitude.values
        self.orig_lon = ds.longitude.values
        self.time = ds.time.values

        if self.cache_dir is not None:
            self._load_from_cache()
        else:
            self.healpix_vals, self.healpix_pos, self.healpix_faces = self._project_to_healpix(
                ds)

            if self.split == "train":
                self.x_mean = self.healpix_vals.mean()
                self.x_std = self.healpix_vals.std()
            else:
                assert self.x_mean is not None and self.x_std is not None, \
                    "Test split requires x_mean and x_std from training split"

        # healpix_pos and faces are always computed from nside, not cached
        npix = hp.nside2npix(self.nside)
        x, y, z = hp.pix2vec(self.nside, np.arange(npix))
        self.healpix_pos = torch.from_numpy(
            np.stack([x, y, z], axis=1)).float()
        self.healpix_faces = torch.tensor(ConvexHull(
            self.healpix_pos).simplices, dtype=torch.long).T

        self.shared_data = self._compute_shared_data()

        assert np.all(np.diff(self.hp_lat) <=
                      0), "hp_lat is not strictly descending"
        assert np.all(np.diff(self.orig_lat) >=
                      0), "orig_lat is not strictly ascending"
        assert np.all(np.diff(ds.latitude) >=
                      0), "ds.latitude is not strictly ascending"

    def len(self) -> int:
        """Returns the amount of time steps in this Weatherbench dataset"""
        return self.healpix_vals.shape[0] - self.input_length - self.train_rollout_steps

    # this is used for training
    def get(self, idx: int) -> Data:
        """Builds a Data object on the fly with the shared attributes and the specific time step."""

        assert idx + \
            self.input_length < self.healpix_vals.shape[0], "Window out of range."

        data = Data(**self.shared_data.to_dict())

        # x should be of shape (num_nodes, input_length, node_features), where node_features is just 1
        # y should be of shape (num_nodes, output_length)
        x = self.healpix_vals[idx: idx + self.input_length].T.unsqueeze(-1)
        y = self.healpix_vals[idx + self.input_length: idx +
                              self.input_length + self.train_rollout_steps].T

        if self.norm:
            xnorm = (x - self.x_mean) / self.x_std
            data.x = xnorm
            data.unnormx = x
        else:
            data.x = x

        data.y = y

        return data

    # this is used for evaluation
    def num_trajectories(self):
        return self.healpix_vals.shape[0] - (self.input_length + self.eval_rollout_steps) + 1

    # this is used for evaluation
    def get_trajectory(self, idx: int):

        # data.x should have shape (num_nodes, trajectory_length)
        # this is definitely a little bit sketchy but we are just going to run with it :3
        K = self.input_length
        T = self.eval_rollout_steps
        start = idx

        assert start + K + \
            T <= self.healpix_vals.shape[0], "Trajectory index out of range"

        data = Data(**self.shared_data.to_dict())

        # the shapes just work out this way, go yell at someone else >:(
        data.x = self.healpix_vals[start: start + T + K].T

        if self.norm:
            data.x = (data.x - self.x_mean) / self.x_std

        data.mesh_idx = torch.tensor([0])
        data.sample_idx = torch.tensor([idx])
        data.init_time = self.time[start]

        return data


if __name__ == "__main__":

    era5_path = "./data/weatherbench/eras5"
    mesh_path = "./data/weatherbench/earth_mesh.vtp"

    train = WeatherbenchHealpix(era5_path, mesh_path, task="z500",
                                split="train", nside=32, lmax=20)

    train_loader = DataLoader(
        train,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # this dry run verfifies that the get() and len() behavior of the dataset won't run into indexing issues
    for i, batch in enumerate(tqdm(train_loader, desc="Dry run for Weatherbench Healpix training frames!")):
        pass

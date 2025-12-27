
from typing import Callable, Optional

import pyvista as pv
import torch
import xarray as xr
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def mesh_to_graph(mesh_path: str):
    """Converts a mesh to graph attributes, specifically `pos` and `face`"""
    mesh = pv.read(mesh_path)
    pos = torch.tensor(mesh.points, dtype=torch.float)
    # looks mystical, but it is because: https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.faces
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    face = torch.tensor(faces, dtype=torch.long).T
    return pos, face


def earth_mesh(mesh_path: str):
    mesh = pv.read(mesh_path)
    return mesh


class WeatherBench(Dataset):

    def __init__(self,
                 eras5_path: str,
                 mesh_path: str,
                 split: str,
                 task: str,
                 norm: bool,
                 rollout_steps: int,
                 x_mean: Optional[torch.Tensor] = None,
                 x_std: Optional[torch.Tensor] = None,
                 pre_transform: Optional[Callable] = None):

        super().__init__(None, None, pre_transform)

        assert split in [
            'train', 'test'], "Split must be one of train or test."
        assert task in ['z500', 't850'], "Task must be one of z500 or t850."

        self.eras5_path = eras5_path
        self.mesh_path = mesh_path
        self.split = split
        self.task = task
        self.norm = norm
        self.rollout_steps = rollout_steps
        self.x_mean = x_mean
        self.x_std = x_std
        self.pre_transform = pre_transform
        self.input_length = 1  # hardcoded, take in one step spit out one step

        # note that the pos, face and edge_index are *shared* across all data objects
        self._read_data()

    def _read_data(self):

        self.pos, self.face = mesh_to_graph(self.mesh_path)

        era5 = xr.open_zarr(self.eras5_path)

        if self.task == "z500":
            ds = era5["geopotential"]
            level = 500
        else:
            ds = era5["temperature"]
            level = 850

        if self.split == "train":
            time = slice("2012-01-01", "2018-12-31")
        elif self.split == "test":
            time = slice("2019-01-01", "2022-12-31")

        ds = ds.sel(level=level, time=time)

        ds_nodes = torch.from_numpy(ds.values.reshape(
            ds.shape[0], -1)).float()  # (time, num_nodes)

        self.x = ds_nodes.unsqueeze(-1).unsqueeze(-1)

        if self.pre_transform is not None:
            print("WARNING: This operation here assumes that no pre-transforms use data specific to at time step. As such we compute the pre-transformed values once using a dummy Data() object, and reuse it for streaming.")
            self.shared_data = self.pre_transform(
                Data(pos=self.pos, face=self.face))
        else:
            self.shared_data = Data(pos=self.pos, face=self.face)

        if self.split == "train":
            x_flat = self.x.view(self.x.shape[0], -1)
            self.x_mean = x_flat.mean()
            self.x_std = x_flat.std()
        else:
            assert self.x_mean is not None and self.x_std is not None, \
                "Test split requires `x_mean` and `x_std` from training split"

    def len(self) -> int:
        """Returns the amount of time steps in this Weatherbench dataset"""
        return self.x.shape[0] - 1

    def get(self, idx: int) -> Data:
        """Builds a Data object on the fly with the shared attributes and the specific time step."""

        assert idx + \
            1 < self.x.shape[0], "Cannot obtain the next step of the last step."

        data = Data(**self.shared_data.to_dict())

        if self.norm:
            x_t_norm = (self.x[idx] - self.x_mean) / self.x_std
            data.x = x_t_norm
            data.unnormx = self.x[idx]
        else:
            data.x = self.x[idx]

        data.y = self.x[idx + 1].squeeze(-1)

        return data

    def num_trajectories(self):
        return self.x.shape[0] // self.rollout_steps

    def get_trajectory(self, idx: int):

        T = self.rollout_steps
        start = idx * T

        assert start + T <= self.x.shape[0], "Trajectory index out of range"

        data = Data(**self.shared_data.to_dict())

        # the shapes just work out this way, go yell at someone else >:(
        data.x = self.x[start: start + T].squeeze(-1).squeeze(-1).T
        data.mesh_idx = torch.tensor([0])
        data.sample_idx = torch.tensor([idx])

        if self.norm:
            data.x = (data.x - self.x_mean) / self.x_std

        return data


if __name__ == "__main__":

    era5_path = "./data/weatherbench/eras5"
    mesh_path = "./data/weatherbench/earth_mesh.vtp"

    train = WeatherBench(era5_path, mesh_path, task="z500",
                         norm=False, rollout_steps=40, split="train")

    train_loader = DataLoader(
        train,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    test = WeatherBench(era5_path, mesh_path, task="z500", norm=False,
                        rollout_steps=40, split="test", x_mean=train.x_mean, x_std=train.x_std)

    # this dry run verfifies that the get() and len() behavior of the dataset won't run into indexing issues
    for i, batch in enumerate(tqdm(train_loader, desc="Dry run")):
        pass

import copy
import os.path as osp
from typing import Callable, Optional

import numpy as np
import pyvista as pv
import torch
import xarray as xr
from plyfile import PlyData
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.data.separate import separate

from external.custom_hermes.dataset.clusterize import clusterize


def mesh_to_graph(mesh_path: str):
    """Converts a mesh to graph attributes, specifically `pos` and `face`"""
    mesh = pv.read(mesh_path)
    pos = torch.tensor(mesh.points, dtype=torch.float)
    # looks mystical, but it is because: https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.faces
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    face = torch.tensor(faces, dtype=torch.long)
    return pos, face


class WeatherBench(Dataset):

    def __init__(self, eras5_path: str, mesh_path: str, split: str, task: str, pre_transform: Optional[Callable] = None):

        super().__init__(None, None, pre_transform)

        assert split in [
            'train', 'test'], "Split must be one of train or test."
        assert task in ['z500', 't850'], "Task must be one of z500 or t850."

        self.eras5_path = eras5_path
        self.mesh_path = mesh_path
        self.split = split
        self.task = task
        self.pre_transform = pre_transform

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
            self.shared_data = None

    def len(self) -> int:
        """Returns the amount of time steps in this Weatherbench dataset"""
        return self.x.shape[0]

    def get(self, idx: int) -> Data:
        """Builds a Data object on the fly with the shared attributes and the specific time step."""

        assert idx + 1 < self.len(), "Cannot obtain the next step of the last step."

        data = Data(**self.shared_data.to_dict())

        data.x = self.x[idx]
        data.y = self.x[idx + 1].squeeze(-1)
                
        return data


if __name__ == "__main__":

    era5_path = "./data/weatherbench/eras5"
    mesh_path = "./data/weatherbench/earth_mesh.vtp"

    train = WeatherBench(era5_path, mesh_path, split="train")
    test = WeatherBench(era5_path, mesh_path, split="test")

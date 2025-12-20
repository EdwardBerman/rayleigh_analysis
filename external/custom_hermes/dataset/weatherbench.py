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
    """Converts a mesh to graph attributes, specifically `pos`, `face` and `edge_index`"""
    mesh = pv.read(mesh_path)
    pos = torch.tensor(mesh.points, dtype=torch.float)
    # looks mystical, but it is because: https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.faces
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    face = torch.tensor(faces, dtype=torch.long)
    edges = set()
    for v0, v1, v2 in faces:
        edges.update({
            (v0, v1), (v1, v0),
            (v1, v2), (v2, v1),
            (v2, v0), (v0, v2)
        })
    edges_array = np.array(list(edges), dtype=np.int64)
    edge_index = torch.from_numpy(edges_array).T
    return pos, face, edge_index


class WeatherBench(Dataset):

    def __init__(self, eras5_path: str, mesh_path: str, split: str, pre_transform: Optional[Callable] = None):

        super().__init__(None, None, pre_transform)

        assert split in [
            'train', 'test'], "Split must be one of train or test."

        self.eras5_path = eras5_path
        self.mesh_path = mesh_path
        self.split = split
        self.pre_transform = pre_transform

        # note that the pos, face and edge_index are *shared* across all data objects
        self.pos, self.face, self.edge_index, self.x = self._read_data()

    def _read_data(self):

        pos, face, edge_index = mesh_to_graph(self.mesh_path)

        era5 = xr.open_zarr(self.eras5_path)

        z500 = era5["geopotential"]
        t850 = era5["temperature"]

        if self.split == "train":
            time = slice("2012-01-01", "2018-12-31")
        elif self.split == "test":
            time = slice("2019-01-01", "2022-12-31")

        z500 = z500.sel(level=500, time=time)
        t850 = t850.sel(level=850, time=time)

        z500_nodes = torch.from_numpy(z500.values.reshape(
            z500.shape[0], -1)).float()  # (time, num_nodes)
        t850_nodes = torch.from_numpy(t850.values.reshape(
            t850.shape[0], -1)).float()  # (time, num_nodes)

        x = torch.stack([z500_nodes, t850_nodes],
                        dim=-1)  # (time, num_nodes, 2)

        return pos, face, edge_index, x

    def len(self) -> int:
        """Returns the amount of time steps in this Weatherbench dataset"""
        return self.x.shape[0]

    def get(self, idx: int) -> Data:
        """Builds a Data object on the fly with the shared attributes and the specific time step."""
        assert len(self) > idx + \
            1, "Cannot obtain the next step of the last step."
        data = Data(x=self.x[idx], y=self.x[idx+1], pos=self.pos,
                    face=self.face, edge_index=self.edge_index)

        # note that this is technically post-transform, but for memory efficiency and since we are doing data creation on the fly, the pretransform is applied here:
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data


if __name__ == "__main__":

    era5_path = "./data/weatherbench/eras5"
    mesh_path = "./data/weatherbench/earth_mesh.vtp"

    train = WeatherBench(era5_path, mesh_path, split="train")
    test = WeatherBench(era5_path, mesh_path, split="test")

    breakpoint()

import copy
import enum
import os.path as osp
from typing import Callable, Optional

import h5py
import numpy as np
import torch
import trimesh
from torch_geometric.data import Data, Dataset
from torch_geometric.data.separate import separate

from external.custom_hermes.dataset.clusterize import clusterize

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


class FlagSimpleDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        maxT=401,
        input_length=5,
        output_order=3,
        cluster: bool = True,
        max_cluster_size: int = 20,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        """
        :param str split: choose between 'train', 'valid', 'test'
        """

        splits = ["train", "valid", "test"]
        assert split in splits

        self.split = split
        if split in ["valid", "test"]:
            self.num_samples = 5
        else:
            self.num_samples = 40
        self.maxT = maxT

        self.input_length = input_length
        self.output_order = output_order

        self.cluster = cluster
        self.max_cluster_size = max_cluster_size

        super().__init__(root, transform, pre_transform, pre_filter=None)

    @property
    def raw_file_names(self) -> str:
        return [f"{self.split}/{i}.h5" for i in range(self.num_samples)]

    @property
    def processed_file_names(self):
        return [
            f"{self.split}_{idx}.pt"
            for idx in range(self.num_samples * (self.maxT - 1))
        ]

    def process(self):
        idx = 0
        for f in self.raw_file_names:
            path_data = self._read_data(osp.join(self.raw_dir, f))

            # Label for which sample
            sample_idx = int(f.rsplit("/", 1)[1].split(".")[0])
            for data in path_data:
                data.sample_idx = torch.tensor([sample_idx], dtype=torch.long)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f"{self.split}_{idx}.pt"))
                idx += 1

    def _read_data(self, path):
        # Load file
        file_data = dict()
        with h5py.File(path, "r") as f:
            for k, v in f.items():
                file_data[k] = v[:]

        data_list = []

        # Compute velocity and target first
        velocity = np.diff(file_data["world_pos"], n=self.input_length - 1, axis=0)
        target = np.diff(file_data["world_pos"], n=self.output_order, axis=0)

        if self.cluster:
            print(f"Clustering with max cluster size {self.max_cluster_size}...")

        for i in range(
            self.input_length - 1,
            self.maxT,
        ):
            # Compute node_attr
            node_type = np.eye(NodeType.SIZE)[file_data["node_type"][i, ...]].squeeze(
                -2
            )
            node_attr = np.concatenate(
                [velocity[i - self.input_length + 1], node_type], axis=-1
            )
            # Let node_attr be trivial reps
            node_attr = torch.from_numpy(node_attr).float().unsqueeze(-1)

            # Compute edge attributes
            # Get edge_index using trimesh
            mesh = trimesh.Trimesh(
                vertices=file_data["world_pos"][i],
                faces=file_data["cells"][i],
                process=False,
                validate=False,
            )
            row, col = mesh.edges.T

            relative_world_pos = (
                file_data["world_pos"][i, row, :] - file_data["world_pos"][i, col, :]
            )
            relative_mesh_pos = (
                file_data["mesh_pos"][i, row, :] - file_data["mesh_pos"][i, col, :]
            )

            edge_attr = np.concatenate(
                [
                    relative_world_pos,
                    np.linalg.norm(relative_world_pos, axis=-1, keepdims=True),
                    relative_mesh_pos,
                    np.linalg.norm(relative_mesh_pos, axis=-1, keepdims=True),
                ],
                axis=-1,
            )
            edge_attr = torch.from_numpy(edge_attr)

            pos = torch.from_numpy(file_data["world_pos"][i, ...])
            face = torch.from_numpy(file_data["cells"][i, ...].T).long()

            y = torch.from_numpy(target[i - self.output_order + 1])

            data = Data(pos=pos, face=face, x=node_attr, edge_attr=edge_attr, y=y)

            if self.cluster:
                pos_np = data.pos.cpu().numpy().astype(np.float32)   # [N, 3]
                labels_np, centers_np = clusterize(
                    pos_np, max_cluster_size=self.max_cluster_size)

                cluster_labels = torch.from_numpy(labels_np).long()    # [N]
                cluster_centers = torch.from_numpy(centers_np).float()

                data.cluster_labels = cluster_labels
                data.cluster_centers = cluster_centers

            data_list.append(data)

        return data_list

    def len(self) -> int:
        # Num meshes * num_samples * num_time_windows
        n = self.num_samples * (self.maxT - 1)
        return n

    def get(self, idx: int) -> Data:
        data = torch.load(osp.join(self.processed_dir, f"{self.split}_{idx}.pt"), weights_only=False)
        return data

    def num_trajectories(self):
        return self.num_samples

    def get_trajectory(self, idx):
        n_trajs = self.num_trajectories()

        if not hasattr(self, "_traj_list") or self._traj_list is None:
            self._traj_list = n_trajs * [None]
        elif self._traj_list[idx] is not None:
            return copy.copy(self._traj_list[idx])

        assert idx < n_trajs, "incorrect idx for trajectory"

        traj = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        traj.x = traj.u

        self._traj_list[idx] = copy.copy(traj)

        return traj

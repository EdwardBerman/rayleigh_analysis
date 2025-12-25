import os
import random

import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from hydra.utils import instantiate
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from torch_geometric.loader import DataLoader

from external.custom_hermes.transform.edge_features import empty_edge_attr
from external.custom_hermes.transform.simple_geometry import SimpleGeometry
from external.custom_hermes.transform.vector_normals import \
    compute_vertex_normals


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def prepare_batch_fn(key="y"):
    def prepare_batch(batch, device, non_blocking=False):
        data = batch.to(device)
        return data, data[key].to(device)

    return prepare_batch


def create_dataset_loaders(cfg, return_datasets=False):
    print("Creating datasets")

    if cfg.dataset.name == "FAUST":
        pre_tf = T.Compose(
            [compute_vertex_normals, empty_edge_attr, SimpleGeometry()])
        splits = ["train", "test", "test_gauge"]
    elif any(cfg.dataset.name.startswith(s) for s in ["Heat", "Wave", "Cahn-Hilliard"]):
        pre_tf = T.Compose(
            [compute_vertex_normals, empty_edge_attr, SimpleGeometry()])
        splits = ["train", "test_time", "test_init", "test_mesh"]
    elif cfg.dataset.name.startswith("Other"):
        pre_tf = T.Compose(
            [compute_vertex_normals, empty_edge_attr, SimpleGeometry()])
        splits = ["train", "test_time", "test_init"]
    elif cfg.dataset.name.startswith("Objects"):
        pre_tf = T.Compose(
            [compute_vertex_normals, empty_edge_attr, SimpleGeometry()])
        splits = ["train", "test"]
    elif cfg.dataset.name.startswith("weatherbench"):
        pre_tf = T.Compose(
            [compute_vertex_normals, empty_edge_attr, SimpleGeometry()])
        splits = ["train", "test"]
    else:
        raise NotImplementedError(
            f"Incorrect cfg.dataset.name {cfg.dataset.name}")

    out_dict = {}

    if cfg.dataset.name.startswith("weatherbench"):
        # this case is special because the train statistics need to go to the test dataset
        train_ds = instantiate(
            cfg.dataset.cls, split='train', pre_transform=pre_tf)

        test_ds = instantiate(
            cfg.dataset.cls,
            split="test",
            x_mean=train_ds.x_mean,
            x_std=train_ds.x_std,
            pre_transform=pre_tf
        )

        if return_datasets:
            out_dict["train"] = train_ds
            out_dict["test"] = test_ds
        else:
            out_dict['train'] = DataLoader(
                train_ds,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                pin_memory=True,
            )
            out_dict['test'] = DataLoader(
                test_ds,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                pin_memory=True,
            )
        return out_dict

    for split in splits:
        train = split == "train"

        if any(cfg.dataset.name.startswith(prefix) for prefix in ["FAUST"]):
            dataset = instantiate(
                cfg.dataset.cls, train=train, pre_transform=pre_tf)
        else:
            dataset = instantiate(
                cfg.dataset.cls, split=split, pre_transform=pre_tf)

        if any(
            cfg.dataset.name.startswith(s) for s in ["Heat", "Wave", "Cahn-Hilliard"]
        ):
            print(
                f"[{split}] Len: {len(dataset)}, Num nodes: {dataset._data.num_nodes}"
            )
        else:
            print(
                f"[{split}] Len: {len(dataset)}, Num nodes: {dataset[0].num_nodes}")

        if return_datasets:
            out_dict[split] = dataset

        else:
            out_dict[split] = DataLoader(
                dataset,
                batch_size=cfg.train.batch_size,
                shuffle=train,
                pin_memory=True,
            )

    return out_dict


class GaugeInvarianceNLLLoss(Metric):
    """
    Custom metric to compute difference in NLLLoss between original dataset and random gauge-transformed dataset
    """

    @reinit__is_reduced
    def reset(self):
        self._gauge_error = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_orig, y_t, y = output[0].detach(
        ), output[1].detach(), output[2].detach()

        loss_orig = F.nll_loss(y_orig, y, reduction="none")
        loss_t = F.nll_loss(y_t, y, reduction="none")

        self._gauge_error += torch.abs(loss_orig - loss_t).sum()

        self._num_examples += loss_orig.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "GaugeInvarianceNLLLoss must have at least one example before it can be computed."
            )
        return self._gauge_error.item() / self._num_examples


def rotate_mesh_video(
    mesh,
    scalars,
    dataset_name,
    name,
    save_path,
    n_frames: int = 180,
    framerate: int = 30,
):
    """
    Rotate a mesh around the global z-axis (z is vertical in the image)
    and save the result as a video.

    Parameters
    ----------
    mesh : pv.DataSet
        PyVista mesh.
    scalars : array-like
        Point-wise scalar values to color the mesh with.
    dataset_name : str
        Used to set color limits, like in screenshot_mesh.
    name : str
        Unused here but kept for API symmetry with screenshot_mesh.
    save_path : str
        Path to output video file (e.g. 'spin.mp4' or 'spin.gif').
    n_frames : int
        Number of frames for a full 360° rotation.
    framerate : int
        Video framerate.
    """
    pl = pv.Plotter(off_screen=True)
    pl.set_background("white")

    # Attach scalars
    mesh.point_data["c"] = scalars
    mesh.set_active_scalars("c")

    # Same clim logic as screenshot_mesh
    if dataset_name == "Wave":
        clim = [-1, 1]
    else:
        clim = [0, 1]

    pl.add_mesh(mesh, clim=clim)
    pl.remove_scalar_bar()

    center = np.asarray(mesh.center)
    radius = float(mesh.length) if mesh.length != 0 else 1.0

    distance = 2.5 * radius
    camera_position = [
        (center[0] + distance, center[1],
         center[2] + 0.25 * radius),  # position
        # focal point
        (center[0], center[1], center[2]),
        # view-up (z-axis)
        (0.0, 0.0, 1.0),
    ]
    pl.camera_position = camera_position

    pl.open_movie(save_path, framerate=framerate)

    pl.show(auto_close=False, interactive=False)
    az_step = 360.0 / n_frames

    for _ in range(n_frames):
        pl.camera.azimuth += az_step    # <-- FIXED: no parentheses, just increment
        pl.render()
        pl.write_frame()

    pl.close()

def plot_mesh_weather(mesh, scalars):
    p = pvqt.BackgroundPlotter(title=dataset_name, auto_update=True)

    mesh.point_data["c"] = scalars
    mesh.set_active_scalars("c")

    print(scalars.min().item(), scalars.max().item())
    clim = [scalars.min().item(), scalars.max().item()]

    p.add_mesh(mesh, clim=clim)
    p.view_xy(True)

    return p

def screenshot_mesh_weather(mesh, scalars, save_path):
    pl = pv.Plotter(off_screen=True)
    pl.set_background("white")

    mesh.point_data["c"] = scalars
    mesh.set_active_scalars("c")

    print(scalars.min().item(), scalars.max().item())
    clim = [scalars.min().item(), scalars.max().item()]

    pl.add_mesh(mesh, clim=clim)
    pl.remove_scalar_bar()

    pl.show(screenshot=save_path)
    pl.close()

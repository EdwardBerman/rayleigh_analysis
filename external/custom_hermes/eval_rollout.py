import copy
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pyvista as pv
import torch
from torch_geometric.utils import degree
from hydra.utils import instantiate
from pyvista import examples

from external.hermes.src.data.pde.utils import screenshot_mesh
from external.custom_hermes.utils import create_dataset_loaders, rotate_mesh_video

import matplotlib.pyplot as plt
from matplotlib import rc

def set_rc_params(fontsize=None):
    '''
    Set figure parameters
    '''

    if fontsize is None:
        fontsize = 16
    else:
        fontsize = int(fontsize)

    rc('font', **{'family': 'serif'})
    rc('text', usetex=False)

    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size': 6})
    plt.rcParams.update({'ytick.direction': 'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

set_rc_params(10)

objects = {
    "armadillo": examples.download_armadillo(),
    "bunny_coarse": examples.download_bunny_coarse(),
    "bunny": examples.download_bunny_coarse(),
    "lucy": examples.download_lucy(),
    "sphere": pv.Sphere(),
    "spider": examples.download_spider(),
    "urn": examples.download_urn(),
    "woman": examples.download_woman(),
    "supertoroid": pv.ParametricSuperToroid(n1=0.5),
    "ellipsoid": pv.ParametricEllipsoid(2, 1, 1),
}
decimate_ratio = {
    "armadillo": 0.99,
    "bunny_coarse": 0.0,
    "bunny": 0.0,
    "lucy": 0.95,
    "sphere": 0.0,
    "spider": 0.0,
    "urn": 0.98,
    "woman": 0.98,
    "supertoroid": 0.7,
    "ellipsoid": 0.8,
}

pv.set_plot_theme("paraview")

def get_mesh(name):
    mesh = objects[name]

    mesh = mesh.decimate(decimate_ratio[name])
    _ = mesh.clean(inplace=True)

    return mesh

@hydra.main(version_base=None, config_path="./conf", config_name="eval_rollout")
def main(cfg):
    datasets_dict = create_dataset_loaders(cfg, return_datasets=True)

    backbone = instantiate(cfg.backbone.net).to(cfg.device)
    model = instantiate(cfg.model, backbone=backbone).to(cfg.device)

    model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))
    model.eval()

    loss_fn = instantiate(cfg.loss)

    def eval_step(dataset):
        results = {
            "losses": defaultdict(list),
            "predictions": defaultdict(list),
            "ground_truth": defaultdict(list),
            "predicted_rayleigh_quotients": defaultdict(list),
            "true_rayleigh_quotients": defaultdict(list),
        }

        model.eval()
        for idx in range(dataset.num_trajectories()):
            data = dataset.get_trajectory(idx)
            data = data.to(cfg.device)

            mesh_idx = data.mesh_idx.item()
            sample_idx = data.sample_idx.item()
            values = copy.copy(data.x)

            # Sketchy asf over here
            edge_index = data.edge_index.to(values.device).long()
            src, dst = edge_index[0], edge_index[1]
            N = values.shape[0]
            deg_in = degree(dst, num_nodes=N, dtype=values.dtype).clamp(min=1.0)
            inv_sqrt_deg = deg_in.rsqrt().view(N, 1)

            def norm_sqrt_deg(x: torch.Tensor) -> torch.Tensor:
                return x * inv_sqrt_deg


            all_preds = []
            all_losses = []
            all_gts = []

            traj_true_rq = []
            traj_pred_rq = []

            with torch.no_grad():
                data.x = values[:, 0 : dataset.input_length][..., None]

                for t in range(dataset.input_length, values.shape[1]):
                    y = values[:, t].unsqueeze(-1)

                    all_gts.append(y.squeeze().detach().cpu().numpy())

                    y_pred = model(data)

                    all_preds.append(y_pred.squeeze().detach().cpu().numpy())

                    loss = loss_fn(y_pred, y)
                    all_losses.append(loss.item())

                    # Sketchy over here
                    y_norm      = norm_sqrt_deg(y)       
                    y_pred_norm = norm_sqrt_deg(y_pred)  
                    diff_true = y_norm[src, 0] - y_norm[dst, 0]         
                    diff_pred = y_pred_norm[src, 0] - y_pred_norm[dst, 0]
                    edge_mse_true = (diff_true ** 2).mean()
                    edge_mse_pred = (diff_pred ** 2).mean()

                    sum_nodes_sq_gt = y.pow(2).sum()
                    sum_nodes_sq_pred = y_pred.pow(2).sum()

                    traj_true_rq.append(edge_mse_true.item()*0.5/(sum_nodes_sq_gt.item()+1e-16))
                    traj_pred_rq.append(edge_mse_pred.item()*0.5/(sum_nodes_sq_pred.item()+1e-16))

                    # end sketchy
                    #print(f"Rayleigh Quotient at time {t}: GT {edge_mse_true.item():.6e}, Pred {edge_mse_pred.item():.6e}")

                    data.x = torch.cat([data.x[:, y_pred.shape[1] :, 0], y_pred], 1)[
                        :, :, None
                    ]

            results["true_rayleigh_quotients"][mesh_idx].append(
                np.array(traj_true_rq, dtype=np.float64)
            )
            results["predicted_rayleigh_quotients"][mesh_idx].append(
                np.array(traj_pred_rq, dtype=np.float64)
            )

            results["ground_truth"][mesh_idx].append(
                np.array(all_gts).T
            )  # [Num_nodes, T]

            results["predictions"][mesh_idx].append(
                np.array(all_preds).T
            )  # [Num_nodes, T]
            results["losses"][mesh_idx].append(np.array(all_losses))


        return results

    print(f"Dataset: {cfg.dataset.name}, Backbone: {cfg.backbone.name}")
    for split, dataset in datasets_dict.items():
        if split in ["train", "test_time", "test_init"]:
            continue

        results = eval_step(dataset)

        for mesh_idx, v in results["losses"].items():
            losses = np.asarray(v)
            print(
                f"[{split}] Mesh idx: {mesh_idx}, last RMSE: {losses[:, -1].mean():.3e} +/- {1.96 * losses[:, -1].std(ddof=1):.3e}"
            )

            object_name = dataset.mesh_names[mesh_idx]
            mesh = get_mesh(object_name)

            save_path = (
                Path(cfg.save_dir)
                / cfg.dataset.name
                / split
                / object_name
                / cfg.backbone.name
            )
            save_path.mkdir(parents=True, exist_ok=True)

            np.save(save_path / "losses.npy", results["losses"][mesh_idx])
            np.save(save_path / "predictions.npy", results["predictions"][mesh_idx])
            np.save(save_path / "ground_truth.npy", results["ground_truth"][mesh_idx])

            true_rq = np.stack(
                results["true_rayleigh_quotients"][mesh_idx], axis=0
            )  # [num_traj, T_out]
            pred_rq = np.stack(
                results["predicted_rayleigh_quotients"][mesh_idx], axis=0
            )  # [num_traj, T_out]

            np.save(save_path / "rayleigh_true.npy", true_rq)
            np.save(save_path / "rayleigh_pred.npy", pred_rq)

            plt.figure()
            t = np.arange(true_rq.shape[1])
            plt.plot(
                t,
                true_rq.mean(axis=0),
                label=f"Ground Truth Rayleigh Quotient, Mesh idx {mesh_idx}",
                color="blue",
            )
            plt.plot(
                t,
                pred_rq.mean(axis=0),
                label=f"Prediction Rayleigh Quotient, Mesh idx {mesh_idx}",
                color="orange",
            )
            plt.xlabel("Time step")
            plt.ylabel("Rayleigh Quotient")
            plt.title("Rayleigh Quotient over Time")
            plt.legend()
            plt.savefig(save_path / f"rayleigh_quotients_mesh_{mesh_idx}_{cfg.backbone.name}.png")
            plt.savefig(save_path / f"rayleigh_quotients_mesh_{mesh_idx}_{cfg.backbone.name}.pdf")

            plt.yscale("log")
            plt.savefig(save_path / f"rayleigh_quotients_log_mesh_{mesh_idx}_{cfg.backbone.name}.png")
            plt.savefig(save_path / f"rayleigh_quotients_log_mesh_{mesh_idx}_{cfg.backbone.name}.pdf")


            for s in range(1):
                for t in range(10, 191, 10):
                    gt = results["ground_truth"][mesh_idx][s][:, t]

                    screenshot_mesh(
                        mesh,
                        gt,
                        cfg.dataset.name,
                        object_name,
                        save_path
                        / f"{cfg.dataset.name}_{object_name}_{cfg.backbone.name}_{s}_t{t}_gt.png",
                    )

                    preds = results["predictions"][mesh_idx][s][:, t]
                    screenshot_mesh(
                        mesh,
                        preds,
                        cfg.dataset.name,
                        object_name,
                        save_path
                        / f"{cfg.dataset.name}_{object_name}_{cfg.backbone.name}_{s}_t{t}_preds.png",
                    )

                    rotate_mesh_video(
                            mesh=mesh,
                            scalars=gt,
                            dataset_name=cfg.dataset.name,
                            name=object_name,
                            save_path=save_path / f"{cfg.dataset.name}_{object_name}_{cfg.backbone.name}_{s}_t{t}_gt_video.mp4",
                            n_frames=240,
                            framerate=30,
                            )

                    rotate_mesh_video(
                            mesh=mesh,
                            scalars=preds,
                            dataset_name=cfg.dataset.name,
                            name=object_name,
                            save_path=save_path / f"{cfg.dataset.name}_{object_name}_{cfg.backbone.name}_{s}_t{t}_preds_video.mp4",
                            n_frames=240,
                            framerate=30,
                            )

        # plot mean and std of rayleigh quotients over the iterations and plot them as a function of t, do this for each mesh 



if __name__ == "__main__":
    main()

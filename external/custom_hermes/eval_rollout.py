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

from tqdm import tqdm

import robust_laplacian

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

set_rc_params(15)

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
    mesh = mesh.delaunay_3d()

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

            pos, face = data.pos.cpu(), data.face.cpu()
            L, M = robust_laplacian.mesh_laplacian(pos.cpu().numpy(), face.T.cpu().numpy())
            print("Computed robust Laplacian")
            # verify L symmetric
            L_np = L.toarray()
            L_torch = torch.from_numpy(L_np).to(values.device)
            print("L symmetric:", torch.allclose(L_torch, L_torch.T, atol=1e-6))
            L_torch = -L_torch # opposite sign convention

            M = M.toarray()
            M = torch.from_numpy(M).to(values.device)
            M_inv_sqrt = M.pow(-0.5)
            L_torch = M_inv_sqrt @ L_torch @ M_inv_sqrt

            L_offdiag = L_torch.clone()
            L_offdiag.fill_diagonal_(0)
            A_M = -L_offdiag

            weighted_edge_index = A_M.nonzero(as_tuple=False).t().long().to(values.device) 
            edge_weights = A_M[weighted_edge_index[0], weighted_edge_index[1]].to(values.device).to(values.dtype)
            deg = torch.zeros(N, device=values.device).index_add_(0, weighted_edge_index[0], edge_weights)
            deg = deg.clamp(min=1.0)
            inv_sqrt_deg = deg.rsqrt().view(N, 1)

            src, dst = weighted_edge_index[0], weighted_edge_index[1]

            def norm_sqrt_deg(x: torch.Tensor) -> torch.Tensor:
                return x * inv_sqrt_deg

            all_preds = []
            all_losses = []
            all_gts = []

            traj_true_rq = []
            traj_pred_rq = []

            with torch.no_grad():
                data.x = values[:, 0 : dataset.input_length][..., None]

                #for t in range(dataset.input_length, values.shape[1]):
                # tqdm ift
                for t in tqdm(
                    range(dataset.input_length, values.shape[1]),
                    desc=f"Evaluating mesh idx {mesh_idx}, sample idx {sample_idx}",
                ):
                    y = values[:, t].unsqueeze(-1)

                    all_gts.append(y.squeeze().detach().cpu().numpy())

                    y_pred = model(data)

                    all_preds.append(y_pred.squeeze().detach().cpu().numpy())
                    
                    try:
                        loss = loss_fn(y_pred, y)
                    except:
                        loss = loss_fn(y_pred, y, data.edge_index)
                    all_losses.append(loss.item())

                    # Sketchy over here
                    y_norm      = norm_sqrt_deg(y)       
                    y_pred_norm = norm_sqrt_deg(y_pred)  
                    diff_true = y_norm[src, 0] - y_norm[dst, 0]         
                    diff_pred = y_pred_norm[src, 0] - y_pred_norm[dst, 0]
                    edge_mse_true_weighted = (edge_weights * (diff_true ** 2)).sum()
                    edge_mse_pred_weighted = (edge_weights * (diff_pred ** 2)).sum()

                    sum_nodes_sq_gt = y.pow(2).sum()
                    sum_nodes_sq_pred = y_pred.pow(2).sum()

                    traj_true_rq.append(edge_mse_true_weighted.item()*0.5/(sum_nodes_sq_gt.item()+1e-16))
                    traj_pred_rq.append(edge_mse_pred_weighted.item()*0.5/(sum_nodes_sq_pred.item()+1e-16))

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

        integrated_errors_all = []

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

            plt.figure(figsize=(6, 4))
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
                color="red",
            )
            true_rq_std = true_rq.std(axis=0)
            pred_rq_std = pred_rq.std(axis=0)
            plt.fill_between(t, true_rq.mean(axis=0) - true_rq_std, true_rq.mean(axis=0) + true_rq_std, color="blue", alpha=0.3)
            plt.fill_between(t, pred_rq.mean(axis=0) - pred_rq_std, pred_rq.mean(axis=0) + pred_rq_std, color="red", alpha=0.3)
            plt.xlabel("Time step")
            plt.ylabel("Rayleigh Quotient")
            print("Rayleigh Quotient ranges: GT [{:.6e}, {:.6e}], Pred [{:.6e}, {:.6e}]".format(
                true_rq.min(), true_rq.max(), pred_rq.min(), pred_rq.max()
            ))
            plt.title("Rayleigh Quotient over Time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path / f"rayleigh_quotients_mesh_{mesh_idx}_{cfg.backbone.name}.png")
            plt.savefig(save_path / f"rayleigh_quotients_mesh_{mesh_idx}_{cfg.backbone.name}.pdf")

            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(save_path / f"rayleigh_quotients_log_mesh_{mesh_idx}_{cfg.backbone.name}.png")
            plt.savefig(save_path / f"rayleigh_quotients_log_mesh_{mesh_idx}_{cfg.backbone.name}.pdf")

            traj_error = np.abs(true_rq - pred_rq).sum(axis=1)
            integrated_errors_all.extend(traj_error.tolist())

            integrated_rayleigh_error = traj_error.mean()
            integrated_rayleigh_error_std = traj_error.std()
            print(
                f"[{split}] Mesh idx: {mesh_idx}, Integrated Rayleigh Quotient Error: {integrated_rayleigh_error:.6e} +/- {integrated_rayleigh_error_std:.6e}"
            )


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

            if len(integrated_errors_all) > 0:
                overall_mean = np.mean(integrated_errors_all)
                overall_std = np.std(integrated_errors_all)
                print(
                    f"[{split}] Combined Integrated Rayleigh Quotient Error over all meshes and rollouts: {overall_mean:.6e} +/- {overall_std:.6e} (n={len(integrated_errors_all)})"
                )

        # plot mean and std of rayleigh quotients over the iterations and plot them as a function of t, do this for each mesh 



if __name__ == "__main__":
    main()

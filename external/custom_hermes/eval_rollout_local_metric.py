import copy
from collections import defaultdict

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import robust_laplacian
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from external.custom_hermes.dataset.heatwave_pde import (compute_adj_mat,
                                                         compute_edges_dense)
from external.custom_hermes.eval_rollout import set_rc_params
from external.custom_hermes.utils import create_dataset_loaders
from metrics.rayleigh import compute_localized_rayleigh_quotient

set_rc_params(15)
pv.set_plot_theme("paraview")


def plot_local_rq_dists(locrq_pred, locrq_y, cfg, traj):

    fig, ax = plt.subplots(figsize=(8, 5))

    xmin = min(locrq_pred.min(), locrq_y.min())
    xmax = max(locrq_pred.max(), locrq_y.max())

    ymax = 0
    for t in range(locrq_pred.shape[0]):
        counts, _ = np.histogram(locrq_pred[t], bins=30, density=True)
        ymax = max(ymax, counts.max())
        counts, _ = np.histogram(locrq_y[t], bins=30, density=True)
        ymax = max(ymax, counts.max())
    ymax *= 1.1

    def animate(t):
        ax.clear()
        ax.hist(locrq_pred[t], bins=30, alpha=0.5,
                color="steelblue", label="Local Rayleigh quotients of f(X)", density=True)
        ax.hist(locrq_y[t], bins=30, alpha=0.5, color="coral",
                label="Local Rayleigh quotients of y", density=True)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Value")
        ax.set_title(
            f"t = {t}, backbone = {cfg.backbone.name}, dataset = {cfg.dataset.name}")
        ax.legend()

    ani = animation.FuncAnimation(
        fig, animate, frames=locrq_pred.shape[0], interval=50)
    ani.save(f"{cfg.dataset.name}_{cfg.backbone.name}_{traj}.gif",
             writer="pillow", fps=20)
    plt.close()


@hydra.main(version_base=None, config_path="./conf", config_name="eval_rollout")
def main(cfg):

    cfg.device = 'cpu'

    datasets_dict = create_dataset_loaders(cfg, return_datasets=True)

    backbone = instantiate(cfg.backbone.net).to(cfg.device)
    model = instantiate(cfg.model, backbone=backbone).to(cfg.device)

    model.load_state_dict(torch.load(
        cfg.model_save_path, map_location=cfg.device))
    model.eval()

    loss_fn = instantiate(cfg.loss)

    def eval_step(dataset):
        results = {
            "losses": defaultdict(list),
            "predictions": defaultdict(list),
            "ground_truth": defaultdict(list),
            "predicted_rayleigh_quotients": defaultdict(list),
            "true_rayleigh_quotients": defaultdict(list),
            "nrmse": defaultdict(list),
            "smape": defaultdict(list),
            "locrq_ypred": defaultdict(list),
            "locrq_ytrue": defaultdict(list),
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
            L, M = robust_laplacian.mesh_laplacian(
                pos.cpu().numpy(), face.T.cpu().numpy())
            print("Computed robust Laplacian")
            # verify L symmetric
            L_np = L.toarray()
            L_torch = torch.from_numpy(L_np).to(values.device)
            print("L symmetric:", torch.allclose(
                L_torch, L_torch.T, atol=1e-6))
            L_torch = -L_torch  # opposite sign convention

            M = M.toarray()
            M = torch.from_numpy(M).to(values.device)
            M_inv_sqrt = M.pow(-0.5)
            # L_torch = M_inv_sqrt @ L_torch @ M_inv_sqrt

            L_offdiag = L_torch.clone()
            L_offdiag.fill_diagonal_(0)
            A_M = L_offdiag

            weighted_edge_index = A_M.nonzero(
                as_tuple=False).t().long().to(values.device)
            edge_weights = A_M[weighted_edge_index[0], weighted_edge_index[1]].to(
                values.device).to(values.dtype)
            print("Computed weighted edge index and weights")
            print(f"Weighted graph has {weighted_edge_index.shape[1]} edges.")
            print(
                f"Edge weights stats: min {edge_weights.min().item():.6e}, max {edge_weights.max().item():.6e}, mean {edge_weights.mean().item():.6e}")
            deg = torch.zeros(N, device=values.device).index_add_(
                0, weighted_edge_index[0], edge_weights)
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

            nrmse = []
            smape = []

            locrq_ypreds = []
            locrq_ytrues = []

            with torch.no_grad():
                data.x = values[:, 0: dataset.input_length][..., None]
                data = compute_adj_mat(compute_edges_dense(data))

                # for t in range(dataset.input_length, values.shape[1]):
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
                    y_norm = norm_sqrt_deg(y)
                    y_pred_norm = norm_sqrt_deg(y_pred)
                    diff_true = y_norm[src, 0] - y_norm[dst, 0]
                    diff_pred = y_pred_norm[src, 0] - y_pred_norm[dst, 0]
                    edge_mse_true_weighted = (
                        edge_weights * (diff_true ** 2)).sum()
                    edge_mse_pred_weighted = (
                        edge_weights * (diff_pred ** 2)).sum()

                    sum_nodes_sq_gt = y.pow(2).sum()
                    sum_nodes_sq_pred = y_pred.pow(2).sum()

                    # compute localized rayleigh quotient for y_pred
                    localized_rq_ypred = compute_localized_rayleigh_quotient(
                        y_pred, data.edge_index, 2, 100)
                    localized_rq_ytrue = compute_localized_rayleigh_quotient(
                        y, data.edge_index, 2, 100)

                    traj_true_rq.append(
                        edge_mse_true_weighted.item()*0.5/(sum_nodes_sq_gt.item()+1e-16))
                    traj_pred_rq.append(edge_mse_pred_weighted.item(
                    )*0.5/(sum_nodes_sq_pred.item()+1e-16))

                    locrq_ypreds.append(localized_rq_ypred)
                    locrq_ytrues.append(localized_rq_ytrue)

                    nrmse.append(
                        torch.sqrt(
                            torch.mean((y_pred - y) ** 2)
                            / (torch.mean(y**2) + 1e-8)
                        )
                        .detach()
                        .cpu()
                        .item()
                    )

                    smape.append(
                        (2*torch.abs(y_pred - y) /
                         (torch.abs(y_pred) + torch.abs(y) + 1e-8))
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    )

                    data.x = torch.cat([data.x[:, y_pred.shape[1]:, 0], y_pred], 1)[
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

            results["nrmse"][mesh_idx].append(np.array(nrmse))
            results["smape"][mesh_idx].append(np.array(smape))

            results['locrq_ypred'][mesh_idx].append(np.array(locrq_ypreds))
            results['locrq_ytrue'][mesh_idx].append(np.array(locrq_ytrues))

            if idx == 1:
                break  # NOTE: do it for just one trajectory for rebuttals

        return results

    print(f"Dataset: {cfg.dataset.name}, Backbone: {cfg.backbone.name}")
    results = []
    for split, dataset in datasets_dict.items():
        if split in ["train", "test_time", "test_init"]:
            continue

        results.append(eval_step(dataset))
        
    ch_mesh_idx = 3
    wave_mesh_idx = 0
    mesh_id = ch_mesh_idx

    locrq_pred_0 = results[0]['locrq_ypred'][mesh_id][0]
    locrq_y_0 = results[0]['locrq_ytrue'][mesh_id][0]
    locrq_pred_1 = results[0]['locrq_ypred'][mesh_id][1]
    locrq_y_1 = results[0]['locrq_ytrue'][mesh_id][1]

    plot_local_rq_dists(locrq_pred_0, locrq_y_0, cfg, 0)
    plot_local_rq_dists(locrq_pred_1, locrq_y_1, cfg, 1)


if __name__ == "__main__":
    main()

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
from external.custom_hermes.utils import create_dataset_loaders

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

            edge_mse_true_per_t = []
            edge_mse_pred_per_t = []

            all_preds = []
            all_losses = []
            all_gts = []

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
                    edge_mse_true_per_t.append(edge_mse_true.item())
                    edge_mse_pred_per_t.append(edge_mse_pred.item())
                    # end sketchy
                    print(f"Rayleigh Quotient at time {t}: GT {edge_mse_true.item():.6e}, Pred {edge_mse_pred.item():.6e}")

                    data.x = torch.cat([data.x[:, y_pred.shape[1] :, 0], y_pred], 1)[
                        :, :, None
                    ]

            results["ground_truth"][mesh_idx].append(
                np.array(all_gts).T
            )  # [Num_nodes, T]

            results["predictions"][mesh_idx].append(
                np.array(all_preds).T
            )  # [Num_nodes, T]
            results["losses"][mesh_idx].append(np.array(all_losses))

            results["true_rayleigh_quotients"][mesh_idx].append(
                np.array(edge_mse_true_per_t)
            )
            results["predicted_rayleigh_quotients"][mesh_idx].append(
                np.array(edge_mse_pred_per_t)
            )

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

            for s in range(1):
                for t in range(10, 101, 10):
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


if __name__ == "__main__":
    main()

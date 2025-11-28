from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from pyvista import examples

from pydec import simplicial_complex

from scipy.sparse.linalg import inv

from external.hermes.src.data.pde.utils import get_mesh_laplacian, plot_mesh, screenshot_mesh

objects = {
    "sphere": pv.Sphere(),
    "armadillo": examples.download_armadillo(),
    "bunny_coarse": examples.download_bunny_coarse(),
    "lucy": examples.download_lucy(),
    "woman": examples.download_woman(),
    "spider": examples.download_spider(),
    "urn": examples.download_urn(),
}
decimate_ratio = {
    "sphere": 0.0,
    "armadillo": 0.99,
    "bunny_coarse": 0.0,
    "lucy": 0.95,
    "woman": 0.98,
    "spider": 0.0,
    "urn": 0.98,
}
dts = {
    "sphere": 1e-5,
    "armadillo": 0.1,
    "bunny_coarse": 1e-6,
    "lucy": 0.8,
    "woman": 0.8,
    "spider": 3e-5,
    "urn": 0.6,
}

save_dir = Path("data/advection/raw")
n_samples = 5
Tmax = 200
sigma = 1e-4
names = objects.keys()
plot = True

pv.set_plot_theme("paraview")

def directional_derivative(u_np, vel_np, V_np):
    """
    Compute v · grad(u) along edges
    """
    grad_u   = (u_np[j_idx] - u_np[i_idx])  # d0 @ u simpler version
    e_vec    = V_np[j_idx] - V_np[i_idx]    # edge vector in R^3
    v_edge   = 0.5 * (vel_np[i_idx] + vel_np[j_idx])
    proj     = (v_edge * e_vec).sum(axis=1) / (np.linalg.norm(e_vec, axis=1) + 1e-12)

    # edge-scalar directional derivative
    return grad_u * proj   # (n_edges,)

def contract_to_vertices(dir_edge):
    out = np.zeros(N)
    np.add.at(out, i_idx, dir_edge * 0.5)
    np.add.at(out, j_idx, dir_edge * 0.5)
    return out

# for name in objects.keys():
for name in names:
    mesh = objects[name]
    mesh = mesh.decimate(decimate_ratio[name])
    _ = mesh.clean(inplace=True)

    # Current mesh
    pos = mesh.points
    face = mesh.faces.reshape(-1, 4)[:, 1:]

    pos = torch.from_numpy(pos)
    face = torch.from_numpy(face).T.long()

    edge_index, edge_weight = get_mesh_laplacian(pos, face, normalization="sym")
    tg = torch.zeros(pos.shape[0], pos.shape[0])
    tg[tuple(edge_index)] = edge_weight

    # shape (N_v, 3), (N_f, 3)
    print(f"[{name}] Vertices Shape: {pos.shape}, Faces Shape: {face.shape}")
    sc = simplicial_complex(pos.numpy(), face.numpy().T)

    edges = sc[1].simplices
    i_idx, j_idx = edges[:, 0], edges[:, 1]

    V_np = pos.numpy()
    speed = 1.0                      # pick whatever you want
    vel_np = np.zeros_like(V_np)     # (N, 3)
    vel_np[:, 0] = speed

    d0 = sc[0].d
    star0 = sc[0].star
    star1 = sc[1].star

    div_op = inv(star0) * d0.T * star1

    # Parameters
    N = pos.shape[0]
    dt = dts[name]
    num_start_points = N // 5

    # Create save_dir and save some files
    (save_dir / name).mkdir(parents=True, exist_ok=True)
    mesh.save(save_dir / name / f"{name}.ply")
    np.save(save_dir / name / f"dt.npy", dt)

    for s in range(n_samples):
        # Store values
        U = torch.zeros(Tmax + 1, N, dtype=torch.float32)

        # Initial conditions
        idx = torch.randperm(N)[:num_start_points]
        dist = torch.cdist(pos, pos)
        U[0] += (1 * torch.exp(-dist[idx].pow(2) / (2 * sigma**2))).sum(0)

        u = U[0].clone()
        print(f"[{name}] Sample: {s}, num nodes: {N}")
        u_np = u.numpy()

        if plot and s == 0:
            p = plot_mesh(mesh, torch.zeros(N), "Heat", name)

            p.add_text(f"time: 0.0", font_size=12, name="timelabel")
            p.add_text(
                f"Num Nodes: {N}", font_size=12, name="num_nodes", position="lower_left"
            )

            p.show()

        t = 0.0
        for i in range(1, Tmax + 1):
            dir_edge = directional_derivative(u_np, vel_np, V_np)
            vdotgrad = contract_to_vertices(dir_edge)
            u_np = u_np - dt * vdotgrad
            u = torch.from_numpy(u_np)
            t += dt

            U[i] = u.clone()

            if plot and s == 0:
                p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
                mesh.point_data["c"] = u
                p.app.processEvents()

                if i == 1 or i == Tmax // 4 or i == Tmax:
                    screenshot = save_dir / name / f"{name}_{s}_T{i}.png"
                    screenshot_mesh(mesh, u, "advection", name, screenshot)

        # Save results
        np.save(save_dir / name / f"{name}_{s}.npy", U)

        if plot and s == 0:
            p.close()

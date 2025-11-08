import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import wandb
from metrics.rayleigh import rayleigh_quotients
from model.model_factory import build_model
from model.predictor import NodeLevelRegressor
from toy_heat_diffusion.pyg_toy import load_autoregressive_dataset


def setup_wandb(config, entity_name="rayleigh_analysis_gnn", project_name="toy_heat_diffusion_graphs"):
    run_name = (
        f"{config['model']}_"
        f"{config['act']}_"
        f"h{config['hidden']}_"
        f"lr{config['lr']}_"
    )
    run = wandb.init(
        entity="rayleigh_analysis_gnn",
        project="toy_heat_diffusion_graphs",
        config=config,
        name=run_name
    )
    return run


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_mse, total_nodes = 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y, reduction='sum')
        loss.backward()
        optimizer.step()
        total_mse += loss.item()
        total_nodes += data.num_nodes
    return total_mse / total_nodes


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse, total_nodes = 0, 0
    rayleigh_quotients_x = []
    rayleigh_quotients_xprime = []
    rayleigh_quotients_y = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        mse = F.mse_loss(out, data.y, reduction="sum").item()
        total_mse += mse
        total_nodes += data.num_nodes

        x, xprime, y = rayleigh_quotients(model, data)
        rayleigh_quotients_x.append(x.item())
        rayleigh_quotients_xprime.append(xprime.item())
        rayleigh_quotients_y.append(y.item())

    avg_mse = total_mse / total_nodes

    return avg_mse, np.mean(rayleigh_quotients_x), np.mean(rayleigh_quotients_xprime), np.mean(rayleigh_quotients_y)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--start_time", type=float, default=0.0)
    parser.add_argument("--train_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument(
        "--model", choices=["gcn", "separable_unitary", "lie_unitary"], default="gcn")
    parser.add_argument("--layers", type=int, default=8)
    # Choices ReLU, GroupSort, Identity
    parser.add_argument("--act", type=str, default="ReLU")
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--entity_name", type=str, default="rayleigh_analysis_gnn")
    parser.add_argument("--project_name", type=str, default="toy_heat_diffusion_graphs")

    args = parser.parse_args()

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.save_dir = os.path.join(
        args.save_dir, f"{current_time}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_graphs, eval_graphs = load_autoregressive_dataset(
        args.data_dir, args.start_time, args.train_steps, args.eval_steps
    )

    train_loader = DataLoader(
        train_graphs, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_graphs, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = train_graphs[0].x.size(1)

    if args.model == "gcn":
        base_gnn = build_model(node_dim=in_ch, model_type="GCN", num_layers=args.layers,
                               hidden_size=args.hidden, activation_function=args.act, skip_connections=False, batch_size=64, batch_norm="None")
    elif args.model == 'lie_unitary':
        base_gnn = build_model(node_dim=in_ch, model_type="LieUni", num_layers=args.layers,
                               hidden_size=args.hidden, activation_function=args.act, skip_connections=False, batch_size=64, batch_norm="None")
    elif args.model == 'separable_unitary':
        base_gnn = build_model(node_dim=in_ch, model_type="Uni", num_layers=args.layers,
                               hidden_size=args.hidden, activation_function=args.act, skip_connections=False, batch_size=64, batch_norm="None")
    else:
        raise Exception("We do not like anything else here.")

    complex_floats = args.model in ["separable_unitary", "lie_unitary"]
    model = NodeLevelRegressor(
        base_gnn, in_ch, 1, complex_floats=complex_floats)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4)

    params_gnn = sum(p.numel()
                     for p in base_gnn.parameters() if p.requires_grad)
    print(f"Total parameters in base GNN: {params_gnn}")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {params}")

    config = vars(args)
    run = setup_wandb(config, entity_name=args.entity_name, project_name=args.project_name)

    train_mse_list, val_mse_list = [], [] 
    train_rayleigh_x_list, train_rayleigh_xprime_list, train_rayleigh_y_list = [], [], [] 
    val_rayleigh_x_list, val_rayleigh_xprime_list, val_rayleigh_y_list = [], [], []

    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, device)
        test_mse, rayleigh_x, rayleigh_xprime, rayleigh_y = evaluate(
            model, eval_loader, device)
        _, rayleigh_x_train, rayleigh_xprime_train, rayleigh_y_train = evaluate(
            model, train_loader, device)
        run.log({
            "epoch": epoch,
            "train_mse": avg_train_loss,
            "val_mse": test_mse,
            "train_rayleigh_x": rayleigh_x_train,
            "train_rayleigh_xprime": rayleigh_xprime_train,
            "train_rayleigh_y": rayleigh_y_train,
            "val_rayleigh_x": rayleigh_x,
            "val_rayleigh_xprime": rayleigh_xprime,
            "val_rayleigh_y": rayleigh_y
        })
        train_mse_list.append(avg_train_loss)
        val_mse_list.append(test_mse)
        train_rayleigh_x_list.append(rayleigh_x_train)
        train_rayleigh_xprime_list.append(rayleigh_xprime_train)
        train_rayleigh_y_list.append(rayleigh_y_train)
        val_rayleigh_x_list.append(rayleigh_x)
        val_rayleigh_xprime_list.append(rayleigh_xprime)
        val_rayleigh_y_list.append(rayleigh_y)

    torch.save(model.state_dict(), os.path.join(
        args.save_dir, "model.pt"))
    np.save(os.path.join(args.save_dir, "train_mse.npy"), np.array(train_mse_list))
    np.save(os.path.join(args.save_dir, "val_mse.npy"), np.array(val_mse_list))
    np.save(os.path.join(args.save_dir, "train_rayleigh_x.npy"), np.array(train_rayleigh_x_list))
    np.save(os.path.join(args.save_dir, "train_rayleigh_xprime.npy"), np.array(train_rayleigh_xprime_list))
    np.save(os.path.join(args.save_dir, "train_rayleigh_y.npy"), np.array(train_rayleigh_y_list))
    np.save(os.path.join(args.save_dir, "val_rayleigh_x.npy"), np.array(val_rayleigh_x_list))
    np.save(os.path.join(args.save_dir, "val_rayleigh_xprime.npy"), np.array(val_rayleigh_xprime_list))
    np.save(os.path.join(args.save_dir, "val_rayleigh_y.npy"), np.array(val_rayleigh_y_list))


if __name__ == "__main__":
    main()

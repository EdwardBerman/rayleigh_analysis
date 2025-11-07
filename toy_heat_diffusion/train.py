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


def setup_wandb(config):
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
        "--model", choices=["gcn", "seperable_unitary", "lie_unitary"], default="gcn")
    parser.add_argument("--layers", type=int, default=8)
    # Choices ReLU, GroupSort
    parser.add_argument("--act", type=str, default="ReLU")
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="runs")

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
    elif args.model == 'seperable_unitary':
        base_gnn = build_model(node_dim=in_ch, model_type="Uni", num_layers=args.layers,
                               hidden_size=args.hidden, activation_function=args.act, skip_connections=False, batch_size=64, batch_norm="None")
    else:
        raise Exception("We do not like anything else here.")

    complex_floats = args.model in ["seperable_unitary", "lie_unitary"]
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

    run = setup_wandb(args)

    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, device)
        test_mse, rayleigh_x, rayleigh_xprime, rayleigh_y = evaluate(
            model, eval_loader, device)
        run.log({
            "epoch": epoch,
            "train_mse": avg_train_loss,
            "val_mse": test_mse,
            "val_rayleigh_x": rayleigh_x,
            "val_rayleigh_xprime": rayleigh_xprime,
            "val_rayleigh_y": rayleigh_y
        })

    torch.save(model.state_dict(), os.path.join(
        args.save_dir, "model.pt"))


if __name__ == "__main__":
    main()

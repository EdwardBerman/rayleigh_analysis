"""
TODO: Need to add Rayleigh error calculation
TODO: Need to add Wandb tracking
"""

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model.model_factory import build_model
from model.predictor import NodeLevelRegressor
from toy_heat_diffusion.pyg_toy import load_autoregressive_dataset


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
    return total_loss / sum(d.num_nodes for d in loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse, total_nodes = 0, 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        mse = F.mse_loss(out, data.y, reduction="sum").item()
        total_mse += mse
        total_nodes += data.num_nodes
    avg_mse = total_mse / total_nodes
    return avg_mse, avg_mse ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--start_time", type=float, default=0.0)
    parser.add_argument("--train_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument(
        "--model", choices=["gcn", "unitary"], default="gcn")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--act", type=str, default="ReLU")
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

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
    elif args.model == 'unitary':
        base_gnn = build_model(node_dim=in_ch, model_type="Uni", num_layers=args.layers,
                               hidden_size=args.hidden, activation_function=args.act, skip_connections=False, batch_size=64, batch_norm="None")
    else:
        raise Exception("We do not like anything else here.")

    # build the full model with the regressor head for both
    model = NodeLevelRegressor(
        base_gnn, in_ch, complex_floats=args.model == "unitary")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {params}")

    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, device)
        test_mse, test_rmse = evaluate(model, eval_loader, device)
        print(
            f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f} | Eval MSE: {test_mse:.4f} | Eval RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()

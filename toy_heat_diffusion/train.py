import argparse
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from toy_heat_diffusion.models import GATReg, GCNReg, MPNNReg
from toy_heat_diffusion.pyg_toy import graphs_to_pyg_data


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
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
        out = model(data.x, data.edge_index)
        mse = F.mse_loss(out.unsqueeze(-1), data.y, reduction="sum").item()
        total_mse += mse
        total_nodes += data.num_nodes
    avg_mse = total_mse / total_nodes
    return avg_mse, avg_mse ** 0.5  # (MSE, RMSE)


def main():
    parser = argparse.ArgumentParser(
        description="Train GNNs for node-level heat regression")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .pkl file")
    parser.add_argument(
        "--model", choices=["gcn", "gat", "mpnn"], default="gcn")
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    with open(args.data_path, "rb") as f:
        graphs_list = pickle.load(f)
    pyg_graphs = graphs_to_pyg_data(graphs_list)

    n = len(pyg_graphs)
    n_train = int(0.8 * n)
    train_dataset = pyg_graphs[:n_train]
    test_dataset = pyg_graphs[n_train:]

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = pyg_graphs[0].x.size(1)

    if args.model == "gcn":
        model = GCNReg(in_ch, args.hidden).to(device)
    elif args.model == "gat":
        model = GATReg(in_ch, args.hidden).to(device)
    else:
        model = MPNNReg(in_ch, args.hidden).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train RMSE':>10} | {'Test Loss':>10} | {'Test RMSE':>10} | {'LR':>8} | {'GradNorm':>9}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_nodes = 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out.unsqueeze(-1), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_nodes
            total_nodes += data.num_nodes

        avg_train_loss = total_loss / total_nodes
        train_rmse = avg_train_loss ** 0.5
        test_mse, test_rmse = evaluate(model, test_loader, device)

        # gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:5d} | {avg_train_loss:10.4f} | {train_rmse:10.4f} | "
                  f"{test_mse:10.4f} | {test_rmse:10.4f} | {lr:8.6f} | {grad_norm:9.4f}")


if __name__ == "__main__":
    main()

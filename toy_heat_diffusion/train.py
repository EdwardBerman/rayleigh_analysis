import argparse
import pickle

import torch
import torch.nn.functional as F
from models import GATReg, GCNReg, MPNNReg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def graphs_to_pyg_data(graphs_list):
    """
    Converts a list of graphs (from one time step) to PyTorch Geometric Data objects.
    """
    pyg_graphs = []
    for g in graphs_list:
        x_t = torch.tensor(g["xt"], dtype=torch.float).unsqueeze(1)
        x0 = torch.tensor(g["x0"], dtype=torch.float).unsqueeze(1)
        node_features = torch.cat([x0, x_t], dim=1)

        A = g["A"]
        src, dst = A.nonzero()
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        y = torch.tensor(g["xt"], dtype=torch.float)

        pyg_graphs.append(Data(x=node_features, edge_index=edge_index, y=y))
    return pyg_graphs


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
    total_mse = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        total_mse += F.mse_loss(out, data.y).item() * data.num_nodes
    avg_mse = total_mse / sum(d.num_nodes for d in loader.dataset)
    return avg_mse ** 0.5  # RMSE


def main():
    parser = argparse.ArgumentParser(
        description="Train GNNs for node-level heat regression")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .pkl file")
    parser.add_argument(
        "--model", choices=["gcn", "gat", "mpnn"], default="gcn")
    parser.add_argument("--hidden", type=int, default=64)
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

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        rmse = evaluate(model, test_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Test RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()

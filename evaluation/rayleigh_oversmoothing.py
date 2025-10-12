import os
import numpy as np
import torch
import torch.nn as nn
from simple_parsing import ArgumentParser
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from metrics.rayleigh import rayleigh_error
from model.model_factory import build_model
from parsers.parser_lrgb import LongRangeGraphBenchmarkParser
from parsers.parser_toy import ToyLongRangeGraphBenchmarkParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument("--dataset", type=str,
                        help="LRGB Datasets are PascalVOC-SP, COCO-SP, Peptides-func, Peptides-struct", required=True)
    parser.add_argument("--architecture", type=str,
                        help="GCN, GAT, MPNN, Sage, Uni, Crawl", required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--skip_connections",
                        action="store_true", help="Enable skip connections")
    parser.add_argument("--activation_function", type=str,
                        required=True, help="ReLU, LeakyReLU, Identity, GroupSort")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--batch_norm", type=str, required=True,
                        help="None, BatchNorm, LayerNorm, GraphNorm")
    parser.add_argument("--num_attention_heads", type=int,
                        default=2, required=False, help="Only for GAT")
    parser.add_argument("--dropout_rate", type=float,
                        default=0.1, required=False)
    parser.add_argument("--hidden_size", type=int, default=128, required=False)
    parser.add_argument("--edge_aggregator", type=str, default=False,
                        required=False, help="'GINE', 'GATED', or 'NONE'")

    parser.add_argument("--optimizer", type=str, default="Cosine",
                        required=False, help="Adam or Cosine")
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--epochs", type=int, default=100, required=False)
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, required=False)

    parser.add_argument("--window_size", type=int,
                        default=4, required=False)  # For CRAWL
    parser.add_argument("--receptive_field", type=int,
                        default=5, required=False)  # For CRAWL

    parser.add_argument("--save_dir", type=str,
                        default='output', required=False)

    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--log_rq", action="store_true",
                        help="Enable logging of Rayleigh Quotient error")
    parser.add_argument("--toy", action="store_true",
                        help="Use a much smaller version of the dataset to test")
    args = parser.parse_args()
    print("Arguments:")
    pprint.pprint(vars(args))

    postprocess = determine_data_postprocessing(args.architecture)

    if args.toy:
        parser = ToyLongRangeGraphBenchmarkParser(
            name=args.dataset, transform=postprocess)
    else:
        parser = LongRangeGraphBenchmarkParser(
            name=args.dataset, transform=postprocess)

    dataset = parser.parse()
    train_dataset, val_dataset, test_dataset = dataset[
        'train_dataset'], dataset['val_dataset'], dataset['test_dataset']
    node_dim, edge_dim = dataset['node_dim'], dataset['edge_dim']

    base_gnn_model = build_model(node_dim=node_dim,
                                 model_type=args.architecture,
                                 num_layers=args.num_layers,
                                 hidden_size=args.hidden_size,
                                 activation_function=args.activation_function,
                                 skip_connections=args.skip_connections,
                                 batch_norm=args.batch_norm,
                                 num_attention_heads=args.num_attention_heads,
                                 window_size=args.window_size,
                                 receptive_field=args.receptive_field,
                                 dropout_rate=args.dropout_rate,
                                 edge_aggregator=args.edge_aggregator,
                                 edge_dim=edge_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = base_gnn_model.to(device)

    batch_size = args.batch_size

    dataloader = determine_dataloader(args.architecture)

    train_loader = dataloader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = dataloader(
        test_dataset, batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load(args.model_path))

    train_rayleigh_error, val_rayleigh_error, test_rayleigh_error = [], [], []
        
    for batch in val_loader:
        batch = batch.to(device)
        val_rayleigh_error.append(rayleigh_error(model, batch).item())

    for batch in test_loader:
        batch = batch.to(device)
        test_rayleigh_error.append(rayleigh_error(model, batch).item())

    for batch in train_loader:
        batch = batch.to(device)
        train_rayleigh_error.append(rayleigh_error(model, batch).item())

    print(f"Train Rayleigh Quotient Error: {np.mean(train_rayleigh_error)} +/- {np.std(train_rayleigh_error)}")
    print(f"Val Rayleigh Quotient Error: {np.mean(val_rayleigh_error)} +/- {np.std(val_rayleigh_error)}")
    print(f"Test Rayleigh Quotient Error: {np.mean(test_rayleigh_error)} +/- {np.std(test_rayleigh_error)}")

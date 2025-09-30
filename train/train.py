import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from torch_geometric.data import DataLoader, Data
from simple_parsing import ArgumentParser

import pprint
from tqdm import tqdm
import wandb
from enum import Enum

from model.model_factory import build_model
from model.predictor import GraphLevelRegressor, NodeLevelRegressor, GraphLevelClassifier, NodeLevelClassifier
from parsers.parser_lrgb import LongeRangeGraphBenchmarkParser
from external.weighted_cross_entropy import weighted_cross_entropy

class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"

def step(model: nn.Module, data: DataLoader, loss: nn.Module, run: wandb.run, mode: Mode, optimizer: torch.optim.Optimizer, acc_scorer: nn.Module | None = None):
    """
    Computes one step of training, evaluation, or testing and logs to wandb. If the task is classification it will also log the accuracy.
    """
    if mode == Mode.TRAIN:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    
    out = model(data)
    l = loss(out, data.y)

    if mode == Mode.TRAIN:
        l.backward()
        optimizer.step()

    acc = acc_scorer(out, data.y) if acc_scorer is not None else None

    if mode == Mode.TRAIN:
        run.log({"train_loss": l.item(), "train_acc": acc}) if acc is not None else run.log({"train_loss": l.item()})
    elif mode == Mode.EVAL:
        run.log({"val_loss": l.item(), "val_acc": acc}) if acc is not None else run.log({"val_loss": l.item()})
    else:
        run.log({"test_loss": l.item(), "test_acc": acc}) if acc is not None else run.log({"test_loss": l.item()})

    return l.item(), acc 

def setup_wandb(lr: float, architecture: str, dataset: str, epochs: int):
    run = wandb.init(
            entity="rayleigh_analysis_gnn", 
            project="eb_ll_rule_the_tri_state_area",
            config={
                "learning_rate": lr,
                "architecture": architecture,
                "dataset": dataset,
                "epochs": epochs,
                },
            )
    return run

def train(model: nn.Module, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          test_loader: DataLoader, 
          loss_fn: nn.Module, 
          optimizer: torch.optim.Optimizer, 
          run: wandb.run, 
          epochs: int,
          output_dir: str,
          device: torch.device,
          acc_scorer: nn.Module | None = None):

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_loss = float('inf')

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = 0, 0
        val_loss, val_acc = 0, 0
        test_loss, test_acc = 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            loss, accuracy = step(model, batch, loss_fn, run, Mode.TRAIN, optimizer, acc_scorer)
            train_loss += loss
            train_acc += accuracy if accuracy is not None else 0
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / len(train_loader) if acc_scorer is not None else 0)

        for batch in val_loader:
            batch = batch.to(device)
            loss, accuracy = step(model, batch, loss_fn, run, Mode.EVAL, optimizer=None, acc_scorer=acc_scorer)
            val_loss += loss
            val_acc += accuracy if accuracy is not None else 0
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc / len(val_loader) if acc_scorer is not None else 0)

        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

        for batch in test_loader:
            batch = batch.to(device)
            loss, accuracy = step(model, batch, loss_fn, run, Mode.TEST, optimizer=None, acc_scorer=acc_scorer)
            test_loss += loss
            test_acc += accuracy if accuracy is not None else 0
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(test_acc / len(test_loader) if acc_scorer is not None else 0)

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))

    np.save(os.path.join(output_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output_dir, "train_accuracies.npy"), np.array(train_accuracies))
    np.save(os.path.join(output_dir, "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(output_dir, "val_accuracies.npy"), np.array(val_accuracies))
    np.save(os.path.join(output_dir, "test_losses.npy"), np.array(test_losses))
    np.save(os.path.join(output_dir, "test_accuracies.npy"), np.array(test_accuracies))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="LRGB Datasets are PascalVOC-SP, COCO-SP, Peptides-func, Peptides-struct", required=True)
    parser.add_argument("--architecture", type=str, help="GCN, GAT, MPNN, Sage, Uni, Crawl", required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--skip_connections", type=bool, required=True)
    parser.add_argument("--activation_function", type=str, required=True, help="ReLU, LeakyReLU, Identity, GroupSort")
    parser.add_argument("--batch_size", type=int, required=True) 
    parser.add_argument("--batch_norm", type=str, required=True, help="None, BatchNorm, LayerNorm, GraphNorm")
    parser.add_argument("--num_attention_heads", type=int, default=2, required=False) # Only for GAT
    parser.add_argument("--dropout_rate", type=float, default=0.1, required=False)
    parser.add_argument("--hidden_size", type=int, default=128, required=False) 
    parser.add_argument("--edge_aggregator", type=bool, default=False, required=False) # For models that don't support edge features on datasets with edge features

    parser.add_argument("--optimizer", type=str, default="Adam", required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--epochs", type=int, default=100, required=False)
    parser.add_argument("--weight_decay", type=float, default=0.0, required=False)
    
    parser.add_argument("--window_size", type=int, default=4, required=False) # For CRAWL 
    parser.add_argument("--receptive_field", type=float, default=5, required=False) # For CRAWL
    
    parser.add_argument("--save_dir", type=str, default='output', required=False) # For CRAWL

    args = parser.parse_args()
    pprint.pprint(vars(args))

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.save_dir = os.path.join(args.save_dir, f"{args.architecture}_{args.dataset}_{current_time}")
    os.makedirs(args.save_dir, exist_ok=True)

    # node_dim and edge_dim will be determined by dataset. Parser should return node_dim, edge_dim, loss function, accuracy function, and the predictor head it needs
    # TODO: When this gets bigger, we can abstract a function that will figure out the dataset based on the keyword. For now, we assume lrgb.
    parser = LongeRangeGraphBenchmarkParser(name=args.dataset)
    dataset = parser.parse()
    train_dataset, val_dataset, test_dataset = dataset['train_dataset'], dataset['val_dataset'], dataset['test_dataset']
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

    is_classification = dataset['is_classification']
    level = dataset['level']

    if is_classification:
        num_classes = dataset['num_classes']
        loss_fn = weighted_cross_entropy 
        # TODO: Make an accuracy function for classification
        acc_scorer = None
        if level == "graph_level":
            model = GraphLevelClassifier(base_gnn_model, node_dim, num_classes)
        else:
            model = NodeLevelClassifier(base_gnn_model, node_dim, num_classes)
    else:
        loss_fn = nn.MSELoss()
        acc_scorer = None
        if level == "graph_level":
            model = GraphLevelRegressor(base_gnn_model, node_dim)
        else:
            model = NodeLevelRegressor(base_gnn_model, node_dim)

    run = setup_wandb(lr=args.lr, 
                      architecture=args.architecture, 
                      dataset=args.dataset, 
                      epochs=args.epochs)

    #TODO: Set up different optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) if args.optimizer == "Adam" else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train(model=model, 
          train_loader=train_loader, 
          val_loader=val_loader, 
          test_loader=test_loader, 
          loss_fn=loss_fn, 
          optimizer=optimizer, 
          run=run, 
          epochs=args.epochs,
          output_dir=args.save_dir,
          device=device,
          acc_scorer=acc_scorer if is_classification else None)

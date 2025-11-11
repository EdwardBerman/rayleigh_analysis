import os
import pprint
from datetime import datetime
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_parsing import ArgumentParser
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from evaluation.basic_learning_curve_diagnostics import plot_learning_curve, plot_accuracy_curve
from external.weighted_cross_entropy import weighted_cross_entropy
from metrics.accuracy import node_level_accuracy
from metrics.rayleigh import rayleigh_error
from model.model_factory import build_model
from model.predictor import (GraphLevelClassifier, GraphLevelRegressor,
                             NodeLevelClassifier, NodeLevelRegressor)
from parsers.parser_lrgb import LongRangeGraphBenchmarkParser
from parsers.parser_toy import ToyLongRangeGraphBenchmarkParser

from torch.optim.lr_scheduler import CosineAnnealingLR
from muon import SingleDeviceMuon

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score


def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def determine_data_postprocessing(model: str):
    if model == "CRAWL":
        from external.crawl.data_utils import preproc
        return preproc
    else:
        # return preprocessing function that does nothing
        return lambda x: x


def determine_dataloader(model: str):
    if model == "CRAWL":
        from external.crawl.data_utils import CRaWlLoader
        return CRaWlLoader
    else:
        return DataLoader

def bce_multilabel_loss(pred, true):
    """
    pred: [B, C] raw logits
    true: [B, C] or [B, 1, C] with 0/1 labels
    """
    true = true.float()
    if true.ndim > 2:
        true = true.view(true.size(0), -1)

    if true.shape != pred.shape:
        true = true.view_as(pred)

    return F.binary_cross_entropy_with_logits(pred, true)

def graph_level_accuracy(pred, true):
    """
    pred: [B, C] raw logits
    true: [B, C] or [B, 1, C] with 0/1 labels
    """
    true = true.float()
    if true.ndim > 2:
        true = true.view(true.size(0), -1)

    if true.shape != pred.shape:
        true = true.view_as(pred)

    probs = torch.sigmoid(pred)
    preds = (probs > 0.5).float()

    correct = (preds == true).float().mean(dim=1)
    return correct.mean()

def graph_level_average_precision(y_pred, y_true):

    ap_list = []

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)

def eval_F1(seq_ref, seq_pred):
    # '''
    #     compute F1 score averaged over samples
    # '''

    precision_list = []
    recall_list = []
    f1_list = []

    for l, p in zip(seq_ref, seq_pred):
        label = set(l)
        prediction = set(p)
        true_positive = len(label.intersection(prediction))
        false_positive = len(prediction - label)
        false_negative = len(label - prediction)

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {'precision': np.average(precision_list),
            'recall': np.average(recall_list),
            'F1': np.average(f1_list)}

class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"

def step(model: nn.Module, 
         data: Data, 
         loss: nn.Module, 
         run: wandb.run, 
         mode: str, 
         optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | None, 
         scheduler: torch.optim.lr_scheduler._LRScheduler | None, 
         acc_scorer: nn.Module | None = None):
    """
    Computes one step of training, evaluation, or testing and logs to wandb. If the task is classification it will also log the accuracy.
    """
    if mode == Mode.TRAIN:
        model.train()
        if optimizer is not None:
            if isinstance(optimizer, list):
                for opt in optimizer:
                    opt.zero_grad()
            else:
                optimizer.zero_grad()
    else:
        model.eval()

    out = model(data)
    l = loss(out, data.y)

    if mode == Mode.TRAIN:
        l.backward()
        if optimizer is not None:
            if isinstance(optimizer, list):
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()
        if scheduler is not None:
            scheduler.step()

    acc = acc_scorer(out, data.y) if acc_scorer is not None else None

    return l.item(), acc


def setup_wandb(entity: str, 
                project: str, 
                name: str, 
                lr: float, 
                architecture: str, 
                dataset: str, 
                epochs: int) -> wandb.run:

    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
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
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          run: wandb.run,
          epochs: int,
          output_dir: str,
          device: torch.device,
          log_rq: bool = False,
          acc_scorer: nn.Module | None = None):

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        run.log({"epoch": epoch + 1})

        train_loss, train_acc = 0, 0
        val_loss, val_acc = 0, 0
        test_loss, test_acc = 0, 0
        test_rayleigh_error = []

        for batch in val_loader:
            batch = batch.to(device)
            loss, accuracy = step(model, batch, loss_fn, run,
                                  Mode.EVAL, optimizer=None, scheduler=None, acc_scorer=acc_scorer)
            val_loss += loss
            val_acc += accuracy if accuracy is not None else 0

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc / len(val_loader)
                              if acc_scorer is not None else 0)
        run.log({"val_loss": val_losses[-1], "val_acc": val_accuracies[-1]}
                ) if acc_scorer is not None else run.log({"val_loss": val_losses[-1]})

        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            torch.save(model.base_model.state_dict(), os.path.join(output_dir, "best_model_gnn.pt"))

        for batch in test_loader:
            batch = batch.to(device)
            loss, accuracy = step(model, batch, loss_fn, run,
                                  Mode.TEST, optimizer=None, scheduler=None, acc_scorer=acc_scorer)
            test_loss += loss
            test_acc += accuracy if accuracy is not None else 0
            
            if log_rq:
                test_rayleigh_error.append(rayleigh_error(model.base_model, batch).item())

        if log_rq:
            run.log({"test_rayleigh_error": np.mean(test_rayleigh_error)})

        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(test_acc / len(test_loader)
                               if acc_scorer is not None else 0)
        run.log({"test_loss": test_losses[-1], "test_acc": test_accuracies[-1]}
                ) if acc_scorer is not None else run.log({"test_loss": test_losses[-1]})

        for batch in train_loader:
            batch = batch.to(device)
            loss, accuracy = step(model, batch, loss_fn,
                                  run, Mode.TRAIN, optimizer, scheduler, acc_scorer)
            train_loss += loss
            train_acc += accuracy if accuracy is not None else 0

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(
            train_acc / len(train_loader) if acc_scorer is not None else 0)
        run.log({"train_loss": train_losses[-1], "train_acc": train_accuracies[-1]}
                ) if acc_scorer is not None else run.log({"train_loss": train_losses[-1]})

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    torch.save(model.base_model.state_dict(), os.path.join(output_dir, "final_model_gnn.pt"))

    np.save(os.path.join(output_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output_dir, "train_accuracies.npy"),
            np.array(train_accuracies))
    np.save(os.path.join(output_dir, "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(output_dir, "val_accuracies.npy"),
            np.array(val_accuracies))
    np.save(os.path.join(output_dir, "test_losses.npy"), np.array(test_losses))
    np.save(os.path.join(output_dir, "test_accuracies.npy"),
            np.array(test_accuracies))

    plot_learning_curve(train_losses, val_losses, test_losses, output_dir)

    if log_rq:
        np.save(os.path.join(output_dir, "test_rayleigh_error.npy"),
                np.array(test_rayleigh_error))

    if acc_scorer is not None:
        plot_accuracy_curve(train_accuracies, val_accuracies, test_accuracies, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        help="LRGB Datasets are PascalVOC-SP, COCO-SP, Peptides-func, Peptides-struct", required=True)
    parser.add_argument("--architecture", type=str,
                        help="GCN, GAT, MPNN, Sage, Uni, LieUni, Crawl", required=True)
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
                        required=False, help="Adam, Cosine, or Muon")
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
    parser.add_argument("--entity", type=str,
                        default="rayleigh_analysis_gnn", help="Wandb entity name", required=False)
    parser.add_argument("--project", type=str,
                        default="eb_ll_rule_the_tri_state_area", help="Wandb project name", required=False)
    args = parser.parse_args()
    print("Arguments:")
    pprint.pprint(vars(args))

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.save_dir = os.path.join(
        args.save_dir, f"architecture_{args.architecture}_dataset_{args.dataset}_num_layers_{args.num_layers}_{current_time}")
    os.makedirs(args.save_dir, exist_ok=True)
    wandb_name = f"arch_{args.architecture}_data_{args.dataset}_layers_{args.num_layers}_{current_time}"

    # TODO: When this gets bigger, we can abstract a function that will figure out the dataset based on the keyword. For now, we assume lrgb.

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

    if args.edge_aggregator == "NONE":
        args.edge_aggregator = None

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
    complex_floats = True if args.architecture == "Uni" else False
    print(f"Complex floats enabled: {complex_floats}")

    if is_classification:
        num_classes = dataset['num_classes']
        acc_scorer = None
        if level == "graph_level":
            loss_fn = bce_multilabel_loss
            model = GraphLevelClassifier(base_gnn_model, node_dim, num_classes, complex_floats=complex_floats)
            acc_scorer = graph_level_average_precision
        else:
            loss_fn = weighted_cross_entropy
            model = NodeLevelClassifier(base_gnn_model, node_dim, num_classes, complex_floats=complex_floats)
            acc_scorer = node_level_accuracy
    else:
        loss_fn = nn.MSELoss()
        acc_scorer = None
        output_dim = dataset['num_classes'] # corresponds to output dimension for regression
        if level == "graph_level":
            model = GraphLevelRegressor(base_gnn_model, node_dim, output_dim, complex_floats=complex_floats)
        else:
            model = NodeLevelRegressor(base_gnn_model, node_dim, output_dim, complex_floats=complex_floats)

    run = setup_wandb(entity=args.entity,
                      project=args.project,
                      name=wandb_name,
                      lr=args.lr,
                      architecture=args.architecture,
                      dataset=args.dataset,
                      epochs=args.epochs)

    match args.optimizer:
        case "Cosine":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) 
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = None
        case "Muon":
            all_params = list(model.parameters())
            muon_params = [p for p in all_params if p.ndim >= 2]   # matrices only
            other_params = [p for p in all_params if p.ndim < 2]
            optimizer_muon = SingleDeviceMuon(muon_params, lr=args.lr, weight_decay=args.weight_decay)
            optimizer_other = torch.optim.Adam(other_params, lr=args.lr, weight_decay=args.weight_decay)
            optimizer = [optimizer_muon, optimizer_other]
            scheduler = None
        case _:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = args.batch_size

    dataloader = determine_dataloader(args.architecture)

    train_loader = dataloader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = dataloader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # one more param count for the road (cowboy emoji)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    set_seeds(42)

    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          test_loader=test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer,
          scheduler=scheduler,
          run=run,
          epochs=args.epochs,
          output_dir=args.save_dir,
          device=device,
          log_rq=args.log_rq,
          acc_scorer=acc_scorer if is_classification else None)

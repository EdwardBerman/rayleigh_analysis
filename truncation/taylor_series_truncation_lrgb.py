import argparse
import os
import pprint
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from muon import SingleDeviceMuon
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from external.unitary_gcn import UnitaryGCNConvLayer
from external.weighted_cross_entropy import weighted_cross_entropy
from metrics.accuracy import node_level_accuracy
from metrics.rayleigh import rayleigh_quotients
from model.edge_aggregator import EdgeModel, NodeModel
from model.model_factory import UniStack, str_to_activation
from model.predictor import (GraphLevelClassifier, GraphLevelRegressor,
                             NodeLevelClassifier, NodeLevelRegressor)
from parsers.parser_lrgb import LongRangeGraphBenchmarkParser
from parsers.parser_toy import ToyLongRangeGraphBenchmarkParser
from train.train import (bce_multilabel_loss, determine_data_postprocessing,
                         determine_dataloader, graph_level_accuracy, set_seeds,
                         train)


def setup_wandb(config, run_name: str, project: str):
    run = wandb.init(
        entity="rayleigh_analysis_gnn",
        project=project,
        config=config,
        name=run_name
    )
    return run


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--architecture", type=str,
                        help="Uni, LieUni", required=True)
    parser.add_argument("--act", type=str, default="Identity")
    parser.add_argument("--edge_agg", type=str, default="GINE")
    parser.add_argument("--truncation", type=int,
                        help="Determines how truncated the taylor series is.", required=True)
    parser.add_argument("--epochs", type=int,
                        required=False, default=100)
    parser.add_argument("--save_dir", type=str,
                        default='output', required=False)
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--toy", action="store_true",
                        help="Use a much smaller version of the dataset to test")
    parser.add_argument("--project", type=str)
    parser.add_argument("--optimizer", type=str, default="Cosine",
                        required=False, help="Adam, Cosine, or Muon")

    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--dropout_rate", type=float,
                        default=0.1, required=False)

    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--weight_decay", type=float,
                        default=0.01, required=False)

    args = parser.parse_args()
    print("Arguments:")
    pprint.pprint(vars(args))

    config = {
        "DATASET": "Peptides-struct",
        "NUM_LAYERS": args.num_layers,
        "SKIP_CONNECTIONS": False,
        "BATCH_SIZE": args.batch_size,
        "BATCH_NORM": "None",
        "DROPOUT_RATE": args.dropout_rate,
        "HIDDEN_SIZE": 128,
        "OPTIMIZER": args.optimizer,
        "LR": args.lr,
        "EPOCHS": args.epochs,
        "WEIGHT_DECAY": args.weight_decay,
    }


    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    name = f"{args.architecture}_tst_{args.truncation}_{args.act}_{args.edge_agg}_{current_time}"
    args.save_dir = os.path.join(
        args.save_dir, name)
    os.makedirs(args.save_dir, exist_ok=True)

    postprocess = determine_data_postprocessing(args.architecture)

    if args.toy:
        parser = ToyLongRangeGraphBenchmarkParser(
            name=config['DATASET'], transform=postprocess)
    else:
        parser = LongRangeGraphBenchmarkParser(
            name=config['DATASET'], transform=postprocess)

    dataset = parser.parse()
    train_dataset, val_dataset, test_dataset = dataset[
        'train_dataset'], dataset['val_dataset'], dataset['test_dataset']
    node_dim, edge_dim = dataset['node_dim'], dataset['edge_dim']

    activation_function = str_to_activation(args.act)

    if args.architecture == "Uni":
        module_list = []
        for layer in range(config['NUM_LAYERS']):
            input_dim = node_dim if layer == 0 else config['HIDDEN_SIZE']
            output_dim = node_dim if layer == config['NUM_LAYERS'] - \
                1 else config['HIDDEN_SIZE']
            module_list.append(UnitaryGCNConvLayer(input_dim,
                                                   output_dim,
                                                   dropout=config['DROPOUT_RATE'],
                                                   residual=config['SKIP_CONNECTIONS'],
                                                   global_bias=True,
                                                   T=args.truncation,
                                                   use_hermitian=False,
                                                   activation=activation_function()))
        model = UniStack(module_list)
        base_gnn_model = EdgeModel(edge_dim, node_dim, model,
                                   args.edge_agg) if args.edge_agg is not None else NodeModel(model)
    elif args.architecture == 'LieUni':
        module_list = []
        input_dim = node_dim
        output_dim = node_dim
        if input_dim != output_dim:
            print(
                f"Warning: For Lie Unitary GCN, input and output dimensions must be the same, but a distinct output size was set. \nSetting output dim {output_dim} to be input dim {input_dim}\nDid you mean Separable Unitary Convolution?")
        if input_dim != config['HIDDEN_SIZE']:
            print(
                f"Warning: For Lie Unitary GCN, input and hidden dimensions must be the same, but a distinct hidden size was set. \nSetting hidden dim {config['HIDDEN_SIZE']} to be input dim {input_dim}\nDid you mean Separable Unitary Convolution?")
        for layer in range(config['NUM_LAYERS']):
            module_list.append(UnitaryGCNConvLayer(input_dim,
                                                   input_dim,
                                                   dropout=config['DROPOUT_RATE'],
                                                   residual=config['SKIP_CONNECTIONS'],
                                                   global_bias=True,
                                                   T=args.truncation,
                                                   use_hermitian=True,
                                                   activation=activation_function()))
        model = UniStack(module_list)
        base_gnn_model = EdgeModel(edge_dim, node_dim, model,
                                   args.edge_agg) if args.edge_agg is not None else NodeModel(model)
    else:
        raise Exception("Architecture not recognized.")

    is_classification = dataset['is_classification']
    level = dataset['level']

    if is_classification:
        num_classes = dataset['num_classes']
        acc_scorer = None
        if level == "graph_level":
            loss_fn = bce_multilabel_loss
            model = GraphLevelClassifier(
                base_gnn_model, node_dim, num_classes, complex_floats=True)
            acc_scorer = graph_level_accuracy
        else:
            loss_fn = weighted_cross_entropy
            model = NodeLevelClassifier(
                base_gnn_model, node_dim, num_classes, complex_floats=True)
            acc_scorer = node_level_accuracy
    else:
        loss_fn = nn.MSELoss()
        acc_scorer = None
        output_dim = dataset['num_classes']
        if level == "graph_level":
            model = GraphLevelRegressor(
                base_gnn_model, node_dim, output_dim, complex_floats=True)
        else:
            model = NodeLevelRegressor(
                base_gnn_model, node_dim, output_dim, complex_floats=True)

    run = setup_wandb(config=config, run_name=name, project=args.project)

    match config['OPTIMIZER']:
        case "Cosine":
            optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        case "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
            scheduler = None
        case "Muon":
            all_params = list(model.parameters())
            # matrices only
            muon_params = [p for p in all_params if p.ndim >= 2]
            other_params = [p for p in all_params if p.ndim < 2]
            optimizer_muon = SingleDeviceMuon(
                muon_params, lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
            optimizer_other = torch.optim.Adam(
                other_params, lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
            optimizer = [optimizer_muon, optimizer_other]
            scheduler = None
        case _:
            raise ValueError(f"Unsupported optimizer.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = config['BATCH_SIZE']

    dataloader = determine_dataloader(args.architecture)

    train_loader = dataloader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = dataloader(
        test_dataset, batch_size=batch_size, shuffle=False)

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
          log_rq=True,
          acc_scorer=acc_scorer if is_classification else None)

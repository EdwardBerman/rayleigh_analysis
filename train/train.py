from simple_parsing import ArgumentParser
import pprint
from model.model_factory import build_model
import wandb

# def step function(model, data, optimizer, criterion):

def setup_wandb(lr: float, architecture: str, dataset: str, epochs: int):
    run = wandb.init(
            entity="rayleigh_analysis_gnn-org",
            project="eb_ll_rule_the_tri_state_area",
            config={
                "learning_rate": lr,
                "architecture": architecture,
                "dataset": dataset,
                "epochs": epochs,
                },
            )
    return run

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="", required=True)
    parser.add_argument("--architecture", type=str, help="GCN, GAT, MPNN, Sage, Uni, Crawl, LINKX", required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--skip_connections", type=bool, required=True)
    parser.add_argument("--activation_function", type=str, required=True, help="ReLU, LeakyReLU, Identity, GroupSort")
    parser.add_argument("--batch_size", type=int, required=True) 
    parser.add_argument("--batch_norm", type=str, required=True, help="None, BatchNorm, LayerNorm, GraphNorm")
    parser.add_argument("--num_attention_heads", type=int, default=2, required=False) # Only for GAT
    parser.add_argument("--dropout_rate", type=float, default=0.1, required=False)
    parser.add_argument("--hidden_size", type=int, default=128, required=False) # Unitary Convolution has to be square
    parser.add_argument("--edge_aggregator", type=bool, default=False, required=False) # For models that don't support edge features on datasets with edge features

    parser.add_argument("--optimizer", type=str, default="Adam", required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument("--epochs", type=int, default=100, required=False)
    parser.add_argument("--weight_decay", type=float, default=0.0, required=False)

    args = parser.parse_args()
    pprint.pprint(vars(args))

    # node_dim and edge_dim will be determined by dataset

    model = build_model(node_dim=node_dim,
                        model_type=args.architecture,
                        num_layers=args.num_layers,
                        hidden_size=args.hidden_size,
                        activation_function=getattr(nn, args.activation_function),
                        skip_connections=args.skip_connections,
                        batch_size=args.batch_size,
                        batch_norm=args.batch_norm,
                        num_attention_heads=args.num_attention_heads,
                        dropout_rate=args.dropout_rate,
                        edge_aggregator=args.edge_aggregator,
                        edge_dim=edge_dim)

    run = setup_wandb(lr=args.lr, architecture=args.architecture, dataset=args.dataset, epochs=args.epochs)


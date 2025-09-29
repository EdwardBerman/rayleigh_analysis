from simple_parsing import ArgumentParser
import pprint
from model.model_factory import build_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--architecture", type=str, help="GCN, GAT, MPNN, Sage, Uni, Crawl, LINKX", default="GCN", required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--skip_connections", type=bool, required=True)
    parser.add_argument("--activation_function", type=str, required=True, help="ReLU, LeakyReLU, Identity, GroupSort")
    parser.add_argument("--batch_size", type=int, required=True) 
    parser.add_argument("--batch_norm", type=str, required=True, help="None, BatchNorm, LayerNorm, GraphNorm")
    parser.add_argument("--num_attention_heads", type=int, default=2, required=False) # Only for GAT
    parser.add_argument("--dropout_rate", type=float, default=0.1, required=False)
    parser.add_argument("--hidden_size", type=int, default=128, required=False) # Unitary Convolution has to be square
    parser.add_argument("--edge_aggregator", type=bool, default=False, required=False) # For models that don't support edge features on datasets with edge features
    args = parser.parse_args()
    pprint.pprint(vars(args))

    model = build_model(model_type=args.architecture,
                        num_layers=args.num_layers,
                        hidden_size=args.hidden_size,
                        activation_function=getattr(nn, args.activation_function),
                        skip_connections=args.skip_connections,
                        batch_size=args.batch_size,
                        batch_norm=args.batch_norm,
                        num_attention_heads=args.num_attention_heads,
                        dropout_rate=args.dropout_rate,
                        edge_aggregator=args.edge_aggregator)


import argparse
import os
import pprint
from datetime import datetime
from types import SimpleNamespace

from external.fast_flops.fast_flops import flops_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision('high')
import torch._dynamo.config
import torch._inductor.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch.backends.cudnn.enabled = True

from external.unitary_gcn import UnitaryGCNConvLayer
from metrics.rayleigh import rayleigh_quotients
from model.edge_aggregator import NodeModel
from model.model_factory import UniStack, str_to_activation
from model.predictor import NodeLevelRegressor
from toy_heat_diffusion.pyg_toy import load_autoregressive_dataset
from train.train import set_seeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(args):

    activation_function = str_to_activation(args.ACTIVATION_FUNCTION)

    if args.architecture in ("Uni", "LieUni"):
        module_list = []

        if args.architecture == "LieUni":
            input_dim = output_dim = 1

            if args.HIDDEN_SIZE != 1:
                print(
                    f"Warning: For Lie Unitary GCN, input/hidden dims must match (got hidden {args.HIDDEN_SIZE}). "
                    "Did you mean Separable Unitary Convolution?"
                )

            for _ in range(args.NUM_LAYERS):
                module_list.append(UnitaryGCNConvLayer(
                    input_dim,
                    input_dim,
                    dropout=0.1,
                    residual=args.SKIP_CONNECTIONS,
                    global_bias=False,
                    T=args.truncation,
                    use_hermitian=True,
                    activation=activation_function()
                ))
        else:
            for layer in range(args.NUM_LAYERS):
                input_dim = 1 if layer == 0 else args.HIDDEN_SIZE
                output_dim = 1 if layer == args.NUM_LAYERS - \
                    1 else args.HIDDEN_SIZE

                module_list.append(UnitaryGCNConvLayer(
                    input_dim,
                    output_dim,
                    dropout=0.1,
                    residual=args.SKIP_CONNECTIONS,
                    global_bias=False,
                    T=args.truncation,
                    use_hermitian=False,
                    activation=activation_function()
                ))
        return UniStack(module_list)
    else:
        raise Exception("Architecture not recognized.")


def evaluate_flops(model, loader, device):
    model.eval()

    flops = []

    def func(data):
        return model(data)

    @flops_counter
    def func_flops(func, data):
        return func(data)

    func = torch.compile(func, fullgraph=True)

    for data in loader:
        data = data.to(device)
        out = model(data)
        out = out.real if torch.is_complex(out) else out # Take the real part if complex, i.e., for unitary models
        out = out.type(torch.float32) if torch.is_complex(out) else out # Same thing
        _, batch_flops = func_flops(func, data)
        flops.append(batch_flops)


    return flops


def run_experiment(args, save_dir):

    set_seeds(args.seed)

    _, eval_graphs = load_autoregressive_dataset(
        args.data_dir, args.start_time, args.train_steps, args.eval_steps
    )

    eval_loader = DataLoader(eval_graphs, batch_size=args.BATCH_SIZE)

    model = NodeModel(build_model(args)).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    flops = evaluate_flops(model, eval_loader, device)


    return flops


def main(save_dir):

    config = {
        "NUM_LAYERS": 1,
        "SKIP_CONNECTIONS": False,
        "ACTIVATION_FUNCTION": "Identity",
        "BATCH_SIZE": 200,
        "BATCH_NORM": "None",
        "HIDDEN_SIZE": 64
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--start_time", type=float, default=0.0)
    parser.add_argument("--train_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--architecture", type=str,
                        help="Uni, LieUni", required=True)
    parser.add_argument("--truncation", type=int,
                        help="Determines how truncated the taylor series is.", required=True)
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--toy", action="store_true",
                        help="Use a much smaller version of the dataset to test")

    args = parser.parse_args()
    print("Arguments:")
    pprint.pprint(vars(args))

    all_args = {**config, **vars(args)}

    flops = run_experiment(all_args, save_dir)

    truncation_str = f"truncation_{args.truncation}"
    np.save(os.path.join(save_dir, f"flops_{truncation_str}.npy"), flops)

def run_all_for_architecture():

    parser = argparse.ArgumentParser()

    parser.add_argument("--architecture", type=str,
                        help="Uni, LieUni", required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parserargs = parser.parse_args()

    config = {
        "NUM_LAYERS": 1,
        "SKIP_CONNECTIONS": False,
        "ACTIVATION_FUNCTION": "Identity",
        "BATCH_SIZE": 200,
        "BATCH_NORM": "None",
        "HIDDEN_SIZE": 64
    }

    train_steps = 3
    eval_steps = 1
    start_time = 0.0

    for truncation in tqdm(range(1, 20)):


        for seed in range(0, 10):

            args = SimpleNamespace(
                **config,
                seed=seed,
                data_dir=parserargs.data_dir,
                start_time=start_time,
                train_steps=train_steps,
                eval_steps=eval_steps,
                architecture=parserargs.architecture,
                truncation=truncation,
                verbose=True,
            )

            flops = run_experiment(args, save_dir)



if __name__ == "__main__":

    save_dir = "output"

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = os.path.join(
        save_dir, current_time)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    run_all_for_architecture()

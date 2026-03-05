import logging

import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events
from ignite.metrics import RootMeanSquaredError, RunningAverage
from ignite.utils import setup_logger
from torch import nn

class MLP(nn.Module):
    # MLP with LayerNorm
    def __init__(
        self,
        in_dim,
        out_dim=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type="LayerNorm",
    ):
        """
        MLP

        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        normalize_output: if True, normalize output
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(MLP, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10**7, epsilon=1e-8, device=None):
        """
        Online normalization module

        size: feature dimension
        max_accumulation: maximum number of batches
        epsilon: std cutoff for constant variable
        device: pytorch device
        """

        super(Normalizer, self).__init__()

        self.max_accumulations = max_accumulations
        self.epsilon = torch.tensor(epsilon, dtype=float, device=device)

        self.register_buffer("acc_count", torch.tensor(0.0, dtype=float, device=device))
        self.register_buffer(
            "num_accumulations", torch.tensor(0.0, dtype=float, device=device)
        )
        self.register_buffer("acc_sum", torch.zeros(size, dtype=float, device=device))
        self.register_buffer(
            "acc_sum_squared", torch.zeros(size, dtype=float, device=device)
        )

    def forward(self, batched_data, accumulate=True):
        """
        Updates mean/standard deviation and normalizes input data

        batched_data: batch of data
        accumulate: if True, update accumulation statistics
        """
        if accumulate and self.num_accumulations < self.max_accumulations:
            self._accumulate(batched_data)

        out = (batched_data - self._mean().to(batched_data.device)) / self._std()

        out = out.to(batched_data.device, dtype=batched_data.dtype)

        return out

    def inverse(self, normalized_batch_data):
        """
        Unnormalizes input data
        """

        return normalized_batch_data * self._std().to(
            normalized_batch_data.device
        ) + self._mean().to(normalized_batch_data.device)

    def _accumulate(self, batched_data):
        """
        Accumulates statistics for mean/standard deviation computation
        """
        count = torch.tensor(batched_data.shape[0]).float()
        data_sum = torch.sum(batched_data, dim=0)
        squared_data_sum = torch.sum(batched_data**2, dim=0)

        self.acc_sum += data_sum.to(self.acc_sum.device)
        self.acc_sum_squared += squared_data_sum.to(self.acc_sum_squared.device)
        self.acc_count += count.to(self.acc_count.device)
        self.num_accumulations += 1

    def _mean(self):
        """
        Returns accumulated mean
        """
        safe_count = torch.max(self.acc_count, torch.tensor(1.0).float())

        return self.acc_sum / safe_count

    def _std(self):
        """
        Returns accumulated standard deviation
        """
        safe_count = torch.max(self.acc_count, torch.tensor(1.0).float())
        std = torch.sqrt(self.acc_sum_squared / safe_count - self._mean() ** 2)

        std = torch.max(std, self.epsilon.to(self.acc_sum_squared.device))

        return std


class PDENormalizeRegressor(nn.Module):
    def __init__(self, backbone, node_in_dim, edge_in_dim, out_dim, output_order):
        super().__init__()
        self.backbone = backbone
        self.output_order = output_order

        self._node_normalizer = Normalizer(size=node_in_dim)
        self._edge_normalizer = Normalizer(size=edge_in_dim)
        self._output_normalizer = Normalizer(size=out_dim)

    def forward(self, data):
        with torch.no_grad():
            data.x = self._node_normalizer(
                data.x.squeeze(-1), accumulate=self.training
            ).unsqueeze(-1)

            data.edge_attr = self._edge_normalizer(
                data.edge_attr, accumulate=self.training
            )

        x = self.backbone(data)

        # Take trivial feature
        x = x[:, :, 0]

        return x

    def predict(self, output, current_state, previous_state):
        """
        Default state update function;
        Extend and override this function, or add as a dataset class attribute

        mgn_output_np: MGN output
        current_state: Current state
        previous_state: Previous state (for acceleration-based updates)
        source_data: Source/scripted node data
        """

        with torch.no_grad():
            if self.output_order == 2:
                assert current_state is not None
                assert previous_state is not None
                next_state = 2 * current_state - previous_state + output
            elif self.output_order == 1:
                assert current_state is not None
                next_state = current_state + output
            else:  # state
                next_state = output.copy()

        return next_state


class PDEEncDecNormalizeRegressor(nn.Module):
    def __init__(
        self,
        backbone,
        node_enc_in_dim,
        node_enc_out_dim,
        node_enc_hid_dim,
        node_enc_num_layers,
        edge_enc_in_dim,
        edge_enc_out_dim,
        edge_enc_hid_dim,
        edge_enc_num_layers,
        node_dec_in_dim,
        node_dec_out_dim,
        node_dec_hid_dim,
        node_dec_num_layers,
        output_order,
        normalize=True,
    ):
        super().__init__()
        self.backbone = backbone
        self.output_order = output_order
        self.normalize = normalize

        self.node_encoder = MLP(
            node_enc_in_dim,
            node_enc_out_dim,
            node_enc_hid_dim,
            node_enc_num_layers,
            norm_type="LayerNorm",
        )
        self.edge_encoder = MLP(
            edge_enc_in_dim,
            edge_enc_out_dim,
            edge_enc_hid_dim,
            edge_enc_num_layers,
            norm_type="LayerNorm",
        )
        self.node_decoder = MLP(
            node_dec_in_dim,
            node_dec_out_dim,
            node_dec_hid_dim,
            node_dec_num_layers,
            norm_type=None,
        )

        if normalize:
            self._node_normalizer = Normalizer(size=node_enc_in_dim)
            self._edge_normalizer = Normalizer(size=edge_enc_in_dim)
            self._output_normalizer = Normalizer(size=node_dec_out_dim)

    def forward(self, data):
        if self.normalize:
            data.x = self.node_encoder(
                self._node_normalizer(data.x.squeeze(-1), accumulate=self.training)
            ).unsqueeze(-1)
        else:
            data.x = self.node_encoder(data.x.squeeze(-1)).unsqueeze(-1)

        if data.edge_attr is not None:
            if self.normalize:
                data.edge_attr = self.edge_encoder(
                    self._edge_normalizer(data.edge_attr, accumulate=self.training)
                )
            else:
                data.edge_attr = self.edge_encoder(data.edge_attr)

        x = self.backbone(data)

        # Take trivial feature
        x = x[:, :, 0]

        x = self.node_decoder(x)

        return x

    def predict(self, output, current_state, previous_state):
        """
        Default state update function;
        Extend and override this function, or add as a dataset class attribute

        mgn_output_np: MGN output
        current_state: Current state
        previous_state: Previous state (for acceleration-based updates)
        source_data: Source/scripted node data
        """

        with torch.no_grad():
            if self.output_order == 2:
                assert current_state is not None
                assert previous_state is not None
                next_state = 2 * current_state - previous_state + output
            elif self.output_order == 1:
                assert current_state is not None
                next_state = current_state + output
            else:  # state
                next_state = output.copy()

        return next_state


class FlagEngine:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        prepare_batch,
        loader_keys,
        disable_tqdm,
        normalize,
        out_dim=None,
        eval_every=1,
        grad_accum_steps=1,
    ):
        self.normalize = normalize
        if normalize:
            self._output_normalizer = Normalizer(size=out_dim)

        def train_step(engine, batch):
            if (engine.state.iteration - 1) % grad_accum_steps == 0:
                optimizer.zero_grad()
            model.train()

            x, y = prepare_batch(batch, device=device)
            y_pred = model(x)

            if self.normalize:
                y = self._output_normalizer(y, accumulate=True)

            loss = loss_fn(y_pred, y)

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            loss.backward()
            if engine.state.iteration % grad_accum_steps == 0:
                optimizer.step()
            return torch.sqrt(loss).item()

        self.trainer = Engine(train_step)

        self.loader_keys = loader_keys
        self.eval_every = eval_every

        RunningAverage(output_transform=lambda x: x).attach(self.trainer, "loss")
        ProgressBar(disable=disable_tqdm).attach(self.trainer, ["loss"])

        def eval_step(engine, batch):
            model.eval()
            with torch.no_grad():
                x, y = prepare_batch(batch, device=device)
                y_pred = model(x)

                if self.normalize:
                    y = self._output_normalizer(y, accumulate=False)

                return y, y_pred

        self.evaluators = {}
        for k in self.loader_keys:
            if k == "train":
                continue

            self.evaluators[k] = Engine(eval_step)

            metric = RootMeanSquaredError()

            metric.attach(self.evaluators[k], "rmse")

            RunningAverage(metric).attach(self.evaluators[k], "running_rmse")
            ProgressBar(persist=False, desc=k.upper(), disable=disable_tqdm).attach(
                self.evaluators[k], ["running_rmse"]
            )

    def set_epoch_loggers(self, loaders_dict):
        # Setup logging level
        setup_logger(name="ignite", level=logging.WARNING)
        self.trainer.logger = setup_logger(name="trainer", level=logging.WARNING)
        for k, evaluator in self.evaluators.items():
            evaluator.logger = setup_logger(name=k, level=logging.WARNING)

        def inner_log(engine, evaluator, tag):
            evaluator.run(loaders_dict[tag])
            metrics = evaluator.state.metrics
            print(
                f"{tag.upper()} Results - Epoch: {engine.state.epoch} "
                f"Avg rmse: {metrics['rmse']:.5E}"
            )

        # Evaluate over loaders_dict
        @self.trainer.on(Events.EPOCH_COMPLETED(every=self.eval_every))
        def log_results(engine):
            for k in self.loader_keys:
                if k == "train":
                    continue
                if loaders_dict[k] is not None:
                    inner_log(engine, self.evaluators[k], k)

    def create_wandb_logger(self, log_interval=1, optimizer=None, **kwargs):
        wandb_logger = WandBLogger(**kwargs)

        # Attach the logger to the trainer to log training loss at each iteration
        wandb_logger.attach_output_handler(
            self.trainer,
            event_name=Events.ITERATION_COMPLETED(every=log_interval),
            tag="train",
            output_transform=lambda loss: {"batch_rmse": loss},
            state_attributes=["epoch"],
        )

        # Attach the logger to the optimizer parameters handler
        wandb_logger.attach_opt_params_handler(
            self.trainer,
            event_name=Events.ITERATION_STARTED(every=1000),
            optimizer=optimizer,
        )

        # Attach logger to evaluator on test dataset
        for k in self.loader_keys:
            if k == "train":
                continue

            metric_names = ["rmse"]

            wandb_logger.attach_output_handler(
                self.evaluators[k],
                event_name=Events.EPOCH_COMPLETED(every=self.eval_every),
                tag=k,
                metric_names=metric_names,
                global_step_transform=lambda *_: self.trainer.state.iteration,
            )

        return wandb_logger

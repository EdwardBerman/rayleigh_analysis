import logging
from math import pow

import torch
import torch.nn.functional as F
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events
from ignite.metrics import RootMeanSquaredError, RunningAverage
from ignite.utils import setup_logger
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from external.custom_hermes.nn.meshgraphnet import MLP

class ExpLR(_LRScheduler):
    """
    Exponential learning rate scheduler
    Based on procedure described in LTS
    If min_lr==0 and decay_steps==1, same as torch.optim.lr_scheduler.ExpLR
    """

    def __init__(
        self, optimizer, decay_steps=10000, gamma=0.1, min_lr=1e-6, last_epoch=-1
    ):
        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.gamma = gamma
        self.decay_steps = decay_steps

        super(ExpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            min_lr
            + max(base_lr - min_lr, 0)
            * pow(self.gamma, self.last_epoch / self.decay_steps)
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]


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


class PDERegressor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, data):
        x = self.backbone(data)

        # Take trivial feature
        x = x[:, :, 0]

        return x


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
        self.edge_enc_in_dim = edge_enc_in_dim
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
            with torch.no_grad():
                data.x = self._node_normalizer(
                    data.x.squeeze(-1), accumulate=self.training
                ).unsqueeze(-1)

        data.x = self.node_encoder(data.x.squeeze(-1)).unsqueeze(-1)

        if data.edge_attr is not None and self.edge_enc_in_dim != 0:
            if self.normalize:
                with torch.no_grad():
                    data.edge_attr = self._edge_normalizer(
                        data.edge_attr, accumulate=self.training
                    )

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


class SpiralNetPDE(nn.Module):
    def __init__(self, in_dim, out_dim, backbone):
        super().__init__()
        self.backbone = backbone

        self.fc0 = nn.Linear(in_dim, backbone.dims[0])
        self.fc1 = nn.Linear(backbone.dims[-1], 128)
        self.fc2 = nn.Linear(128, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.backbone.reset_parameters()
        nn.init.xavier_uniform_(self.fc0.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, data):
        data.x = F.elu(self.fc0(data.x.squeeze(-1)))

        x = self.backbone(data)

        # Take trivial feature
        x = x[:, :, 0]

        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


class MGNPDEEngine:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        prepare_batch,
        loader_keys,
        disable_tqdm,
        eval_every=1,
        grad_accum_steps=1,
    ):
        def train_step(engine, batch):
            if (engine.state.iteration - 1) % grad_accum_steps == 0:
                optimizer.zero_grad()
            model.train()

            x, yy = prepare_batch(batch, device=device)
            cur_edge_attr = x.edge_attr

            loss = 0
            for i in range(yy.shape[1]):
                y = yy[..., i].unsqueeze(-1)
                # Save input data in case it is modified in forward()
                cur_x = x.x

                y_pred = model(x)

                # revert edge_attr
                x.edge_attr = cur_edge_attr

                x.x = torch.cat([cur_x[:, y_pred.shape[1] :, 0], y_pred], 1)[:, :, None]

                loss += loss_fn(y_pred, y)
            loss /= yy.shape[1]

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
            y_preds = []
            ys = []

            model.eval()
            with torch.no_grad():
                x, yy = prepare_batch(batch, device=device)
                cur_edge_attr = x.edge_attr
                for i in range(yy.shape[1]):
                    y = yy[..., i].unsqueeze(-1)
                    # Save input data in case it is modified in forward()
                    cur_x = x.x

                    ys.append(y)
                    y_pred = model(x)

                    # revert edge_attr
                    x.edge_attr = cur_edge_attr

                    y_preds.append(y_pred)

                    x.x = torch.cat([cur_x[:, y_pred.shape[1] :, 0], y_pred], 1)[
                        :, :, None
                    ]

                return torch.cat(y_preds, dim=-1), yy

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

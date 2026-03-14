import logging

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, RunningAverage
from ignite.utils import setup_logger
from torch import nn


class DragForceRegressor(nn.Module):
    """
    Simple wrapper for graph-level regression model for drag force prediction.
    The backbone model itself handles pooling and MLP readout.
    Takes a batch of meshes and predicts a single scalar value per mesh.
    """
    def __init__(self, backbone):
        """
        Args:
            backbone: GNN backbone model (Hermes, GCN, EGNN, etc.)
                     Should include its own pooling and readout layers
                     and output [batch_size, 1] predictions
        """
        super().__init__()
        self.backbone = backbone

    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data or Batch object containing:
                - x: Node features [num_nodes, in_channels]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch assignment [num_nodes] (which graph each node belongs to)
                - pos: Node positions [num_nodes, 3] (optional)
                - Any other attributes needed by the backbone
                
        Returns:
            y_pred: Predicted drag force [batch_size, 1]
        """
        # Backbone handles everything: message passing, pooling, readout
        y_pred = self.backbone(data)
        
        return y_pred


class DragForceEngine:
    """
    Training and evaluation engine for drag force prediction.
    Non-autoregressive, single-step graph-level regression.
    """
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        prepare_batch,
        loader_keys,
        disable_tqdm,
        grad_accum_steps=1,
    ):
        self.loader_keys = loader_keys

        def train_step(engine, batch):
            """Single training step."""
            if (engine.state.iteration - 1) % grad_accum_steps == 0:
                optimizer.zero_grad()
            
            model.train()
            
            # Prepare batch and move to device
            data = prepare_batch(batch, device=device)
            
            # Forward pass
            y_pred = model(data)  # [batch_size, 1]
            y_true = data.y  # [batch_size, 1]
            
            # Compute loss
            loss = loss_fn(y_pred, y_true)
            
            # Gradient accumulation
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            
            loss.backward()
            
            if engine.state.iteration % grad_accum_steps == 0:
                optimizer.step()
            
            # Return metrics for logging
            mse = loss.item() ** 2 if grad_accum_steps == 1 else (loss.item() * grad_accum_steps) ** 2
            mae = torch.mean(torch.abs(y_pred - y_true)).item()
            
            return {
                'loss': loss.item(),
                'mse': mse,
                'mae': mae,
            }

        self.trainer = Engine(train_step)

        # Attach running averages for training metrics
        RunningAverage(output_transform=lambda x: x['loss']).attach(self.trainer, "loss")
        RunningAverage(output_transform=lambda x: x['mse']).attach(self.trainer, "mse")
        RunningAverage(output_transform=lambda x: x['mae']).attach(self.trainer, "mae")
        
        ProgressBar(disable=disable_tqdm).attach(self.trainer, ["loss", "mse", "mae"])

        def eval_step(engine, batch):
            """Single evaluation step."""
            model.eval()
            
            with torch.no_grad():
                data = prepare_batch(batch, device=device)
                
                # Forward pass
                y_pred = model(data)  # [batch_size, 1]
                y_true = data.y  # [batch_size, 1]
                
            return y_pred, y_true

        # Create evaluators for each dataset split
        self.evaluators = {}
        for k in self.loader_keys:
            if k == "train":
                continue

            self.evaluators[k] = Engine(eval_step)

            # Attach metrics
            rmse = RootMeanSquaredError()
            rmse.attach(self.evaluators[k], "rmse")
            
            mse = MeanSquaredError()
            mse.attach(self.evaluators[k], "mse")
            
            mae = MeanAbsoluteError()
            mae.attach(self.evaluators[k], "mae")
            
            # Running average for progress bar
            RunningAverage(rmse).attach(self.evaluators[k], "running_rmse")
            
            ProgressBar(persist=False, desc=k.upper(), disable=disable_tqdm).attach(
                self.evaluators[k], ["running_rmse"]
            )

    def set_epoch_loggers(self, loaders_dict):
        """Set up logging for each epoch."""
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
                f"RMSE: {metrics['rmse']:.5E} | "
                f"MSE: {metrics['mse']:.5E} | "
                f"MAE: {metrics['mae']:.5E}"
            )

        # Evaluate over loaders_dict
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_results(engine):
            for k in self.loader_keys:
                if k == "train":
                    continue
                if loaders_dict[k] is not None:
                    inner_log(engine, self.evaluators[k], k)

    def create_wandb_logger(self, log_interval=1, optimizer=None, **kwargs):
        """Create WandB logger for experiment tracking."""
        wandb_logger = WandBLogger(**kwargs)

        # Attach the logger to the trainer to log training loss at each iteration
        wandb_logger.attach_output_handler(
            self.trainer,
            event_name=Events.ITERATION_COMPLETED(every=log_interval),
            tag="train",
            output_transform=lambda output: {
                "loss": output['loss'],
                "mse": output['mse'],
                "mae": output['mae'],
            },
            state_attributes=["epoch"],
        )

        # Attach the logger to the optimizer parameters handler
        wandb_logger.attach_opt_params_handler(
            self.trainer,
            event_name=Events.ITERATION_STARTED(every=1000),
            optimizer=optimizer,
        )

        # Attach logger to evaluators
        for k in self.loader_keys:
            if k == "train":
                continue
            wandb_logger.attach_output_handler(
                self.evaluators[k],
                event_name=Events.EPOCH_COMPLETED,
                tag=k,
                metric_names=["rmse", "mse", "mae"],
                global_step_transform=lambda *_: self.trainer.state.iteration,
            )

        return wandb_logger


def prepare_batch_drag_force(batch, device):
    """
    Prepare batch for drag force prediction.
    
    Args:
        batch: PyTorch Geometric Batch object
        device: torch device
        
    Returns:
        batch moved to device
    """
    return batch.to(device)

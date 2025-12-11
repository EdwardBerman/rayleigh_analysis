import os
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, cast

import shutil
from datetime import datetime
from glob import glob
import sys

import debugpy
import hydra
import jax
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from typer import Argument, Typer

def delete_old_results(results_dir, force, trial, resume):
    """Delete existing checkpoints and wandb logs if --force is enabled."""
    wandb_dir = Path(results_dir) / 'wandb'
    wandb_matches = list(wandb_dir.glob(f'*-trial-{trial}-*'))

    chkpt_dir = Path(results_dir) / 'checkpoints'
    chkpt_matches = list(chkpt_dir.glob(f'trial-{trial}-*'))

    if force and wandb_matches:
        [shutil.rmtree(p) for p in wandb_matches]

    if force and chkpt_matches:
        [shutil.rmtree(p) for p in chkpt_matches]

    if not force and not resume and wandb_matches:
        raise ExistingExperimentFound(f'Directory already exists: {wandb_dir}')

    if not force and not resume and chkpt_matches:
        raise ExistingExperimentFound(f'Directory already exists: {chkpt_dir}')

def get_experiment_id(checkpoint_id, trial, save_dir, resume):
    chkpt_dir = Path(save_dir) / 'checkpoints'
    if resume and not checkpoint_id and chkpt_dir.exists:
        paths = chkpt_dir.glob('*/last.ckpt')
        checkpoint_id = next(paths).parent.name
    now = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    return checkpoint_id or f'trial-{trial}-{now}'

def import_string(dotted_path):
    """Import a dotted module path.

    And return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.

    Adatped from https://stackoverflow.com/a/34963527/3790116.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError as e:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError.with_traceback(ImportError(msg), sys.exc_info()[2])

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (
            module_path, class_name)
        raise ImportError.with_traceback(ImportError(msg), sys.exc_info()[2])

def upload_code_to_wandb(config_path, wandb_logger):
    """Upload all Python code for save the exact state of the experiment."""
    code_artifact = wandb.Artifact('fourierflow', type='code')
    code_artifact.add_file(config_path, 'config.yaml')
    paths = glob('fourierflow/**/*.py')
    for path in paths:
        code_artifact.add_file(path, path)
    wandb_logger.experiment.log_artifact(code_artifact)

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Path,
         overrides: Optional[List[str]] = Argument(None),
         force: bool = False,
         resume: bool = False,
         checkpoint_id: Optional[str] = None,
         trial: int = 0,
         debug: bool = False,
         no_logging: bool = False):
    """Train a Pytorch Lightning experiment."""
    config_dir = config_path.parent
    config_name = config_path.stem
    hydra.initialize(config_path=Path('../..') /
                     config_dir, version_base='1.2')
    config = hydra.compose(config_name, overrides=overrides)
    OmegaConf.set_struct(config, False)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        debugpy.listen(5678)
        debugpy.wait_for_client()
        # debugger doesn't play well with multiple processes.
    config.builder.num_workers = 0
    jax.config.update('jax_disable_jit', True)
    # jax.config.update("jax_debug_nans", True)

    # Set up directories to save experimental outputs.
    delete_old_results(config_dir, force, trial, resume)

    # Set seed for reproducibility.
    rs = np.random.RandomState(7231 + trial)
    seed = config.get('seed', rs.randint(1000, 1000000))
    pl.seed_everything(seed, workers=True)
    config.seed = seed
    wandb_id = get_experiment_id(checkpoint_id, trial, config_dir, resume)
    config.trial = trial
    if 'seed' in config.trainer:
        config.trainer.seed = seed

    # Initialize the dataset and experiment modules.
    builder = instantiate(config.builder)
    routine = instantiate(config.routine)

    # Support fine-tuning mode if a pretrained model path is supplied.
    pretrained_path = config.get('pretrained_path', None)
    if pretrained_path:
        routine.load_lightning_model_state(pretrained_path)

    # Resume from last checkpoint. We assume that the checkpoint file is from
    # the end of the previous epoch. The trainer will start the next epoch.
    # Resuming from the middle of an epoch is not yet supported. See:
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/5325
    chkpt_path = Path(config_dir) / 'checkpoints' / wandb_id / 'last.ckpt' \
        if resume else None

    # Initialize the main trainer.
    callbacks = [instantiate(p) for p in config.get('callbacks', [])]
    multi_gpus = config.trainer.get('gpus', 0) > 1
    plugins = DDPPlugin(find_unused_parameters=False) if multi_gpus else None

    # Strange bug: We need to check if cuda is availabe first; otherwise,
    # sometimes lightning's CUDAAccelerator.is_available() returns false :-/
    torch.cuda.is_available()

    if no_logging:
        logger = False
        enable_checkpointing = False
        callbacks = []
    else:
        # We use Weights & Biases to track our experiments.
        # Clean cache before creating any new artifacts
        c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
        c.cleanup(wandb.util.from_human_size("200GB"))

        config.wandb.name = f"{config.wandb.group}/{trial}"
        wandb_opts = cast(dict, OmegaConf.to_container(config.wandb))
        logger = WandbLogger(save_dir=str(config_dir),
                             mode=os.environ.get('WANDB_MODE', 'offline'),
                             config=deepcopy(OmegaConf.to_container(config)),
                             id=wandb_id,
                             **wandb_opts)
        upload_code_to_wandb(Path(config_dir) / 'config.yaml', logger)
        enable_checkpointing = True

    Trainer = import_string(config.trainer.pop(
        '_target_', 'pytorch_lightning.Trainer'))
    trainer = Trainer(logger=logger,
                      enable_checkpointing=enable_checkpointing,
                      callbacks=callbacks,
                      plugins=plugins,
                      resume_from_checkpoint=chkpt_path,
                      enable_model_summary=False,
                      **OmegaConf.to_container(config.trainer))

    # Tuning only has an effect when either auto_scale_batch_size or
    # auto_lr_find is set to true.
    trainer.tune(routine, datamodule=builder)
    trainer.fit(routine, datamodule=builder)

    # Load best checkpoint before testing.
    chkpt_dir = Path(config_dir) / 'checkpoints'
    paths = list(chkpt_dir.glob(f'trial-{trial}-*/epoch*.ckpt'))
    assert len(paths) == 1
    checkpoint_path = paths[0]
    routine.load_lightning_model_state(str(checkpoint_path))
    trainer.test(routine, datamodule=builder)

    # Compute inference time
    batch = builder.inference_data()
    if logger and batch is not None:
        T = batch['data'].shape[-1]
        n_steps = routine.n_steps or (T - 1)
        routine = routine.cuda()
        batch = routine.convert_data(batch)
        routine.warmup()

        start = time.time()
        routine.infer(batch)
        elapsed = time.time() - start

        elapsed /= len(batch['data'])
        elapsed /= routine.step_size * n_steps
        logger.experiment.log({'inference_time': elapsed})


if __name__ == "__main__":
    app()

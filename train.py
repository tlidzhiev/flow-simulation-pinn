import logging
from pathlib import Path
from typing import Any, Callable

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.dataset.utils import get_dataloaders
from src.logger.base import BaseWriter
from src.loss.base import BaseLoss
from src.metrics.tracker import MetricTracker
from src.utils.jax import get_optimizer

logger = logging.getLogger(Path(__file__).name)


@nnx.jit(static_argnums=(3,))
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: dict[str, Any],
    criterion: Callable,
) -> tuple[jax.Array, dict[str, Any]]:
    def loss_fn(model):
        return criterion(model=model, batch=batch)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_dict), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss, loss_dict


@hydra.main(version_base=None, config_path='src/configs', config_name='train')
def main(cfg: DictConfig) -> None:
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Train and
    evaluate the model.

    Parameters
    cfg : DictConfig
        Hydra experiment config.
    """
    project_config = OmegaConf.to_container(cfg, resolve=True)
    writer: BaseWriter = instantiate(cfg.writer, project_config)

    dataloader = get_dataloaders(cfg)

    model: nnx.Module = instantiate(cfg.model)
    logger.info(f'Model:\n{model}')

    optimizer: nnx.Optimizer = get_optimizer(cfg, model)
    logger.info(f'Optimizer:\n{optimizer}')

    criterion: BaseLoss = instantiate(cfg.criterion)
    logger.info(f'Criterion: {criterion}')

    metric_tracker = MetricTracker(*criterion.loss_names, *criterion.loss_weight_names)
    pbar = tqdm(enumerate(dataloader), total=cfg.trainer.num_steps, desc='Training')
    for epoch, batch in pbar:
        writer.set_step(epoch)

        loss, loss_dict = train_step(model, optimizer, batch, criterion)
        for k, v in loss_dict.items():
            metric_tracker.update(k, v)

        if epoch % cfg.trainer.log_interval == 0:
            pbar.set_postfix(
                {
                    'loss': f'{float(loss):.6f}',
                    'pde': f'{float(loss_dict["loss_pde"]):.6f}',
                    'ic': f'{float(loss_dict["loss_ic"]):.6f}',
                    'bc': f'{float(loss_dict["loss_bc"]):.6f}',
                }
            )
            for name in metric_tracker.keys():
                writer.add_scalar(name, metric_tracker[name])

    t = jnp.linspace(0, 1, 100)
    x = jnp.linspace(0, 2, 100)
    t, x = jnp.meshgrid(t, x, indexing='ij')
    t, x = t.flatten(), x.flatten()

    u_pred = model(t=t.reshape(-1, 1), x=x.reshape(-1, 1))['u']
    u_pred = u_pred.reshape((100, 100))

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(u_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Predicted Solution')
    plt.show()
    fig.savefig('predicted_solution.png')
    writer.close()


if __name__ == '__main__':
    main()

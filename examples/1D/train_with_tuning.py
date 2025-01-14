from pathlib import Path
from typing import Callable, Dict

import hydra
import numpy as np
import optuna
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from optuna import Trial

from pinns.data.datasets import SimulationDataset
from pinns.logger import Logger
from pinns.trainers.base import BaseTrainer
from pinns.utils.init_utils import set_random_seed

CURRENT_DIR = Path(__file__).absolute().resolve().parent

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def get_datasets(cfg: DictConfig):
    train_dataset = np.load(CURRENT_DIR / 'data/train/data.npy', allow_pickle=True)[0]
    train_dataset = instantiate(cfg['dataset'], **train_dataset).to(DEVICE)

    test_dataset = np.load(CURRENT_DIR / 'data/test/data.npy', allow_pickle=True)[0]
    test_dataset = SimulationDataset(**test_dataset['simulation_data']).to(DEVICE)
    return train_dataset, test_dataset


def objective(trial: Trial, cfg: DictConfig) -> float:
    set_random_seed(17989380322705251799)
    cfg['trainer']['config']['ic_weight'] = trial.suggest_float('ic_weight', 1.0e-2, 10.0, log=True)
    cfg['trainer']['config']['bc_weight'] = trial.suggest_float('bc_weight', 1.0e-2, 10.0, log=True)
    cfg['trainer']['config']['sim_weight'] = trial.suggest_float('sim_weight', 1.0e-2, 10.0, log=True)
    cfg['optimizer']['lr'] = trial.suggest_float('lr', 1.0e-5, 1.0, log=True)
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')

    train_dataset, test_dataset = get_datasets(cfg)
    model: nn.Module = instantiate(cfg['model']).to(DEVICE)
    criterion: Dict[str, Callable] = {name: instantiate(loss_cfg) for name, loss_cfg in cfg['criterion'].items()}
    optimizer = instantiate(cfg['optimizer'], params=model.parameters())
    lr_scheduler = instantiate(cfg['scheduler'], optimizer=optimizer, T_max=cfg['trainer']['config']['num_epochs'])
    logger = Logger(project_config=OmegaConf.to_container(cfg), **cfg['logger'])
    trainer: BaseTrainer = instantiate(
        cfg['trainer'],
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
    )

    trainer.fit(train_dataset, test_dataset)
    return logger.metrics_history['eval_loss_data'][-1]


@hydra.main(version_base='1.3', config_path='configs', config_name='config.yaml')
def main(cfg: DictConfig):
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(lambda trial: objective(trial, cfg), n_trials=10)


if __name__ == '__main__':
    main()

from typing import Callable, Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm

from pinns.logger import Logger
from pinns.utils.io_utils import get_root


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: Dict[str, Callable],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        logger: Logger,
        config: DictConfig,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger

        self.trainer_config = config
        self.num_epochs = self.trainer_config['num_epochs']
        self.save_period = self.trainer_config['save_period']
        self.checkpoint_dir = get_root() / self.trainer_config['checkpoint_dir'] / self.logger.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, train_dataset, val_dataset):
        for epoch in tqdm(range(1, self.num_epochs + 1), desc='Train', total=self.num_epochs):
            train_result = self._train_epoch(train_dataset)
            eval_result = self._eval_epoch(val_dataset)

            self.logger.add_scalars({**train_result, **eval_result})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, train_dataset) -> Dict[str, float]:
        raise NotImplementedError('Subclass must implement _train_epoch method')

    def _eval_epoch(self, val_dataset) -> Dict[str, float]:
        raise NotImplementedError('Subclass must implement _eval_epoch method')

    def _clip_grad_norm(self):
        if self.trainer_config.get('max_grad_norm', None) is not None:
            clip_grad_norm_(self.model.parameters(), self.trainer_config['max_grad_norm'])

    @torch.no_grad()
    def _get_grad_norm(self, norm_type: int = 2) -> float:
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _save_checkpoint(self, epoch: int):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        filename = str(self.checkpoint_dir / f'checkpoint-epoch-{epoch}.pth')
        torch.save(state, filename)
        self.logger.add_checkpoint(checkpoint_path=filename, save_dir=str(self.checkpoint_dir))

from typing import Dict

import numpy as np
import torch

import pinns

from .base import BaseTrainer


class DataDrivenTrainer(BaseTrainer):
    def _train_epoch(self, train_dataset) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        ic_data, bc_data, pde_data, sim_data = train_dataset.get_data()

        ic_output = self.model(t=ic_data.t, x=ic_data.x)
        ic_loss = self.criterion['loss_data'](output=ic_output, target=ic_data.target)

        bc_output = self.model(t=bc_data.t, x=bc_data.x)
        bc_loss = self.criterion['loss_data'](output=bc_output, target=bc_data.target)

        sim_output = self.model(t=sim_data.t, x=sim_data.x)
        sim_loss = self.criterion['loss_data'](output=sim_output, target=sim_data.target)

        loss: torch.Tensor = (
            self.trainer_config['ic_weight'] * ic_loss
            + self.trainer_config['bc_weight'] * bc_loss
            + self.trainer_config['sim_weight'] * sim_loss
        )
        loss.backward()

        self._clip_grad_norm()
        grad_norm = self._get_grad_norm()
        self.optimizer.step()

        result = {
            'train_loss_total': loss.item(),
            'train_loss_ic': ic_loss.item(),
            'train_loss_bc': bc_loss.item(),
            'train_loss_data': sim_loss.item(),
            'grad_norm': grad_norm,
        }
        return {key: np.log10(value) for key, value in result.items()}

    def _eval_epoch(self, val_dataset) -> Dict[str, float]:
        self.model.eval()
        self.optimizer.zero_grad()

        output = self.model(t=val_dataset.t, x=val_dataset.x)
        output['dp_dx'] = pinns.utils.gradient(output['p'], val_dataset.x)[0]
        loss = self.criterion['loss_data'](output, val_dataset.target)

        self.optimizer.zero_grad()
        return {'eval_loss_data': np.log10(loss.item())}

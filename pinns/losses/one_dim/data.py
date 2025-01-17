from typing import Dict

import torch
import torch.nn as nn


class DataLoss(nn.Module):
    def __init__(self, kappa: float = 10.0):
        super().__init__()
        self.kappa = kappa

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        constraint_loss = ((output['sw'] + output['so'] - 1.0) ** 2).mean()
        loss = 0.0
        for name in target.keys():
            loss += ((output[name] - target[name]) ** 2).mean()
            if name in ('uo',):
                # `kappa` is for normalization, because velocity is smaller than other physics parameters
                loss += ((output[name] - target[name]) ** 2).mean() * self.kappa
        return constraint_loss + loss

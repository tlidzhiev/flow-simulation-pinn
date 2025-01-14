from typing import List

import torch
import torch.nn as nn

import pinns.utils


class LossBalancer:
    """
    Implementation of adaptive loss balancing for multi-objective optimization.
    Based on: https://arxiv.org/pdf/2308.08468 --- Algorithm 1
    """

    def __init__(self, loss_number: int, alpha: float = 0.9, device: str = 'cpu'):
        self.alpha = alpha
        self.loss_number = loss_number
        self.device = device
        self.lambdas = torch.ones(self.loss_number, dtype=torch.float32).to(device)

    def update_weights(
        self,
        losses: List[torch.Tensor],
        model: nn.Module,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        if len(losses) != self.loss_number:
            raise ValueError(f'Expected {self.loss_number} losses, got {len(losses)}')

        grad_norms = torch.zeros_like(self.lambdas, dtype=torch.float32)
        with torch.no_grad():
            for i, loss in enumerate(losses):
                loss_grads = pinns.utils.gradient(
                    outputs=loss,
                    inputs=list(model.parameters()),
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_norms[i] = torch.sqrt(sum(torch.sum(g * g) for g in loss_grads)).item()
            sum_grad_norms = torch.sum(grad_norms)
            new_lambdas = sum_grad_norms / (grad_norms + eps)

        total_loss = sum([self.lambdas[i] * loss for i, loss in enumerate(losses)])
        with torch.no_grad():
            self.lambdas = self.alpha * self.lambdas + (1.0 - self.alpha) * new_lambdas
        return total_loss

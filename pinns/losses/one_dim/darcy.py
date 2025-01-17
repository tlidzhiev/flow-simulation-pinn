import torch
import torch.nn as nn

import pinns.utils


class DarcyLoss(nn.Module):
    def __init__(
        self,
        alpha_wat: float = 2.0,
        alpha_oil: float = 4.0,
        mu_water: float = 1.0,
        mu_oil: float = 3.0,
    ):
        super().__init__()
        self.alpha_wat = alpha_wat
        self.alpha_oil = alpha_oil
        self.mu_water = mu_water
        self.mu_oil = mu_oil

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        output: dict[str, torch.Tensor],
    ):
        return self._compute_darcy_law_loss(t, x, output)

    def _compute_darcy_law_loss(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        p = output['p']
        sw, so = output['sw'], output['so']
        uw, uo = output['uw'], output['uo']

        dp_dx = pinns.utils.gradient(
            outputs=p,
            inputs=x,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

        water = uw + (self.k_wat(sw, self.alpha_wat) / self.mu_water) * dp_dx
        oil = uo + (self.k_oil(so, self.alpha_oil) / self.mu_oil) * dp_dx
        return (water**2).mean() + (oil**2).mean()

    @staticmethod
    def k_wat(s, alpha_wat=2.0, k=1.0):
        eps = 1.0e-10
        sp = (s + eps) / (1.0 + eps)
        return k * (sp**alpha_wat)

    @staticmethod
    def k_oil(s, alpha_oil=4.0, k=0.1):
        eps = 1.0e-10
        sp = (s + eps) / (1.0 + eps)
        return k * (sp**alpha_oil)

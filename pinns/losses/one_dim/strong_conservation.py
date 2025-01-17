from typing import Dict

import torch
import torch.nn as nn

import pinns.utils


class StrongConservationLoss(nn.Module):
    def __init__(self, phi: float = 0.1):
        super().__init__()
        self.phi = phi

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._compute_conservation_law_loss(t, x, output)

    def _compute_conservation_law_loss(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        sw, so = output['sw'], output['so']
        uw, uo = output['uw'], output['uo']
        dsw_dt = pinns.utils.gradient(
            outputs=sw,
            inputs=t,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        dso_dt = pinns.utils.gradient(
            outputs=so,
            inputs=t,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

        duw_dx = pinns.utils.gradient(
            outputs=uw,
            inputs=x,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        duo_dx = pinns.utils.gradient(
            outputs=uo,
            inputs=x,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

        water = self.phi * dsw_dt + duw_dx
        oil = self.phi * dso_dt + duo_dx
        return (water**2).mean() + (oil**2).mean()

from typing import Tuple

import torch
import torch.nn as nn

from pinns.data.datasets import SpatialQuadratureDataset, TimeQuadratureDataset


class WeakConservationLoss(nn.Module):
    def __init__(self, phi: float = 0.1, normalize: bool = False):
        super().__init__()
        self.phi = phi
        self.normalize = normalize

        self.water_mass, self.oil_mass = None, None
        self.water_flux, self.oil_flux = None, None

    def forward(
        self,
        t_quads: TimeQuadratureDataset,
        x_quads: SpatialQuadratureDataset,
        model: nn.Module,
    ) -> torch.Tensor:
        return self._compute_conservation_law_loss(
            t_quads=t_quads,
            x_quads=x_quads,
            model=model,
        )

    def _compute_conservation_law_loss(
        self,
        t_quads: TimeQuadratureDataset,
        x_quads: SpatialQuadratureDataset,
        model: nn.Module,
    ) -> torch.Tensor:
        volumes = 1.0
        if self.normalize:
            volumes = t_quads.volumes * x_quads.volumes

        self.water_mass, self.oil_mass = self._compute_liquid_mass(t_quads=t_quads, x_quads=x_quads, model=model)
        self.water_flux, self.oil_flux = self._compute_velocity_flux(t_quads=t_quads, x_quads=x_quads, model=model)

        water = (self.water_mass + self.water_flux) / volumes
        oil = (self.oil_mass + self.oil_flux) / volumes

        return (water**2).mean() + (oil**2).mean()

    def _compute_liquid_mass(
        self,
        t_quads: TimeQuadratureDataset,
        x_quads: SpatialQuadratureDataset,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_values = model(t=t_quads.t_lower.view(-1, 1), x=x_quads.x.view(-1, 1))
        upper_values = model(t=t_quads.t_upper.view(-1, 1), x=x_quads.x.view(-1, 1))

        water_mass = self._integrate_density(
            x_quads=x_quads,
            lower_saturation=lower_values['sw'],
            upper_saturation=upper_values['sw'],
        )
        oil_mass = self._integrate_density(
            x_quads=x_quads,
            lower_saturation=lower_values['so'],
            upper_saturation=upper_values['so'],
        )
        return water_mass, oil_mass

    def _integrate_density(
        self,
        x_quads: SpatialQuadratureDataset,
        lower_saturation: torch.Tensor,
        upper_saturation: torch.Tensor,
    ) -> torch.Tensor:
        liquid_values = self.phi * (upper_saturation - lower_saturation)
        liquid_values = liquid_values.view(x_quads.num_bounds, x_quads.num_nodes, 1)
        liquid_weighted = liquid_values * x_quads.weights
        liquid_mass = liquid_weighted.sum(dim=1) * x_quads.normalize_scalars
        return liquid_mass

    def _compute_velocity_flux(
        self,
        t_quads: TimeQuadratureDataset,
        x_quads: SpatialQuadratureDataset,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_values = model(t=t_quads.t.view(-1, 1), x=x_quads.x_lower.view(-1, 1))
        upper_values = model(t=t_quads.t.view(-1, 1), x=x_quads.x_upper.view(-1, 1))

        water_flux = self._integrate_velocity(
            t_quads=t_quads, lower_velocity=lower_values['uw'], upper_velocity=upper_values['uw']
        )

        oil_flux = self._integrate_velocity(
            t_quads=t_quads, lower_velocity=lower_values['uo'], upper_velocity=upper_values['uo']
        )
        return water_flux, oil_flux

    def _integrate_velocity(
        self,
        t_quads: TimeQuadratureDataset,
        lower_velocity: torch.Tensor,
        upper_velocity: torch.Tensor,
    ) -> torch.Tensor:
        liquid_values = upper_velocity - lower_velocity
        liquid_values = liquid_values.view(t_quads.num_bounds, t_quads.num_nodes, 1)
        liquid_weighted = liquid_values * t_quads.weights
        liquid_flux = liquid_weighted.sum(dim=1) * t_quads.normalize_scalars
        return liquid_flux

# import torch
# import torch.nn as nn

# from pinns.data.datasets import SpatialQuadratureDataset, TimeQuadratureDataset


# class WeakConservationLoss:
#     def __init__(self, phi: float = 0.1):
#         self.phi = phi
#         self.tquads: TimeQuadratureDataset = None
#         self.xquads: SpatialQuadratureDataset = None

#     def __call__(
#         self,
#         t: TimeQuadratureDataset,
#         x: SpatialQuadratureDataset,
#         model: nn.Module,
#     ) -> torch.Tensor:
#         self.tquads = t
#         self.xquads = x
#         return self._compute_conservation_law_loss(model=model)

#     def _compute_conservation_law_loss(self, model: nn.Module) -> torch.Tensor:
#         volumes = self.tquads.volumes * self.xquads.volumes

#         water_mass, oil_mass = self._compute_liquid_mass(model=model)
#         water_flux, oil_mass_flux = self._compute_speed_flux_over_time(model=model)

#         water = (water_mass + water_flux) / volumes
#         oil = (oil_mass + oil_mass_flux) / volumes
#         return (water**2).mean() + (oil**2).mean()

#     def _compute_liquid_mass(
#         self,
#         model: nn.Module,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         lower_values = model(t=self.tquads.lower_t, x=self.xquads.x)
#         upper_values = model(t=self.tquads.upper_t, x=self.xquads.x)

#         water_mass = self._integrate_density(
#             lower_values=lower_values,
#             upper_values=upper_values,
#             output='sw',
#         )
#         oil_mass = self._integrate_density(
#             lower_values=lower_values,
#             upper_values=upper_values,
#             output='so',
#         )
#         return water_mass, oil_mass

#     def _integrate_density(
#         self,
#         lower_values: dict[str, torch.Tensor],
#         upper_values: dict[str, torch.Tensor],
#         output: str,
#     ) -> torch.Tensor:
#         upper_saturation = upper_values[output]
#         lower_saturation = lower_values[output]

#         liquid_values = self.phi * (upper_saturation - lower_saturation)
#         liquid_values = liquid_values.view(
#             self.xquads.num_bounds,
#             len(self.xquads.weights),
#         )
#         liquid_weighted = liquid_values * self.xquads.weights.T
#         liquid_mass = liquid_weighted.sum(dim=1, keepdim=True)
#         liquid_mass *= self.xquads.normalize_scalars
#         return liquid_mass

#     def _compute_speed_flux_over_time(
#         self,
#         model: nn.Module,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         water_inner, oil_inner = self._compute_velocity_flux_through_surface(model=model)

#         water_weighted = water_inner * self.tquads.weights.T
#         oil_weighted = oil_inner * self.tquads.weights.T

#         water_flux = water_weighted.sum(dim=1, keepdim=True)
#         oil_flux = oil_weighted.sum(dim=1, keepdim=True)

#         water_flux *= self.tquads.normalize_scalars
#         oil_flux *= self.tquads.normalize_scalars
#         return water_flux, oil_flux

#     def _compute_velocity_flux_through_surface(
#         self,
#         model: nn.Module,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         lower_values = model(t=self.tquads.t, x=self.xquads.lower_x)
#         upper_values = model(t=self.tquads.t, x=self.xquads.upper_x)

#         water_inner = self._integrate_velocity(
#             lower_values=lower_values,
#             upper_values=upper_values,
#             output='uw',
#         )

#         oil_inner = self._integrate_velocity(
#             lower_values=lower_values,
#             upper_values=upper_values,
#             output='uo',
#         )
#         return water_inner, oil_inner

#     def _integrate_velocity(
#         self,
#         lower_values: dict[str, torch.Tensor],
#         upper_values: dict[str, torch.Tensor],
#         output: str,
#     ) -> torch.Tensor:
#         upper_velocity = upper_values[output]
#         lower_velocity = lower_values[output]
#         liquid_lower_values = -1.0 * lower_velocity * self.xquads.normals
#         liquid_upper_values = 1.0 * upper_velocity * self.xquads.normals
#         liquid = liquid_upper_values + liquid_lower_values
#         return liquid.view(self.tquads.num_bounds, len(self.tquads.weights))

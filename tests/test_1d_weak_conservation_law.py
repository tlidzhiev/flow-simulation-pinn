import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import quad

from pinns.data.datasets import SpatialQuadratureDataset, TimeQuadratureDataset
from pinns.data.quadratures import Spatial1DQuadrature, TimeQuadrature
from pinns.losses.one_dim import WeakConservationLoss


def test_weak_conservation_loss():
    phi = 0.1

    t0, t1 = sorted(np.random.rand(2))
    x0, x1 = sorted(np.random.rand(2))

    print(f't0, t1: {t0, t1}')
    print(f'x0, x1: {x0, x1}')

    t_boundary = np.array([[[t0, t1]]])
    x_boundary = np.array([[[x0, x1]]])

    t_quads = TimeQuadrature(quad_order=5, bounds=t_boundary)
    x_quads = Spatial1DQuadrature(quad_order=5, bounds=x_boundary)

    t_quads = t_quads.generate_planes(num_points=x_quads.x.shape[1])
    x_quads = x_quads.generate_planes(num_points=t_quads.t.shape[1])

    t_quads = TimeQuadratureDataset(t_quads).to(device='cpu')
    x_quads = SpatialQuadratureDataset(x_quads).to(device='cpu')

    def sw_np(t, x):
        return np.sin(np.pi * t) * x

    def so_np(t, x):
        return np.cos(np.pi * x) * t

    def uw_np(t, x):
        return t * x

    def uo_np(t, x):
        return t**2 + x**2

    water_mass_true, _ = quad(lambda x: phi * (sw_np(t1, x) - sw_np(t0, x)), x0, x1)
    oil_mass_true, _ = quad(lambda x: phi * (so_np(t1, x) - so_np(t0, x)), x0, x1)

    water_flux_true, _ = quad(lambda t: uw_np(t, x1) - uw_np(t, x0), t0, t1)
    oil_flux_true, _ = quad(lambda t: uo_np(t, x1) - uo_np(t, x0), t0, t1)

    loss_true = (water_mass_true + water_flux_true) ** 2 + (oil_mass_true + oil_flux_true) ** 2

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, t, x):
            sw = torch.sin(torch.pi * t) * x  # sw(t, x) = sin(πt) * x
            so = torch.cos(torch.pi * x) * t  # so(t, x) = cos(πx) * t
            uw = t * x  # uw(t, x) = t * x
            uo = t**2 + x**2  # uo(t, x) = t² + x²
            return {'sw': sw, 'so': so, 'uw': uw, 'uo': uo}

    model = SimpleModel()
    criterion = WeakConservationLoss(phi=phi, normalize=False)
    loss_pred = criterion(t_quads=t_quads, x_quads=x_quads, model=model).item()

    water_mass_pred = criterion.water_mass.item()
    oil_mass_pred = criterion.oil_mass.item()
    water_flux_pred = criterion.water_flux.item()
    oil_flux_pred = criterion.oil_flux.item()

    assert np.isclose(
        loss_true, loss_pred, atol=1.0e-6
    ), f'Loss mismatch: Computed loss {loss_pred} != Expected loss {loss_true}'
    assert np.isclose(
        water_mass_pred, water_mass_true, atol=1.0e-6
    ), f'Water mass mismatch: {water_mass_pred} != {water_mass_true}'
    assert np.isclose(
        oil_mass_pred, oil_mass_true, atol=1.0e-6
    ), f'Oil mass mismatch: {oil_mass_pred} != {oil_mass_true}'
    assert np.isclose(
        water_flux_pred, water_flux_true, atol=1.0e-6
    ), f'Water flux mismatch: {water_flux_pred} != {water_flux_true}'
    assert np.isclose(
        oil_flux_pred, oil_flux_true, atol=1.0e-6
    ), f'Oil flux mismatch: {oil_flux_pred} != {oil_flux_true}'

from typing import Tuple

import numpy as np

from .base import BaseTimeSpatialDomain


class Time1DSpatialDomain(BaseTimeSpatialDomain):
    def __init__(
        self,
        t_domain: Tuple[float, float],
        x_domain: Tuple[float, float],
    ):
        self.t_min, self.t_max = t_domain
        self.x_min, self.x_max = x_domain

    def generate_collocation_data(self, shape: Tuple[int, int]) -> np.ndarray:
        nt, nx = shape
        t = np.linspace(self.t_min, self.t_max, nt)
        x = np.linspace(self.x_min, self.x_max, nx)
        t_mesh, x_mesh = np.meshgrid(t, x, indexing='ij')
        collocation_data = np.stack([t_mesh.flatten(), x_mesh.flatten()], axis=1)
        return collocation_data

    def generate_initial_data(self, shape: Tuple[int, int]) -> np.ndarray:
        nt, nx = shape
        assert nt == nx
        x_initial = np.linspace(self.x_min, self.x_max, nx)
        initial_data = np.stack([np.full_like(x_initial, self.t_min), x_initial], axis=1)
        return initial_data

    def generate_boundary_data(self, shape: Tuple[int, int]) -> np.ndarray:
        nt, nx = shape
        assert nt == nx
        t_boundary = np.linspace(self.t_min, self.t_max, nt)
        left_boundary = np.stack([t_boundary, np.full_like(t_boundary, self.x_min)], axis=1)
        right_boundary = np.stack([t_boundary, np.full_like(t_boundary, self.x_max)], axis=1)
        boundary_data = np.vstack([left_boundary, right_boundary])
        return boundary_data

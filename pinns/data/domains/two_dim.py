from typing import Tuple

import numpy as np

from .base import BaseTimeSpatialDomain


class Time2DSpatialDomain(BaseTimeSpatialDomain):
    def __init__(
        self,
        t_domain: Tuple[float, float],
        x_domain: Tuple[float, float],
        y_domain: Tuple[float, float],
    ):
        self.t_min, self.t_max = t_domain
        self.x_min, self.x_max = x_domain
        self.y_min, self.y_max = y_domain

    def generate_collocation_data(self, shape: Tuple[int, int, int]) -> np.ndarray:
        nt, nx, ny = shape
        t = np.linspace(self.t_min, self.t_max, nt)
        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        t_mesh, x_mesh, y_mesh = np.meshgrid(t, x, y, indexing='ij')
        collocation_data = np.stack([t_mesh.flatten(), x_mesh.flatten(), y_mesh.flatten()], axis=1)
        return collocation_data

    def generate_initial_data(self, shape: Tuple[int, int, int]) -> np.ndarray:
        nt, nx, ny = shape
        assert nx == ny
        assert nt == nx * ny

        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
        initial_data = np.stack(
            [np.full_like(x_mesh.flatten(), self.t_min), x_mesh.flatten(), y_mesh.flatten()],
            axis=1,
        )
        return initial_data

    def generate_boundary_data(self, shape: Tuple[int, int, int]) -> np.ndarray:
        nt, nx, ny = shape
        assert nx == ny
        t_boundary = np.linspace(self.t_min, self.t_max, nt)
        x_boundary = np.linspace(self.x_min, self.x_max, nx)
        y_boundary = np.linspace(self.y_min, self.y_max, ny)

        # Create boundary meshes
        t_mesh_b, y_mesh_b = np.meshgrid(t_boundary, y_boundary, indexing='ij')
        t_mesh_b_x, x_mesh_b = np.meshgrid(t_boundary, x_boundary, indexing='ij')

        # Left and right boundaries (x = x_min, x_max)
        left_boundary = np.stack(
            [t_mesh_b.flatten(), np.full_like(t_mesh_b.flatten(), self.x_min), y_mesh_b.flatten()],
            axis=1,
        )
        right_boundary = np.stack(
            [t_mesh_b.flatten(), np.full_like(t_mesh_b.flatten(), self.x_max), y_mesh_b.flatten()],
            axis=1,
        )

        # Bottom and top boundaries (y = y_min, y_max)
        bottom_boundary = np.stack(
            [t_mesh_b_x.flatten(), x_mesh_b.flatten(), np.full_like(t_mesh_b_x.flatten(), self.y_min)],
            axis=1,
        )
        top_boundary = np.stack(
            [t_mesh_b_x.flatten(), x_mesh_b.flatten(), np.full_like(t_mesh_b_x.flatten(), self.y_max)],
            axis=1,
        )

        boundary_data = np.vstack([left_boundary, right_boundary, bottom_boundary, top_boundary])
        return boundary_data

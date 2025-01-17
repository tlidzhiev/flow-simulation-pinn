from typing import Dict

import numpy as np

from .base import BaseQuadrature


class Spatial1DQuadrature(BaseQuadrature):
    def __init__(self, quad_order: int, bounds: np.ndarray):
        super().__init__(dim=1, quad_order=quad_order, bounds=bounds)
        self.x = self.nodes

        self.x_lower = None
        self.x_upper = None

    def generate_planes(self, num_points: int):
        bounds = np.repeat(self.bounds, num_points, axis=0).reshape(self.num_bounds, num_points, 2)
        self.x_lower = bounds[:, :, 0:1]
        self.x_upper = bounds[:, :, -1:]
        return self

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            'dim': self.dim,
            'quad_order': self.quad_order,
            'x': self.x,
            'x_lower': self.x_lower,
            'x_upper': self.x_upper,
            'weights': self.weights,
            'normalize_scalars': self.normalize_scalars,
            'volumes': self.volumes,
        }

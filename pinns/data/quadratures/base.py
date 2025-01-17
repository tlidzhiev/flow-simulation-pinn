from typing import Tuple

import numpy as np
import scipy.special as sp


class BaseQuadrature:
    def __init__(self, dim: int, quad_order: int, bounds: np.ndarray):
        assert bounds.shape == (len(bounds), dim, 2)
        self.dim = dim
        self.quad_order = quad_order
        self.num_bounds = len(bounds)
        self.bounds = bounds

        data = self._generate_mapped_quadrature()
        self.nodes, self.weights, self.jacobians, self.normalize_scalars = data
        self.volumes = np.diff(bounds, axis=2)[:, :, 0]

        assert self.nodes.shape == (self.num_bounds, self.quad_order**self.dim, self.dim)
        assert self.weights.shape == (self.quad_order**self.dim, 1)
        assert self.jacobians.shape == (self.num_bounds, 1)
        assert self.normalize_scalars.shape == (self.num_bounds, 1)
        assert self.volumes.shape == (self.num_bounds, 1)

    def _generate_mapped_quadrature(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nodes, weights, weight_sum = self._generate_legendre_quadrature(self.dim, self.quad_order)
        nodes, jacobians = self._mapping(nodes, self.bounds)
        normalize_scalars = (weight_sum**self.dim) * jacobians
        return nodes, weights, jacobians, normalize_scalars

    @staticmethod
    def _generate_legendre_quadrature(dim: int, quad_order: int) -> Tuple[np.ndarray, np.ndarray, float]:
        nodes, weights, weight_sum = sp.roots_legendre(quad_order, mu=True)
        weights /= weight_sum

        nodes = np.meshgrid(*([nodes] * dim), indexing='ij')
        nodes = np.stack(nodes, axis=-1).reshape(-1, dim)
        weights = np.meshgrid(*([weights] * dim), indexing='ij')
        weights = np.prod(np.stack(weights, axis=-1), axis=-1).reshape(-1, 1)
        return nodes, weights, weight_sum

    @staticmethod
    def _mapping(x: np.ndarray, bounds: np.ndarray) -> Tuple[np.ndarray, float]:
        scaling = np.diff(bounds, axis=2)[:, :, 0] * 0.5
        shift = np.sum(bounds, axis=2) * 0.5

        scaling = scaling[:, np.newaxis, :]
        shift = shift[:, np.newaxis, :]
        x = x[np.newaxis, :, :]
        mapped_x = x * scaling + shift
        jacobians = np.absolute(np.prod(scaling, axis=1))
        return mapped_x, jacobians

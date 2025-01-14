from typing import Tuple

import numpy as np


class BaseTimeSpatialDomain:
    def generate_collocation_data(self, shape: Tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError('Subclass must implement generate_collocation_data method')

    def generate_initial_data(self, shape: Tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError('Subclass must implement generate_initial_data method')

    def generate_boundary_data(self, shape: Tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError('Subclass must implement generate_boundary_data method')

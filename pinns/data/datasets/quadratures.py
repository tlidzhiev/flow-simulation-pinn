import pinns.utils
from pinns.data.quadratures import Spatial1DQuadrature, TimeQuadrature


class TimeQuadratureDataset:
    def __init__(self, t_quads: TimeQuadrature):
        assert t_quads.t_lower is not None and t_quads.t_upper is not None
        data = t_quads.to_dict()
        self.t = data['t']
        self.num_bounds, self.num_nodes, _ = self.t.shape
        self.t_lower = data['t_lower']
        self.t_upper = data['t_upper']
        self.weights = data['weights']
        self.volumes = data['volumes']
        self.normalize_scalars = data['normalize_scalars']

    def to(self, device: str = 'cpu'):
        tensors = [self.t, self.t_lower, self.t_upper]
        tensors = pinns.utils.to_tensors(tensors, requires_grad=True, device=device)
        self.t, self.t_lower, self.t_upper = tensors

        tensors = [self.weights, self.volumes, self.normalize_scalars]
        tensors = pinns.utils.to_tensors(tensors, requires_grad=False, device=device)
        self.weights, self.volumes, self.normalize_scalars = tensors
        return self


class SpatialQuadratureDataset:
    def __init__(self, x_quads: Spatial1DQuadrature):
        assert x_quads.x_lower is not None and x_quads.x_upper is not None
        data = x_quads.to_dict()
        self.x = data['x']
        self.num_bounds, self.num_nodes, _ = self.x.shape
        self.x_lower = data['x_lower']
        self.x_upper = data['x_upper']
        self.weights = data['weights']
        self.volumes = data['volumes']
        self.normalize_scalars = data['normalize_scalars']

    def to(self, device: str = 'cpu'):
        tensors = [self.x, self.x_lower, self.x_upper]
        tensors = pinns.utils.to_tensors(tensors, requires_grad=True, device=device)
        self.x, self.x_lower, self.x_upper = tensors

        tensors = [self.weights, self.volumes, self.normalize_scalars]
        tensors = pinns.utils.to_tensors(tensors, requires_grad=False, device=device)
        self.weights, self.volumes, self.normalize_scalars = tensors
        return self

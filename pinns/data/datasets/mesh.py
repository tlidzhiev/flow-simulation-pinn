from typing import Dict, Optional, Union

import numpy as np
import torch

import pinns.utils


class BaseMeshDataset:
    def __init__(
        self,
        t: Union[np.ndarray, torch.Tensor],
        x: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        z: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        self.t = t
        self.x = x
        self.y = y
        self.z = z

    def to(self, device: str = 'cpu'):
        tensors = [self.t, self.x] + ([self.y] if self.y is not None else []) + ([self.z] if self.z is not None else [])
        tensors = pinns.utils.to_tensors(tensors, requires_grad=True, device=device)

        self.t, self.x = tensors[:2]
        if self.y is not None:
            self.y = tensors[2]
        if self.z is not None:
            self.z = tensors[3] if self.y is not None else tensors[2]
        return self

    def to_dict(self, is_numpy: bool = False) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        data = {'t': self.t, 'x': self.x}
        if self.y is not None:
            data['y'] = self.y
        if self.z is not None:
            data['z'] = self.z

        if is_numpy:
            for key in data.keys():
                data[key] = pinns.utils.to_numpy([data[key]])[0]
        return data


class BaseMeshTargetDataset(BaseMeshDataset):
    def __init__(
        self,
        t: Union[np.ndarray, torch.Tensor],
        x: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        z: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target: Dict[str, Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__(t=t, x=x, y=y, z=z)
        self.target = target
        self.output_names = list(target.keys())

    def to(self, device: str = 'cpu'):
        super().to(device)

        batch = pinns.utils.to_tensors(
            [self.target[name] for name in self.output_names],
            requires_grad=False,
            device=device,
        )
        self.target = {name: batch[i] for i, name in enumerate(self.output_names)}
        return self

    def to_dict(self, is_numpy: bool = False) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        data = super().to_dict(is_numpy)
        target = self.target
        if is_numpy:
            target = {key: pinns.utils.to_numpy([value])[0] for key, value in self.target.items()}
        data['target'] = target
        return data


class InitialConditionDataset(BaseMeshTargetDataset):
    def __init__(
        self,
        t: Union[np.ndarray, torch.Tensor],
        x: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        z: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target: Dict[str, Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__(t=t, x=x, y=y, z=z, target=target)


class BoundaryConditionDataset(BaseMeshTargetDataset):
    def __init__(
        self,
        t: Union[np.ndarray, torch.Tensor],
        x: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        z: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target: Dict[str, Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__(t=t, x=x, y=y, z=z, target=target)


class SimulationDataset(BaseMeshTargetDataset):
    def __init__(
        self,
        t: Union[np.ndarray, torch.Tensor],
        x: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        z: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target: Dict[str, Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__(t=t, x=x, y=y, z=z, target=target)


class CollocationDataset(BaseMeshDataset):
    def __init__(
        self,
        t: Union[np.ndarray, torch.Tensor],
        x: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        z: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__(t=t, x=x, y=y, z=z)

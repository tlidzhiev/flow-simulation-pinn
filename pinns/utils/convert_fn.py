from typing import List, Union

import numpy as np
import torch


def to_tensors(
    batch: List[Union[np.ndarray, List]],
    requires_grad: bool = False,
    device: str = 'cpu',
) -> List[torch.Tensor]:
    return [
        torch.tensor(
            x,
            requires_grad=requires_grad,
            dtype=torch.float32,
            device=device,
        )
        if x is not None
        else None
        for x in batch
    ]


def to_numpy(batch: List[torch.Tensor]) -> List[np.ndarray]:
    return [x.detach().cpu().numpy() for x in batch]

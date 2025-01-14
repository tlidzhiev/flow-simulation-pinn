from typing import List, Union

import torch


def gradient(
    outputs: torch.Tensor,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
    create_graph: bool = False,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> List[torch.Tensor]:
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    gradients = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )
    gradients = [grad if grad is not None else torch.zeros_like(inputs[i]) for i, grad in enumerate(gradients)]
    return gradients

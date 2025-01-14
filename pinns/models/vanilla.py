from typing import Dict, List, Union

import torch
import torch.nn as nn


class VanillaPINN(nn.Module):
    def __init__(
        self,
        layers: List[int],
        activation: nn.Module,
        output_names: List[str],
    ):
        super().__init__()
        self.output_names = output_names
        self.net = self.init_net(layers, activation)

    @staticmethod
    def init_net(layers: List[int], activation: nn.Module) -> nn.Sequential:
        net = nn.Sequential()

        net.add_module('input_layer', nn.Linear(layers[0], layers[1]))
        net.add_module('activation_1', activation)

        for i in range(1, len(layers) - 2):
            hidden_layer = nn.Linear(layers[i], layers[i + 1], bias=True)
            nn.init.xavier_uniform_(hidden_layer.weight)
            net.add_module(f'hidden_layer_{i + 1}', hidden_layer)
            net.add_module(f'activation_{i + 1}', activation)

        net.add_module('output_layer', nn.Linear(layers[-2], layers[-1]))
        return net

    def forward(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            z = torch.column_stack([t, x])
        elif isinstance(x, list):
            if len(x) == 1:
                z = torch.column_stack([t, x[0]])
            elif len(x) == 2:
                z = torch.column_stack([t, x[0], x[1]])
            elif len(x) == 3:
                z = torch.column_stack([t, x[0], x[1], x[2]])
            else:
                raise ValueError('x must be a Tensor or a list of up to 3 Tensors')
        else:
            raise TypeError('x must be a Tensor or a list of Tensors')

        z: torch.Tensor = self.net(z)
        return {output: z[:, i].view(-1, 1) for i, output in enumerate(self.output_names)}

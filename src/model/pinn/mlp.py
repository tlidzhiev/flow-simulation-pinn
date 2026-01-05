from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from src.utils.jax import get_activation, get_weight_initializer


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        output_names: list[str],
        activation: Literal['tanh', 'silu', 'gelu'],
        lb: tuple[float, ...] | list[float],
        ub: tuple[float, ...] | list[float],
        rngs: int,
        init_mode: Literal['normal', 'uniform'],
    ) -> None:
        if output_dim != len(output_names):
            raise ValueError('Output dimension must match the number of output names.')

        _rngs = nnx.Rngs(rngs)
        kernel_init = get_weight_initializer(activation, init_mode)
        self.input_proj = nnx.Sequential(
            nnx.Linear(input_dim, hidden_dim, kernel_init=kernel_init, rngs=_rngs),
            get_activation(activation),
        )
        hiddens = []
        for i in range(num_blocks):
            hiddens.extend(
                [
                    nnx.Linear(hidden_dim, hidden_dim, kernel_init=kernel_init, rngs=_rngs),
                    get_activation(activation),
                ]
            )
        self.hidden_layers = nnx.Sequential(*hiddens)
        self.output_proj = nnx.Linear(hidden_dim, output_dim, kernel_init=kernel_init, rngs=_rngs)
        self.output_names = output_names
        self.lb = jnp.array(lb)
        self.ub = jnp.array(ub)

    def __call__(self, t: jax.Array, x: jax.Array | list[jax.Array]) -> dict[str, jax.Array]:
        if isinstance(x, list):
            x = jnp.concatenate(x, axis=1)
        z = jnp.concatenate([t, x], axis=1)
        z = 2.0 * (z - self.lb) / (self.ub - self.lb) - 1.0

        h = self.input_proj(z)
        h = self.hidden_layers(h)
        h = self.output_proj(h)
        return {name: h[:, i].reshape(-1, 1) for i, name in enumerate(self.output_names)}

from typing import Callable, Literal

import flax.nnx as nnx
import jax
import jax.nn.initializers as initializers
import optax
from hydra.utils import instantiate
from omegaconf import DictConfig


def _parse_act(activation: str) -> tuple[str, float | None]:
    """
    Parse activation string to extract name and optional parameter.

    Parameters
    ----------
    activation : str
        Activation string, optionally with parameter (e.g., "leaky_relu:0.2").

    Returns
    -------
    act_name : str
        Activation name in lowercase.
    param : float or None
        Activation parameter if provided, None otherwise.

    Raises
    ------
    ValueError
        If parameter string cannot be converted to float.
    """
    if ':' in activation:
        act_name, param_str = activation.split(':', 1)
        try:
            param = float(param_str)
        except ValueError:
            raise ValueError(
                f'Invalid activation parameter "{param_str}" in "{activation}". '
                f'Parameter must be a number.'
            )
        return act_name.lower(), param
    return activation.lower(), None


def get_activation(activation: str) -> Callable[[jax.Array], jax.Array]:
    """
    Get activation function by name for flax.nnx.

    Parameters
    ----------
    activation : str
        Activation name, optionally with parameter (e.g., "leaky_relu:0.2").
        Supported: "tanh", "silu", "gelu".

    Returns
    -------
    Callable[[jax.Array], jax.Array]
        JAX activation function compatible with flax.nnx.

    Raises
    ------
    ValueError
        If activation type is not supported.
    """
    act_name, param = _parse_act(activation)
    match act_name:
        case 'tanh':
            return nnx.tanh
        case 'silu':
            return nnx.silu
        case 'gelu':
            return nnx.gelu
        case _:
            raise ValueError(
                f'Unknown activation type: "{act_name}". Supported types: "tanh", "silu", "gelu".'
            )


def get_weight_initializer(
    activation: str,
    mode: Literal['normal', 'uniform'],
) -> nnx.Initializer:
    """
    Returns weight initializer for neural network module using Kaiming/Xavier initialization.

    Parameters
    ----------
    activation : str
        Activation function name.
        Supported types: "tanh", "silu", "gelu".
    mode : {'normal', 'uniform'}
        Initialization mode.

    Returns
    -------
    nnx.Initializer
        A JAX initializer compatible with Flax NNX.

    Raises
    ------
    ValueError
        If the initialization mode or activation type is not supported.
    """
    if mode not in ['normal', 'uniform']:
        raise ValueError(
            f'Unknown initialization mode: "{mode}". Supported modes: "normal", "uniform".'
        )

    act_name, param = _parse_act(activation)
    param = param if param is not None else 0.0

    if act_name not in ['tanh', 'silu', 'gelu']:
        raise ValueError(
            f"Unknown activation type for initialization: '{act_name}'. "
            f'Supported types: "tanh", "silu", "gelu".'
        )

    if act_name == 'tanh':
        return initializers.glorot_normal() if mode == 'normal' else initializers.glorot_uniform()

    return initializers.he_normal() if mode == 'normal' else initializers.he_uniform()


def get_optimizer(cfg: DictConfig, model: nnx.Module) -> nnx.Optimizer:
    transforms = []
    if cfg.trainer.get('max_grad_norm') is not None:
        transforms.append(optax.clip_by_global_norm(cfg.trainer.max_grad_norm))

    lr_scheduler = instantiate(cfg.lr_scheduler)
    optimizer = instantiate(cfg.optimizer, learning_rate=lr_scheduler)
    transforms.append(optimizer)

    return nnx.Optimizer(model, *transforms, wrt=nnx.Param)

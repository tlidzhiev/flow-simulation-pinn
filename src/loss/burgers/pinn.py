from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ..base import BaseLoss


class PINNLoss(BaseLoss):
    def __init__(
        self,
        lambda_ic: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_pde: float = 1.0,
    ) -> None:
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.lambda_pde = lambda_pde

        self.loss_names = ['loss', 'loss_pde', 'loss_ic', 'loss_bc']
        self.loss_weight_names = ['lambda_pde', 'lambda_ic', 'lambda_bc']

    def pde_loss(self, model: nnx.Module, batch: dict[str, Any]) -> jax.Array:
        t = jnp.asarray(batch['t'])
        x = jnp.asarray(batch['x'])

        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)

        def u_batched(t_in: jax.Array, x_in: jax.Array) -> jax.Array:
            u = model(t=t_in, x=x_in)['u']
            return u.reshape(-1)

        ones_t = jnp.ones_like(t)
        ones_x = jnp.ones_like(x)

        _, u_t = jax.jvp(lambda t_: u_batched(t_, x), (t,), (ones_t,))
        _, u_x = jax.jvp(lambda x_: u_batched(t, x_), (x,), (ones_x,))
        u = u_batched(t, x)
        residual = u_t + u * u_x
        return jnp.mean(jnp.square(residual))

    def ic_loss(self, model: nnx.Module, ic_data: dict[str, jax.Array]) -> jax.Array:
        u_pred = model(t=ic_data['t'], x=ic_data['x'])['u']
        return jnp.mean((u_pred - ic_data['u']) ** 2)

    def bc_loss(self, model: nnx.Module, bc_data: list[dict[str, jax.Array]]) -> jax.Array:
        t_all = jnp.concatenate([bc['t'] for bc in bc_data], axis=0)
        x_all = jnp.concatenate([bc['x'] for bc in bc_data], axis=0)
        u_true_all = jnp.concatenate([bc['u'] for bc in bc_data], axis=0)

        u_pred_all = model(t=t_all, x=x_all)['u']
        return jnp.mean((u_pred_all - u_true_all) ** 2)

    def __call__(
        self,
        model: nnx.Module,
        batch: dict[str, Any],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        loss_pde = self.pde_loss(model, batch['pde'])
        loss_ic = self.ic_loss(model, batch['ic'])
        loss_bc = self.bc_loss(model, batch['bc'])
        loss = self.lambda_pde * loss_pde + self.lambda_ic * loss_ic + self.lambda_bc * loss_bc

        loss_dict = {
            'loss': loss,
            'loss_pde': loss_pde,
            'loss_ic': loss_ic,
            'loss_bc': loss_bc,
        }
        return loss, loss_dict

    def __str__(self) -> str:
        return f'{type(self).__name__}(lambda_pde={self.lambda_pde}, lambda_ic={self.lambda_ic}, lambda_bc={self.lambda_bc})'

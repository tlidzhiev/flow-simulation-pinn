from typing import Literal

import jax
import jax.numpy as jnp

from ..pinn_base import PINNBaseDataset


class BurgersEqDataset(PINNBaseDataset):
    def __init__(
        self,
        ic_size: int = 100,
        bc_size: int = 100,
        ic_sample_type: Literal['grid', 'uniform'] = 'grid',
        bc_sample_type: Literal['grid', 'uniform'] = 'grid',
        collocation_size: tuple[int, int] | int = (100, 100),
        collocation_sample_type: Literal['grid', 'uniform'] = 'grid',
        seed: int | None = None,
        resample_step: int | None = None,
    ) -> None:
        super().__init__(
            t_domain=(0.0, 1.0),
            x_domain=(0.0, 2.0),
            pde_params={},
            collocation_size=collocation_size,
            collocation_sample_type=collocation_sample_type,
            seed=seed,
            resample_step=resample_step,
        )

        self.ic_size = ic_size
        self.bc_size = bc_size
        self.ic_sample_type = ic_sample_type
        self.bc_sample_type = bc_sample_type

        self.ic = self._sample_initial_condition()
        self.bc = self._sample_boundary_condition()

    def __getitem__(self, record_key):
        batch = super().__getitem__(record_key)

        if (
            self.ic_sample_type != 'grid'
            and self._resample_step is not None
            and self._current_step % self._resample_step == 0
        ):
            print(f'Resampling IC data at step {self._current_step}')
            self.ic = self._sample_initial_condition()
        if (
            self.bc_sample_type != 'grid'
            and self._resample_step is not None
            and self._current_step % self._resample_step == 0
        ):
            print(f'Resampling BC data at step {self._current_step}')
            self.bc = self._sample_boundary_condition()

        batch.update({'ic': self.ic, 'bc': self.bc})
        return batch

    def _sample_initial_condition(self) -> dict[str, jax.Array]:
        t_0 = self.t_domain[0]

        if self.ic_sample_type == 'grid':
            x = jnp.linspace(self.x_domain[0], self.x_domain[1], self.ic_size)
            x = x.reshape(-1, 1)
            t = jnp.full_like(x, t_0)

        elif self.ic_sample_type == 'uniform':
            assert self.rng is not None, (
                'When ic_sample_type is "uniform", the rng must be initialized'
            )
            self.rng, rng_x = jax.random.split(self.rng)
            x = jax.random.uniform(
                rng_x,
                shape=(self.ic_size, 1),
                minval=self.x_domain[0],
                maxval=self.x_domain[1],
            )
            t = jnp.full_like(x, t_0)

        else:
            raise ValueError(f'Unknown ic_sample_type: {self.ic_sample_type}')

        u = -1.0 * jnp.sin(jnp.pi * (x - 1.0))
        return {'t': t, 'x': x, 'u': u}

    def _sample_boundary_condition(self) -> list[dict[str, jax.Array]]:
        x_0 = self.x_domain[0]
        x_1 = self.x_domain[1]

        if self.bc_sample_type == 'grid':
            t = jnp.linspace(self.t_domain[0], self.t_domain[1], self.bc_size)
            t = t.reshape(-1, 1)
            bc_left = {
                't': t,
                'x': jnp.full_like(t, x_0),
            }
            bc_right = {
                't': t,
                'x': jnp.full_like(t, x_1),
            }

        elif self.bc_sample_type == 'uniform':
            assert self.rng is not None, (
                'When bc_sample_type is "uniform", the rng must be initialized'
            )
            self.rng, rng_t = jax.random.split(self.rng)

            t = jax.random.uniform(
                rng_t,
                shape=(self.bc_size, 1),
                minval=self.t_domain[0],
                maxval=self.t_domain[1],
            )
            bc_left = {
                't': t,
                'x': jnp.full_like(t, x_0),
            }
            bc_right = {
                't': t,
                'x': jnp.full_like(t, x_1),
            }

        else:
            raise ValueError(f'Unknown bc_sample_type: {self.bc_sample_type}')

        bc_left['u'] = jnp.zeros_like(bc_left['t'])
        bc_right['u'] = jnp.zeros_like(bc_right['t'])
        return [bc_left, bc_right]

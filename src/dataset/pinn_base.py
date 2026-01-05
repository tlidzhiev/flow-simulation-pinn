from typing import Literal

import grain
import jax
import jax.numpy as jnp


class PINNBaseDataset(grain.sources.RandomAccessDataSource):
    def __init__(
        self,
        t_domain: tuple[float, float],
        x_domain: tuple[float, float],
        pde_params: dict[str, float],
        collocation_size: tuple[int, int] | int,
        collocation_sample_type: Literal['grid', 'uniform'] = 'grid',
        seed: int | None = None,
        resample_step: int | None = None,
    ) -> None:
        if collocation_sample_type != 'grid' and seed is None:
            raise ValueError('seed must be provided when collocation_sample_type is not "grid".')

        if collocation_sample_type != 'grid' and resample_step is None:
            raise ValueError(
                'resample_step must be provided when collocation_sample_type is not "grid".'
            )

        self.t_domain = t_domain
        self.x_domain = x_domain
        self.pde_params = pde_params

        self.collocation_sample_type = collocation_sample_type
        self.collocation_size = (
            (collocation_size, collocation_size)
            if isinstance(collocation_size, int)
            else tuple(collocation_size)
        )

        self.rng = jax.random.PRNGKey(seed) if seed is not None else None
        self._current_step = 1
        self._resample_step = resample_step

        self.t, self.x = self._sample_collocation()

    def __getitem__(self, record_key):
        self._current_step += 1
        if (
            self.collocation_sample_type != 'grid'
            and self._resample_step is not None
            and self._current_step % self._resample_step == 0
        ):
            print(f'Resampling collocation points at step {self._current_step}')
            self.t, self.x = self._sample_collocation()

        return {'pde': {'t': self.t, 'x': self.x, 'pde_params': self.pde_params}}

    def _sample_collocation(self) -> tuple[jax.Array, jax.Array]:
        if self.collocation_sample_type == 'grid':
            assert isinstance(self.collocation_size, tuple), (
                'collocation_size must be a tuple for grid sampling.'
            )
            nt, nx = self.collocation_size
            t, x = self._sample_grid(nt=nt, nx=nx)

        elif self.collocation_sample_type == 'uniform':
            assert isinstance(self.collocation_size, int), (
                'collocation_size must be an int for uniform sampling.'
            )
            t, x = self._sample_uniform(n=self.collocation_size)
        else:
            raise ValueError(f'Unknown collocation_sample_type: {self.collocation_sample_type}')
        return t, x

    def _sample_grid(self, nt: int, nx: int) -> tuple[jax.Array, jax.Array]:
        t = jnp.linspace(self.t_domain[0], self.t_domain[1], nt)
        x = jnp.linspace(self.x_domain[0], self.x_domain[1], nx)

        t, x = jnp.meshgrid(t, x, indexing='ij')
        t, x = t.reshape(-1, 1), x.reshape(-1, 1)
        return t, x

    def _sample_uniform(self, n: int) -> tuple[jax.Array, jax.Array]:
        assert self.collocation_sample_type == 'uniform' and self.rng is not None, (
            'When collocation_sample_type is "uniform", the rng must be initialized.'
        )
        self.rng, rng_t, rng_x = jax.random.split(self.rng, 3)

        t = jax.random.uniform(
            rng_t, shape=(n, 1), minval=self.t_domain[0], maxval=self.t_domain[1]
        )
        x = jax.random.uniform(
            rng_x, shape=(n, 1), minval=self.x_domain[0], maxval=self.x_domain[1]
        )
        return t, x

    def __len__(self) -> int:
        return 1

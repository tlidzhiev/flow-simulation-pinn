from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import numpy as np
from flow_sim import compute_solution_vec
from omegaconf import DictConfig, ListConfig, OmegaConf
from plots import test_data_plots, train_data_plots

from pinns.data.domains import Time1DSpatialDomain


def sample_initial_data(
    domain: Time1DSpatialDomain,
    shape: Tuple[int, int],
    output_names: List[str],
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    mesh = domain.generate_initial_data(shape)
    t, x = mesh[:, 0].reshape(-1, 1), mesh[:, 1].reshape(-1, 1)
    p = 1 - x.copy()
    sw = np.zeros_like(x, dtype=np.float32)
    so = np.ones_like(x, dtype=np.float32)
    target = {name: arr.reshape(-1, 1) for name, arr in zip(output_names, [p, sw, so])}
    return {'t': t, 'x': x, 'target': target}


def sample_boundary_data(
    domain: Time1DSpatialDomain,
    shape: Tuple[int, int],
    output_names: List[str],
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    def sample(
        shape: int,
        p_value: float,
        sw_value: float,
        so_value: float,
    ):
        p = np.full(shape, p_value, dtype=np.float32).reshape(-1, 1)
        sw = np.full(shape, sw_value, dtype=np.float32).reshape(-1, 1)
        so = np.full(shape, so_value, dtype=np.float32).reshape(-1, 1)
        return p, sw, so

    p_values = [1.0, 0.0]
    sw_values = [1.0, 0.0]
    so_values = [0.0, 1.0]
    for i in range(len(p_values)):
        p, sw, so = sample(
            shape=shape[0],
            p_value=p_values[i],
            sw_value=sw_values[i],
            so_value=so_values[i],
        )
        p_values[i] = p
        sw_values[i] = sw
        so_values[i] = so

    mesh = domain.generate_boundary_data(shape)
    t, x = mesh[:, 0].reshape(-1, 1), mesh[:, 1].reshape(-1, 1)
    p = np.vstack(p_values)
    sw = np.vstack(sw_values)
    so = np.vstack(so_values)
    target = {name: arr.reshape(-1, 1) for name, arr in zip(output_names, [p, sw, so])}
    return {'t': t, 'x': x, 'target': target}


def sample_simulation_data(
    domain: Time1DSpatialDomain,
    shape: tuple[int, int],
    output_names: list[str],
):
    mesh = domain.generate_collocation_data(shape)
    t, x = mesh[:, 0].reshape(-1, 1), mesh[:, 1].reshape(-1, 1)

    p, dp_dx, sw, so, uw, uo = compute_solution_vec(t=t.flatten(), x=x.flatten(), nx=shape[0])
    target = {name: arr.reshape(-1, 1) for name, arr in zip(output_names, [p, sw, so, uw, uo])}
    return {'t': t, 'x': x, 'target': target}


def sample_collocation_data(
    domain: Time1DSpatialDomain,
    shape: tuple[int, int],
):
    mesh = domain.generate_collocation_data(shape)
    t, x = mesh[:, 0].reshape(-1, 1), mesh[:, 1].reshape(-1, 1)
    return {'t': t, 'x': x}


def sample_train_data(cfg: ListConfig) -> Dict[str, Any]:
    domain = Time1DSpatialDomain(t_domain=cfg.t_domain, x_domain=cfg.x_domain)

    data = {}
    data['initial_data'] = sample_initial_data(
        domain=domain,
        shape=cfg.initial_data.shape,
        output_names=cfg.initial_data.output_names,
    )

    data['boundary_data'] = sample_boundary_data(
        domain=domain,
        shape=cfg.boundary_data.shape,
        output_names=cfg.boundary_data.output_names,
    )
    data['collocation_data'] = sample_collocation_data(
        domain=domain,
        shape=cfg.collocation_data.shape,
    )
    data['simulation_data'] = sample_simulation_data(
        domain=domain,
        shape=cfg.simulation_data.shape,
        output_names=cfg.simulation_data.output_names,
    )

    return data


def sample_test_data(cfg: ListConfig) -> Dict[str, Any]:
    x = np.linspace(cfg.x_domain[0], cfg.x_domain[1], cfg.shape[1])
    t = np.full_like(x, cfg.t_point)
    p, dp_dx, sw, so, uw, uo = compute_solution_vec(t=t, x=x, nx=cfg.shape[1])
    target = {name: arr.reshape(-1, 1) for name, arr in zip(cfg.output_names, [p, dp_dx, sw, so, uw, uo])}
    return {'simulation_data': {'t': t.reshape(-1, 1), 'x': x.reshape(-1, 1), 'target': target}}


@hydra.main(version_base='1.3', config_path='../configs', config_name='config.yaml')
def main(cfg: DictConfig):
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    save_dir = Path(__file__).absolute().resolve().parent
    print(f'Save directory: {save_dir}')
    train_data = sample_train_data(cfg.train)
    np.save(save_dir / 'train/data.npy', [train_data])
    train_data_plots(data=train_data, title='Train Data', savepath=save_dir / 'train/data.png')
    train_data_plots(data=train_data, title='Train Data', savepath=save_dir / 'train/data.pdf')

    test_data = sample_test_data(cfg.test)
    np.save(save_dir / 'test/data.npy', [test_data])
    test_data_plots(data=test_data, title='Test Data', savepath=save_dir / 'test/data.png')
    test_data_plots(data=test_data, title='Test Data', savepath=save_dir / 'test/data.pdf')


if __name__ == '__main__':
    main()

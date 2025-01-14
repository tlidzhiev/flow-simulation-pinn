from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from pinns.utils.io_utils import get_root


def train_data_plots(
    data: Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    title: str,
    savepath: Path,
    is_show: bool = False,
):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
    axes = axes.flatten()

    mapping = {
        'initial_data': 'Initial Condition Nodes',
        'boundary_data': 'Boundary Condition Nodes',
        'collocation_data': 'Collocation Nodes',
        'simulation_data': 'Simulation Nodes',
    }

    for i, (field, title_item) in enumerate(mapping.items()):
        axes[i].scatter(data[field]['t'], data[field]['x'])
        axes[i].set_xlabel('Time (T)')
        axes[i].set_ylabel('Space (X)')
        axes[i].set_title(f'{title_item} in T x X space')
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        if field == 'simulation_data':
            axes[i].axvline(x=0.1, color='red', linestyle='--', label='t = 0.1')
            axes[i].legend()
    fig.suptitle(f'{title}', fontsize=20)

    path = get_root() / savepath
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    if is_show:
        plt.show()


def test_data_plots(
    data: Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
    title: str,
    savepath: Path,
    is_show: bool = False,
):
    mapping = {
        'p': r'$p(t, x)$',
        'dp_dx': r'$\nabla \, p(t, x)$',
        'sw': r'$s_w(t, x)$',
        'so': r'$s_o(t, x)$',
        'uw': r'$u_w(t, x)$',
        'uo': r'$u_o(t, x)$',
    }
    data = data['simulation_data']
    output_names = list(data['target'].keys())
    fig, axes = plt.subplots(nrows=1, ncols=len(output_names), figsize=(20, 5))
    axes = axes.flatten()

    for i, name in enumerate(output_names):
        axes[i].plot(data['x'], data['target'][name])
        axes[i].set_title(mapping.get(name, name), fontsize=15)
        axes[i].set_xlabel(r'$x$')
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.suptitle(title, fontsize=25)
    plt.tight_layout()
    path = get_root() / savepath
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    if is_show:
        plt.show()

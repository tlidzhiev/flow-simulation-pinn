from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import wandb
from matplotlib.figure import Figure


class Logger:
    def __init__(
        self,
        project_config: Dict[str, Any],
        project_name: str,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        wandb.login()

        self.run_id = run_id
        self.run_name = run_name

        wandb.init(
            project=project_name,
            entity=entity,
            config=project_config,
            name=run_name,
            id=run_id,
        )
        self.wandb = wandb
        self.run_id = self.wandb.run.id
        self.run_name = self.wandb.run.name
        self.metrics_history = defaultdict(list)

    def add_checkpoint(self, checkpoint_path: str, save_dir: str):
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def add_scalar(self, scalar_name: str, scalar: float):
        self.metrics_history[scalar_name].append(scalar)
        self.wandb.log({scalar_name: scalar})

    def add_scalars(self, scalars: Dict[str, float]):
        for scalar_name, scalar in scalars.items():
            self.metrics_history[scalar_name].append(scalar)

        self.wandb.log({scalar_name: scalar for scalar_name, scalar in scalars.items()})

    def add_image(self, image_name: str, image: Union[Path, np.ndarray, Figure]):
        self.wandb.log({image_name: self.wandb.Image(image)})

    def finish(self):
        self.wandb.finish()

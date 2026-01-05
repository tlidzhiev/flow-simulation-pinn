import grain
from hydra.utils import instantiate
from omegaconf import DictConfig


def get_dataloaders(cfg: DictConfig) -> grain.DataLoader:
    dataset = instantiate(cfg.dataset)
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=grain.samplers.IndexSampler(num_records=1, num_epochs=cfg.trainer.num_steps),
        worker_count=0,
    )
    return dataloader

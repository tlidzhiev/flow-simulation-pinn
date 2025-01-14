import random

import numpy as np
import torch


def set_random_seed(seed: int):
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

"""Central configuration for modeling reproducibility and hyperparameters.

Defines default random seeds, train/test split sizes, and provides a utility
to set random seeds for Python, NumPy, and PyTorch.
"""

import random
import numpy as np
import torch

# Central configuration values
DEFAULT_SEED = 42
DEFAULT_TEST_TRIALS = 10  # 40/10 split for a 50-trial dataset


def set_seed(seed: int):
    """Seed all random number generators to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Configure deterministic behaviors if possible, but benchmark can remain True
    # since Neural ODEs rely on adaptive steps which can be slightly different
    # on different GPU architectures, but seeding provides consistency on the same hardware.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

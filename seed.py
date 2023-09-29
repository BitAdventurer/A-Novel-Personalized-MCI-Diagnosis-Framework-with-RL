import os
import numpy as np
import torch
import random

def seed_everything(seed=42):  # Changed default seed to 42, often used as default in ML
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set to False for more deterministic behavior

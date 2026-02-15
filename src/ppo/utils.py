import numpy as np
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def explained_variance(y_pred, y_true):
    var_y = torch.var(y_true)
    if var_y < 1e-12:
        return torch.tensor(float("nan"), device=y_true.device)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-12)

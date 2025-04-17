import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def num2onehot(a):
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    b = torch.zeros((a.shape[0], int(a.max()) + 1))
    b[np.arange(a.shape[0]), a.int()] = 1
    return b

def onehot2num(a):
    return torch.argmax(a, dim=1)

def check_for_gpu(device):
    """Check if a CUDA device is available.

    Args:
        device (torch.device): current set device.
    """
    if device.type == "cuda":
        if torch.cuda.is_available():
            print("GPU is available, running on GPU.\n")
        else:
            print("GPU is NOT available, running instead on CPU.\n")
    else:
        if torch.cuda.is_available():
            print("Warning: You have a CUDA device, so you may consider using GPU for potential acceleration\n by setting device to 'cuda'.\n")
        else:
            print("Running on CPU.\n")

def make_dataloader(x, z, y=None, batch_size=128, shuffle=True, num_workers=0):
    """Make dataloader.

    Args:
        x (torch.Tensor): data of predictors.
        z (torch.Tensor): data of teacher predictors.
        y (torch.Tensor): data of responses.
        batch_size (int, optional): batch size. Defaults to 128.
        shuffle (bool, optional): whether to shuffle data. Defaults to True.
        num_workers (int, optional): number of workers. Defaults to 0.

    Returns:
        DataLoader: data loader
    """
    if y is None:
        dataset = TensorDataset(x, z)
    else:
        dataset = TensorDataset(x, y, z)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
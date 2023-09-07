from typing import Tuple
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST

# def rand_dataset(num_rows=60000, num_columns=100) -> Dataset:
#     return TensorDataset(torch.rand(num_rows, num_columns))
class rand_dataset(Dataset):
    def __init__(self, num_rows=60000, num_columns=100):
        self.data = torch.rand(num_rows, num_columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def mnist_dataset(train=True) -> Dataset:
    """
    Returns the MNIST dataset for training or testing.
    
    Args:
    train (bool): If True, returns the training dataset. Otherwise, returns the testing dataset.
    
    Returns:
    Dataset: The MNIST dataset.
    """
    return MNIST(root='./data', train=train, download=True, transform=None)
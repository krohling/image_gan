import torch

class NoiseDataset(Dataset):
    def __init__(self, len, z_size):
        self.len = len
        self.z_size = z_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return torch.randn(1, self.z_size, 1, 1)
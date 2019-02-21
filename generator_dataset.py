import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

class GeneratorDataset(Dataset):
    def __init__(self, generator, length, z_size, label, device):
        self.generator = generator
        self.length = length
        self.z_size = z_size
        self.label = label
        self.device = device
        self.items = []

    def generate(self):
        noise = torch.randn(self.length, self.z_size, 1, 1, device=self.device)
        self.items = self.generator(noise).detach().to(self.device)
        

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        input = self.items[idx]
        target = torch.FloatTensor([self.label]).to(self.device)

        return input, target
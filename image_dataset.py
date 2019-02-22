import csv
import torch
import glob
import random
from random import shuffle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

class ImageDataset(Dataset):
    def __init__(self, dataDir, transforms, label, device, search_string='*.jpg'):
        self.files = glob.glob(dataDir + '/' + search_string)
        shuffle(self.files)
        self.transform = transforms
        self.label = label
        self.device = device
        self.randomize_rate = 0
    
    def set_randomize_rate(self, randomize_rate):
        self.randomize_rate = randomize_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        image = Image.open(file_path)
        input = self.transform(image).to(self.device)

        if self.randomize_rate > 0 and random.uniform(0, 1) < self.randomize_rate:
            if self.label == 0:
                new_label = 1
            else:
                new_label = 0
            target = torch.FloatTensor([new_label]).to(self.device)
        else:
            target = torch.FloatTensor([self.label]).to(self.device)

        return input, target
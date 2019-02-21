import csv
import torch
import glob
from random import shuffle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

class ImageDataset(Dataset):
    def __init__(self, dataDir, transforms, label, search_string='*.jpg'):
        self.files = glob.glob(dataDir + '/' + search_string)
        shuffle(self.files)
        self.transform = transforms
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        image = Image.open(file_path)
        input = self.transform(image)
        target = torch.FloatTensor([self.label])

        return input, target
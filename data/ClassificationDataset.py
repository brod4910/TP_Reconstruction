import os
import torch
import pandas as pd
from torch.utils.data import Dataset
# from torchvision import transforms
from PIL import Image

class ClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform= None):
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        transform = self.transform

        img_name = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])

        image = Image.open(img_name)
        label = self.csv_data.iloc[idx, 1]
        if transform:
            image = transform(image)

        return {'input' : image, 'label' : label}
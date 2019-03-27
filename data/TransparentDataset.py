import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image

'''
'''
class TransparentDataset(Dataset):
    def __init__(self, csv_file, input_dir, gt_dir, sample, input_transforms=None, label_transforms= None):
        self.csv_data = pd.read_csv(csv_file)
        self.csv_data = self.csv_data.sample(n= sample)
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_transforms = input_transforms
        self.label_transforms = label_transforms

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        input_transforms = self.input_transforms
        label_transforms = self.label_transforms

        file_name = self.csv_data.iloc[idx, 0]

        img_name = os.path.join(self.input_dir, self.csv_data.iloc[idx, 0])
        label_name = os.path.join(self.gt_dir, self.csv_data.iloc[idx, 0])

        image = Image.open(img_name)
        label = Image.open(label_name)

        if input_transforms:
            image = input_transforms(image)
        if label_transforms:
            label = label_transforms(label)

        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label, file_name



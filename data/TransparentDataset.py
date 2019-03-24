import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image

'''
    Dataset for the lensless images.
    The csv file must contain the relative path to the images
    The root dir must be the top level directory of where all of the images.

    Bare transform is a list of image transformations. These transformations will be taken as is, if and only if
    extra_transform is None. If extra_transform is not none then bare_transform MUST contain a None entry where the
    extra transforms will be placed. 

    The extra transforms are a list of more image transformations. These transformations are randomly chosen. The
    random selection either contains one of the transformations or all of them.
'''
class TransparentDataset(Dataset):
    def __init__(self, csv_file, input_dir, gt_dir, input_transforms=None, label_transforms= None):
        self.csv_data = pd.read_csv(csv_file)
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



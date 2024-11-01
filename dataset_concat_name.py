import pandas as pd
from torch.utils.data import Dataset
import torch
import PIL
import numpy as np
from torchvision import datasets, transforms
import os
import ast

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(annotations_file, sep="\t", header=None)
        self.labels = dict(zip(df[0], df[1])) # file_name and labels
        self.image_filenames = list(self.labels.keys()) # keys will be the filename

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        # print(image_name.split('-')[0])
        if image_name.split('-')[0] == 'A2019':
            folder_name = 'A2019-NX-01/epoch_15_concat/'
            img_path = os.path.join(self.img_dir, folder_name + image_name)
            # print(img_path)
            images = torch.from_numpy(np.load(img_path).astype(np.float32)/255.0)
        elif image_name.split('-')[0] == 'A2020':
            folder_name = 'A2020-NX-01/epoch_15_concat/'
            img_path = os.path.join(self.img_dir, folder_name + image_name)
            images = torch.from_numpy(np.load(img_path).astype(np.float32)/255.0)
        else:
            img_path = os.path.join(self.img_dir, image_name)
            # no need to divide by 255.0 for 2018-NX-01
            images = torch.from_numpy(np.load(img_path).astype(np.float32))
        
        # images = torch.from_numpy(np.load(img_path).astype(np.float32)/255.0)
        labels = torch.tensor(ast.literal_eval(self.labels[image_name]))

        return image_names, images, labels
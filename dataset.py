from PIL import Image
import PIL
import numpy as np
from multiprocessing import Pool
import os
from tqdm import tqdm
import cv2
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
import ast

DATA_PATH = "./shhs1/"
IMG_PATH = "/home/hjlee/shhs1/SHHS1_duplicated_IMG/"


def transform(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.float32)
    upper = img[129:473]
    flow = img[473:602]
    flow_resized = cv2.resize(flow, (1920, 43), interpolation=cv2.INTER_AREA)
    breathing = img[602:731]
    # bottom = img[817:903]
    # concat = np.concatenate((upper, flow_resized, breathing, bottom), axis=0)
    # concat_pil = Image.fromarray(np.uint8(concat))
    # concat_pil = concat_pil.resize((224, 224))
    # final = np.array(concat_pil, dtype=np.float32)/255.0
    # final = np.transpose(final, (2, 0, 1))
    oxy = img[903:1075]
    oxy_resized = cv2.resize(oxy, (1920, 86), interpolation=cv2.INTER_AREA)
    concat = np.concatenate((upper, flow_resized, breathing, oxy_resized), axis=0)
    concat_pil = Image.fromarray(np.uint8(concat))
    concat_pil = concat_pil.resize((224, 224))
    final = np.array(concat_pil, dtype=np.float32)/255.0
    final = np.transpose(final, (2, 0, 1))
    return final


def lineout(image, p):
    if np.random.random() > p:
        return image
    channels = [i for i in range(0, 240, 16)]
    tuples = []
    for i in range(len(channels)-1):
        tuples.append((channels[i], channels[i+1]))
    c = np.random.randint(len(tuples))
    pick = tuples[c]
    y_min, y_max = pick[0], pick[1]
    image[:, y_min:y_max, :] = 0
    return image


class IntraEpochImageDataset(Dataset):
    def __init__(self, annotations_file, lineout_prob=1.):
        df = pd.read_csv(annotations_file, header=None)
        self.labels = dict(zip(df[0], df[1]))
        self.image_path = list(self.labels.keys())
        self.lineout_prob = lineout_prob

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # bring the image from corresponding location
        img_path = os.path.join(IMG_PATH, self.image_path[idx])
        label = int(self.labels[self.image_path[idx]])
        
        # process the image
        image = torch.from_numpy(transform(img_path))
        # apply line-mix if p > 0
        if self.lineout_prob > 0:
            image = lineout(image, self.lineout_prob)

        return image, label


class InterEpochImageDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path
        df = pd.read_csv(label_path, header=None)
        self.labels = dict(zip(df[0], df[1])) # file_name and labels
        self.image_filenames = list(self.labels.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        images = torch.load(image_path)
        labels = torch.tensor(ast.literal_eval(self.labels[image_path]))

        return images, labels


class ExtractVectorDataset(Dataset):
    def __init__(self, annotations_file):
        df = pd.read_csv(annotations_file, header=None)
        self.labels = dict(zip(df[0], df[1]))
        self.image_path = list(self.labels.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # bring the image from corresponding location
        img_path = os.path.join(IMG_PATH, self.image_path[idx])
        label = int(self.labels[self.image_path[idx]])
        
        # process the image
        image = torch.from_numpy(transform(img_path))

        return image, label, img_path
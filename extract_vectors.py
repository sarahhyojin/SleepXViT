import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
import timm
from collections import OrderedDict

from dataset import ExtractVectorDataset
import datetime

import pandas as pd

DATA_PATH = "./shhs1/"

def extract_features(model, data_loader, device):
    progress_bar = tqdm(data_loader)
    for img, lbl, img_paths in progress_bar:
        img, lbl = img.to(device), lbl.to(device)
        outputs = model(img).cpu().detach().numpy()
        
        for i, (output, img_path) in enumerate(zip(outputs, img_paths)):
            # Get the corresponding label for the current output
            # lb = lbl[i].item()
            patient = img_path.split('/')[5]
            img_name = img_path.split('/')[6][:-4]
            output_path = os.path.join(DATA_PATH, f"feature_vectors/{patient}/{img_name}")
            patient_path = os.path.join(DATA_PATH, f"feature_vectors/{patient}")

            if not os.path.exists(patient_path):
                os.mkdir(patient_path)
            
            if not os.path.exists(output_path):
                np.save(output_path, output)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_label_path', type=str, default=os.path.join(DATA_PATH, "shhs1_train_label.txt"))
    parser.add_argument('--valid_label_path', type=str, default=os.path.join(DATA_PATH, "shhs1_valid_label.txt"))
    parser.add_argument('--test_label_path', type=str, default=os.path.join(DATA_PATH, "shhs1_test_label.txt"))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)

    args=parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU 1 to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Multi-GPU
    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("Multi GPU activate")
    else:
        print("Device: ", device)

    # Dataset
    train_dataset = ExtractVectorDataset(args.train_label_path)
    valid_dataset = ExtractVectorDataset(args.valid_label_path)
    test_dataset = ExtractVectorDataset(args.test_label_path)

    # Set seed
    torch.manual_seed(args.seed)

    # Data load
    print("========== Load Data ==========")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    
    # Get the model
    num_classes = 5

    # Load the model
    state_dict = torch.load(os.path.join(DATA_PATH, 'ckpt/SHHS1-0610-2024-06-14.pth'))

    ##### Multi-GPU trained #####
    keys_to_delete = ['module.head.weight', 'module.head.bias']
    new_state_dict = state_dict.copy()  # Make a copy of the original state_dict

    for key in keys_to_delete:
        if key in new_state_dict:
            del new_state_dict[key]

    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=False, num_classes=0)
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(new_state_dict)

    ##### Single-GPU trained #####
    # keys_to_delete = ['head.weight', 'head.bias']
    # new_state_dict = state_dict.copy()  # Make a copy of the original state_dict

    # for key in keys_to_delete:
    #     if key in new_state_dict:
    #         del new_state_dict[key]

    # model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=False, num_classes=0)
    # model.to(device)
    # model.load_state_dict(new_state_dict)
    # model = nn.DataParallel(model)


    print("========= Start Processing - Train =========")
    extract_features(model, train_dataloader, device)
    print("========= Start Processing - Valid =========")
    extract_features(model, valid_dataloader, device)
    print("========= Start Processing - Test =========")
    extract_features(model, test_dataloader, device)
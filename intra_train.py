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
from torchvision.transforms import ToTensor, CenterCrop
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score

from dataset import IntraEpochImageDataset, InterEpochImageDataset
import wandb
import datetime
import timm
from utils import *

DATA_PATH = "./shhs1/"

if __name__ == '__main__':
    tqdm._instances.clear()
    wandb.init(project="SHHS-ViT")
    # args
    parser = argparse.ArgumentParser()

    # large
    parser.add_argument('--train_label_path', type=str, default=os.path.join(DATA_PATH, "shhs1_train_label.txt"))
    parser.add_argument('--eval_label_path', type=str, default=os.path.join(DATA_PATH, "shhs1_valid_label.txt"))
    parser.add_argument('--ckpt_path', type=str, default=os.path.join(DATA_PATH, "shhs1/ckpt/"))
    parser.add_argument('--model_name', type=str, default='SHHS1-github-')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)  # use 1 GPUS
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--decay', type=float, default=1e-5) # 1e-5 was best

    args=parser.parse_args()
    wandb.config.update(args)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU num to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("Multi GPU activate")
    else:
        print("Device: ", device)

    # Dataset
    train_dataset = IntraEpochImageDataset(args.train_label_path)
    eval_dataset = IntraEpochImageDataset(args.eval_label_path)

    # Set seed
    torch.manual_seed(args.seed)

    # Data load
    print("========== Load Data ==========")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    # Get the model
    num_classes = 5
    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True, num_classes=num_classes)
    # model = nn.DataParallel(model)
    model.to(device)

    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay) # this is best
    wandb.watch(model)
    loss_fn = nn.CrossEntropyLoss()

    # Create a cosine annealing learning rate scheduler
    # steps_per_epoch = len(train_dataset) // args.batch_size
    # scheduler = lr_scheduler.stepLR(optimizer, T_max=args.epochs * steps_per_epoch)

    # Train
    num_epochs = args.epochs
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Get the current date
    model_name = args.model_name + current_date

    min_loss = np.inf

    print("========= Train Model =========")
    for epoch in range(0, num_epochs, 1): # evaluate every epoch
        # train_loss, train_acc, train_f1 = model_train(model, train_dataloader, loss_fn, optimizer, device)
        train_loss, train_acc, train_f1 = model_train_linemix(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc, val_f1 = model_evaluate(model, eval_dataloader, loss_fn, device)

        if val_loss < min_loss:
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, f'{model_name}.pth'))
  
        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, f1: {train_f1:.5f},\
                val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}, val_f1: {val_f1:.5f}')

        wandb.log({
        "Train Accuracy": 100. * train_acc,
        "Train Loss": train_loss,
        "Train F1": 100 * train_f1,
        "Test Accuracy": 100. * val_acc,
        "Test Loss": val_loss,
        "Test F1": 100 * val_f1})

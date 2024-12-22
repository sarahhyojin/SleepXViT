import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from inter_epoch_transformer import InterEpochTransformer
from dataset import InterEpochImageDataset
import wandb
import datetime
import pickle
from utils import *

DATA_PATH = "/home/hjlee/shhs1/shhs1-trained/"
SEQ_LEN = 10

if __name__ == '__main__':
    tqdm._instances.clear()
    wandb.init(project="SHHS-ViT")
    # args
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_paths', type=str, default=os.path.join(DATA_PATH, f"train_vector_{SEQ_LEN}/"))
    parser.add_argument('--train_labels', type=str, default=os.path.join(DATA_PATH, f"train_{SEQ_LEN}_label.txt"))
    parser.add_argument('--valid_paths', type=str, default=os.path.join(DATA_PATH, f"valid_vector_{SEQ_LEN}/"))
    parser.add_argument('--valid_labels', type=str, default=os.path.join(DATA_PATH, f"valid_{SEQ_LEN}_label.txt"))
    parser.add_argument('--model_name', type=str, default=f'Inter-{SEQ_LEN}-SHHS1-')
    parser.add_argument('--ckpt_path', type=str, default=os.path.join(DATA_PATH, "ckpt/"))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_seq', type=int, default=SEQ_LEN)
    parser.add_argument('--lr', type=float, default=1e-5) # 1e-4
    parser.add_argument('--decay', type=float, default=1e-8)
    args=parser.parse_args()
    wandb.config.update(args)

    os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU num to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Multi-GPU
    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("Multi GPU activate")
    else:
        print("Device: ", device)

    # Dataset
    train_dataset = InterEpochImageDataset(args.train_paths, args.train_labels)
    eval_dataset = InterEpochImageDataset(args.valid_paths, args.valid_labels)

    # Set seed
    torch.manual_seed(args.seed)

    # Data load
    print("========== Load Data ==========")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    # Add seq-attention & Multi-GPU option
    model = InterEpochTransformer(num_classes=5, embed_dim=768, depth=4,
                 num_heads=8, num_seq=args.num_seq, mlp_ratio=0.5, qkv_bias=False, mlp_head=False, drop_rate=0.2, attn_drop_rate=0.2)
    model = nn.DataParallel(model)
    model.to(device)

    # Optimizer & Loss function
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5) 

    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay) # this is the best
    # optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4) # experiment
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    wandb.watch(model)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    num_epochs = args.epochs
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Get the current date
    model_name = args.model_name + current_date
    num_seq = args.num_seq

    min_f1 = -np.inf

    print("========= Train Model =========")
    for epoch in range(0, num_epochs, 1): # evaluate every epoch
        train_labels, train_preds, train_loss, train_acc, train_f1 = model_train_inter(model, train_dataloader, loss_fn, optimizer, device, num_seq)
        val_labels, val_preds, val_loss, val_acc, val_f1 = model_eval_inter(model, eval_dataloader, loss_fn, device, num_seq)

        if val_f1 > min_f1:
            print(f'[INFO] val_f1 has been improved from {min_f1:.5f} to {val_f1:.5f}. Saving Model!')
            min_f1 = val_f1
            torch.save(model.state_dict(), f'/home/hjlee/shhs1/shhs1-trained/checkpoint/{model_name}.pth')
            # with open(f'/tf/data_AIoT1/Att_weights/{model_name}_labels.pkl', 'wb') as l:
            #     pickle.dump(val_labels, l)
            # with open(f'/tf/data_AIoT1/Att_weights/{model_name}_preds.pkl', 'wb') as p:
            #     pickle.dump(val_preds, p)
  
        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, f1: {train_f1:.5f},\
                val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}, val_f1: {val_f1:.5f}')

        wandb.log({
        "Train Accuracy": 100. * train_acc,
        "Train Loss": train_loss,
        "Train F1": 100 * train_f1,
        "Test Accuracy": 100. * val_acc,
        "Test Loss": val_loss,
        "Test F1": 100 * val_f1})
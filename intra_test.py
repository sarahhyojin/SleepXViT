import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import timm

from dataset import IntraEpochImageDataset
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import classification_report

DATA_PATH = "./shhs1/"

def model_test(model, data_loader, device, save_path):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ('Wake', 'N1', 'N2', 'N3', 'REM')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1).reshape((5, 1)), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (5,4))
    sn.heatmap(df_cm, annot=True, cmap="Blues")

    # Save confusion matrix plot to the specified directory
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'intra_confusion_matrix_{current_date}.png'))

    # Show confusion matrix plot
    plt.show()


    # classification report
    report = classification_report(y_true, y_pred, target_names = classes, digits=4)
    report_filename = 'intra_classification_report.txt'
    report_path = os.path.join(save_path, report_filename) if save_path else report_filename

    print(report)

    # cohen's kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa}")

    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f"\nCohen's Kappa: {kappa:.4f}")


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_lbl_dir', type=str, default=os.path.join(DATA_PATH, "shhs1_test_label.txt"))
    parser.add_argument('--model_name', type=str, default="SHHS-Intra-")
    parser.add_argument('--ckpt_path', type=str, default=os.path.join(DATA_PATH, "ckpt/SHHS1-0610-2024-06-14.pth"))
    parser.add_argument('--save_path', type=str, default=os.path.join(DATA_PATH, "result/"))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512) # faster than 320

    args=parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU 2 to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seed
    torch.manual_seed(args.seed)

    # Dataset
    test_dataset = IntraEpochImageDataset(args.test_lbl_dir)

    # Set seed
    torch.manual_seed(args.seed)

    # Data load
    print("========== Load Data ==========")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    # Get the model
    num_classes = 5
    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=False, num_classes=num_classes)
    model.to(device)

    # Load the model
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.ckpt_path))

    # Test
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Get the current date
    model_name = args.model_name + current_date


    print("========= Test Model =========")
    model_test(model, test_dataloader, device, args.save_path)
  

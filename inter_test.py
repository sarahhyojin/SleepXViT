import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import timm

from inter_epoch_transformer import InterEpochTransformer
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import classification_report
from collections import OrderedDict
import pickle

DATA_PATH = "/home/hjlee/shhs1/shhs1-trained/"
SEQ_LEN = 10
CUR_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

def sliding_windows(patient_vector, num_seq, batch_size, length, model, device):
    """
    patient_vector : extracted feature vectors per image and concatenated from one patient
    num_seq : sliding window's kernel size
    batch_size : batch size to be processed
    length : length of the total epoch of one patient
    model : trained model
    final : return the prediction of sequence that of one patient (softmax aggregated)
    """
    final = [torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32, device=device) for _ in range(length)]

    current_seq = []

    with torch.no_grad():
        for i in range(length - num_seq + 1):
            current_seq.append(patient_vector[i:i+num_seq])

            if (i+1) % batch_size == 0:
                batch_seq = torch.stack(current_seq).to(device)
                current_seq = []
                output = model(batch_seq)

                for k in range(batch_size):  # batch
                    for j in range(num_seq):
                        final[k + j + i - (batch_size-1)].add_(torch.softmax(output[j][k], dim=0))

        # for remainder
        if len(current_seq) != 0:
            batch_seq = torch.stack(current_seq).to(device)
            output_2 = model(batch_seq)

            new_i = batch_size * ((length - num_seq + 1)//batch_size)

            for k in range(output_2[0].shape[0]):  # batch
                for j in range(num_seq):  # sequence length
                    final[k + j + new_i].add_(torch.softmax(output_2[j][k], dim=0))

    return final

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()


    # large
    parser.add_argument('--img_dir', type=str, default=os.path.join(DATA_PATH, "vectors"))
    parser.add_argument('--test_labels', type=str, default=os.path.join(DATA_PATH, "shhs1_test_label.txt"))
    parser.add_argument('--model_name', type=str, default=f'Seq-sliding-{SEQ_LEN}-test-{CUR_DATE}')
    parser.add_argument('--checkpoint_path', type=str, default=os.path.join(DATA_PATH, "checkpoint/Inter-10-SHHS1-2024-06-25.pth")) # use one with relprop
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_seq', type=int, default=SEQ_LEN)
    parser.add_argument('--save_path', type=str, default = "/home/hjlee/SleepXViT-1/shhs1/result/")

    args=parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # read test_labels
    df_file_list = pd.read_csv(args.test_labels, header=None)
    # files = df_file_list[0].tolist()
    # df_file_list[0] = df_file_list[0].apply(lambda x: "/".join(x.split("/")[5:]))
    # create patient_dict
    patients = [x.split("/")[0] for x in df_file_list[0].tolist()]
    patient_dict = {}

    for p in patients:
        if p in patient_dict:
            patient_dict[p] += 1
        else:
            patient_dict[p] = 1

    
    y_softmax = []
    y_pred = []
    y_true = []

    # model
    print("========= Load Model =========")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Multi-GPU
    os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU 1 to use
    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("Multi GPU activate")
    else:
        print("Device: ", device)

    model = InterEpochTransformer(num_classes=5, embed_dim=768, depth=4,
        num_heads=8, num_seq=args.num_seq, mlp_ratio=0.5, qkv_bias=False, mlp_head=False, drop_rate=0.2, attn_drop_rate=0.2)
    # model = nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(checkpoint)
    model.to(device)

    print("========= Process by patient =========")
    for patient, length in tqdm(patient_dict.items(), total=len(patient_dict)):
        # hospital = '-'.join(patient.split('-')[:-1])
        patient_epoch = df_file_list[df_file_list[0].str.startswith(patient)]
        sorted_epoch = patient_epoch.sort_values(by=0)
        
        temp = []
        lbl = []
        # sorted_epoch = sorted(os.listdir(patient_folder))
        # print(sorted_epoch)
        for index, row in sorted_epoch.iterrows():
            # print(f"Loading {i}th vector")
            # vector_np = torch.from_numpy(np.load(os.path.join(patient_folder, row[0])).astype(np.float32))
            vector_np = torch.from_numpy(np.load(os.path.join(args.img_dir, row[0].replace(".png", ".npy"))).astype(np.float32))
            temp.append(vector_np)
            lbl.append(row[1])
            # save the image name
            # print(len(temp))
        patient_vector = torch.stack(temp)

        final = sliding_windows(patient_vector=patient_vector, num_seq=args.num_seq, batch_size = args.batch_size,
                                length = length, model = model, device=device)
        y_softmax.extend(final)
        y_true.extend(lbl)
        
        # print(patient_vector.shape, len(lbl))
    
    print("========= Inference Ended =========")

    for seq in y_softmax:
        _, pred = seq.max(dim=0)
        y_pred.append(pred.item())
    
    # save the labels and 
    with open(os.path.join(args.save_path, f'{args.model_name}_labels.pkl'), 'wb') as l:
        pickle.dump(y_true, l)
    with open(os.path.join(args.save_path, f'{args.model_name}_preds.pkl'), 'wb') as p:
        pickle.dump(y_pred, p)

    classes = ('Wake', 'N1', 'N2', 'N3', 'REM')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1).reshape((5, 1)), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (5,4))
    sn.heatmap(df_cm, annot=True, cmap="Blues")

    # Save confusion matrix plot to the specified directory
    plt.savefig(os.path.join(args.save_path, f'inter_confusion_matrix_{CUR_DATE}.png'))

    # Show confusion matrix plot
    plt.show()

    # classification report
    report = classification_report(y_true, y_pred, target_names = classes, digits=4)
    print(report)
    report_filename = f'classification_report_test_{CUR_DATE}.txt'
    report_path = os.path.join(args.save_path, report_filename)

    # cohen's kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa}")

    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f"\nCohen's Kappa: {kappa:.4f}")
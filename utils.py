import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import pandas as pd
from PIL import Image
import cv2
from torch.utils.data import Dataset
import ast

def model_train(model, data_loader, loss_fn, optimizer, device):

    model.train()
    running_loss = 0
    corr = 0
    all_preds = []
    all_labels = []

    prograss_bar = tqdm(data_loader)
    for img, lbl in prograss_bar:
        img, lbl = img.to(device), lbl.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, lbl)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        _, pred = output.max(dim=1)
        
        # accuracy
        corr += pred.eq(lbl).sum().item()
        # macro f1 score
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())
        running_loss += loss.item() * img.size(0)

    acc = corr / len(data_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return running_loss / len(data_loader.dataset), acc, f1


def model_train_linemix(model, data_loader, loss_fn, optimizer, device):

    model.train()
    running_loss = 0
    corr = 0
    all_preds = []
    all_labels = []

    channels = [i for i in range(0, 240, 16)]
    tuples = []
    for i in range(len(channels)-1):
        tuples.append((channels[i], channels[i+1]))

    prograss_bar = tqdm(data_loader)
    for img, lbl in prograss_bar:
        # generate mixed sample
        rand_index = torch.randperm(img.size()[0]) # shuffle random index of the batch
        img = img.to(device)
        labels_a = lbl.to(device) # original labels
        labels_b = lbl[rand_index].to(device) # randomly mixed labels

        # select channel
        c = np.random.randint(len(tuples)) # what is channel?
        pick = tuples[c]
        y_min, y_max = pick[0], pick[1]
        img[:, :, y_min:y_max, :] = img[rand_index, :, y_min:y_max, :]

        # adjust lambda to exactly match pixel ratio
        lam = (y_max - y_min) / 224

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, labels_a)  * (1. - lam) + loss_fn(output, labels_b) * lam
        loss.backward()
        optimizer.step()
        # scheduler.step()

        _, pred = output.max(dim=1)
        
        # accuracy
        corr += pred.eq(labels_a).sum().item()
        # macro f1 score
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels_a.cpu().numpy())
        running_loss += loss.item() * img.size(0)

    acc = corr / len(data_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return running_loss / len(data_loader.dataset), acc, f1


def model_train_inter(model, data_loader, loss_fn, optimizer, device, num_seq):

    model.train()
    running_loss = 0
    corr = 0
    all_preds = []
    all_labels = []
    losses = []
    attention_scores_list = []

    prograss_bar = tqdm(data_loader)
    for img, lbl in prograss_bar:
        img, lbl = img.to(device), lbl.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = 0
        
        for ep in range(num_seq):
            loss += loss_fn(outputs[ep], lbl[:,ep])
            _, pred = outputs[ep].max(dim=1)
            corr += pred.eq(lbl[:,ep]).sum().item() # number of correct predictions
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lbl[:,ep].cpu().numpy())
            
        loss.backward()
        optimizer.step()
        losses.append(loss)
    
    avg_loss = sum(losses) / (len(losses) * num_seq)
    acc = corr / (len(data_loader.dataset) * num_seq)
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    return all_labels, all_preds, avg_loss, acc, f1


def model_evaluate(model, data_loader, loss_fn, device):

    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        all_preds = []
        all_labels = []
        prograss_bar = tqdm(data_loader)

        for img, lbl in prograss_bar:
            img, lbl = img.to(device), lbl.to(device)

            output = model(img)
            _, pred = output.max(dim=1)

            corr += torch.sum(pred.eq(lbl)).item()
            # macro f1 score
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            running_loss += loss_fn(output, lbl).item() * img.size(0)

        acc = corr / len(data_loader.dataset)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return running_loss / len(data_loader.dataset), acc, f1


def model_eval_inter(model, data_loader, loss_fn, device, num_seq):

    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        all_preds = []
        all_labels = []
        losses = []
        attention_scores_list = []
        
        prograss_bar = tqdm(data_loader)

        for img, lbl in prograss_bar:
            img, lbl = img.to(device), lbl.to(device)
            
            outputs = model(img)
            loss = 0
            # calculate loss by each epoch
            for ep in range(num_seq):
                loss += loss_fn(outputs[ep], lbl[:,ep])
                _, pred = outputs[ep].max(dim=1)
                corr += pred.eq(lbl[:,ep]).sum().item() # number of correct predictions
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(lbl[:,ep].cpu().numpy())
                
            losses.append(loss)
            
        avg_loss = sum(losses) / (len(losses) * num_seq)
        acc = corr / (len(data_loader.dataset) * num_seq)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return all_labels, all_preds, avg_loss, acc, f1

class EarlyStopping:
    def __init__(self, patience=2, verbose=False, delta=0, path='model.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

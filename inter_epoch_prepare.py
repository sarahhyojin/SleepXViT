import pandas as pd
from torch.utils.data import Dataset
import torch
import PIL
import numpy as np
from torchvision import datasets, transforms
import os
import multiprocessing
from tqdm import tqdm
import pickle
from functools import partial
from contextlib import contextmanager

SEQ_LEN = 10
DATA_PATH = "/home/hjlee/shhs1/"
VECTOR_DIR = "/home/hjlee/shhs1/shhs1-trained/"
FLAG = ["train", "valid", "test"]

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def save_sequence(patient_list, flag):
    df_patient = pd.read_csv(patient_list, header=None)
    # df_patient[0] = df_patient[0].apply(lambda x: "/".join(x.split("/")[5:]))
    patient_list = [x.split("/")[0] for x in df_patient[0].tolist()]
    # delete the duplicates
    patient_list = set(patient_list)
    image_paths = []

    for patient in tqdm(patient_list):
        patient_folder = os.path.join(VECTOR_DIR, "vectors", patient)
        if os.path.exists(patient_folder):
            tmp_img = []
            for image_files in os.listdir(patient_folder):
                image_path = os.path.join(patient_folder, image_files)
                tmp_img.append(image_path)

            num_sequence = len(tmp_img) // SEQ_LEN
            start_idx = 0
            tmp_img = sorted(tmp_img)
            for i in range(num_sequence):
                sequence = tmp_img[start_idx : start_idx + SEQ_LEN]
                image_paths.append(sequence)
                start_idx += SEQ_LEN
    
    return image_paths


def create_labels(image_list, train_set, fold):
    # with open(file_path, 'rb') as picklefile:
    #     image_list = pickle.load(picklefile)
    
    new_data = {'file_label': [], 'labels': []}

    for data in image_list:
        temp = []
        # changed here
        start_epoch = data[0][-8:-4]
        end_epoch = data[-1][-8:-4]
        patient = data[0].split('/')[6]

        for image_path in data:
            image = "/".join(image_path.split("/")[6:])[:-4] + '.png'  # changed here
            label = train_set[image]
            temp.append(label)

        new_dir = VECTOR_DIR + fold + f"_vector_{SEQ_LEN}"
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        new_path = os.path.join(new_dir, f"{patient}_{start_epoch}_{end_epoch}.pt")

        new_data['file_label'].append(new_path)
        new_data['labels'].append(temp)

    new_data = pd.DataFrame(new_data)
    new_data.to_csv(VECTOR_DIR + fold + f"_{SEQ_LEN}_label.txt", index=False, header=False)


def stack_vectors(seq, fold):
    # print(fold)
    images = []
    for img_path in seq:
        # print(img_path)
        image = torch.from_numpy(np.load(img_path).astype(np.float32))
        images.append(image)

    start_epoch = seq[0][-8:-4]
    end_epoch = seq[-1][-8:-4]
    patient = seq[0].split('/')[6]

    images = torch.stack(images)
    output_folder = VECTOR_DIR + FLAG[fold] + f"_vector_{SEQ_LEN}/"
    output_path = os.path.join(output_folder, f"{patient}_{start_epoch}_{end_epoch}.pt")
    torch.save(images, output_path)


if __name__ == '__main__':
    train_patient_path = os.path.join(DATA_PATH, "shhs1_test_label.txt")
    valid_patient_path = os.path.join(DATA_PATH, "shhs1_train_label.txt")
    test_patient_path = os.path.join(DATA_PATH, "shhs1_valid_label.txt")

    print("==========Save path's sequence==========")
    train_paths = save_sequence(train_patient_path, "train")
    valid_paths = save_sequence(valid_patient_path, "valid")
    test_paths = save_sequence(test_patient_path, "test")

    df_train = pd.read_csv(train_patient_path, header=None)
    df_valid = pd.read_csv(valid_patient_path, header=None)
    df_test = pd.read_csv(test_patient_path, header=None)

    # delete the path
    # df_train[0] = df_train[0].apply(lambda x: "/".join(x.split("/")[5:]))
    # df_valid[0] = df_valid[0].apply(lambda x: "/".join(x.split("/")[5:]))
    # df_test[0] = df_test[0].apply(lambda x: "/".join(x.split("/")[5:]))

    train_labels = dict(zip(df_train[0], df_train[1]))
    valid_labels = dict(zip(df_valid[0], df_valid[1]))
    test_labels = dict(zip(df_test[0], df_test[1]))

    print("==========Creating labels for sequence==========")
    create_labels(train_paths, train_labels, "train")
    create_labels(valid_paths, valid_labels, "valid")
    create_labels(test_paths, test_labels, "test")

    print("==========Stack vectors as sequence==========")
    with poolcontext(processes=8) as pool:
        pool.starmap(stack_vectors, tqdm(zip(train_paths, [0]*len(train_paths)), total=len(train_paths)))
        pool.starmap(stack_vectors, tqdm(zip(valid_paths, [1]*len(valid_paths)), total=len(valid_paths)))
        pool.starmap(stack_vectors, tqdm(zip(test_paths, [2]*len(test_paths)), total=len(test_paths)))

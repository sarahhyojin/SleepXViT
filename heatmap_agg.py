import os
import argparse
import copy
import json
import cv2
import ast

import numpy as np
import pandas as pd
from tqdm import tqdm


def match_row_sum(file_path):
    
    heatmap = np.load(file_path)
    
    # color map values
    color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET).reshape(-1, 3)
    value_map = {tuple(color): value for value, color in enumerate(reversed(color_map))}
    
    # mapping
    mapped_values = np.vectorize(lambda r, g, b: value_map.get((r, g, b), 0), otypes=[np.uint8])
    data_values = mapped_values(heatmap[:, :, 2], heatmap[:, :, 1], heatmap[:, :, 0])
    
    new = (255-data_values) / 255
    row_sums = np.sum(new, axis=1)
    
    return row_sums, data_values


def make_filepath(files, heatmap_path):

    # make filepath
    img_files = []
    for fr in files:
        file_path = os.path.join(heatmap_path, fr)
        img_files.append(file_path)
        
    return img_files


def stack_sum(img_files, save_name, save_root='./res'):
    
    stack = None
    intensity = None
    
    for file in tqdm(img_files):
        row_sum, data_value = match_row_sum(file)
        
        if stack is None:
            stack = np.expand_dims(row_sum, axis=0)
        else:
            stack = np.concatenate((stack, np.expand_dims(row_sum, axis=0)), axis=0)
            
        if intensity is None:
            intensity = np.expand_dims(data_value, axis=0)
        else:
            intensity = np.concatenate((intensity, np.expand_dims(data_value, axis=0)), axis=0)
        
    stack_save_path = os.path.join(save_root, save_name+'_stack.npy')
    np.save(stack_save_path, stack)
    
    intensity_save_path = os.path.join(save_root, save_name+'_intensity.npy')
    np.save(intensity_save_path, intensity)

    return stack, intensity


# =============================================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_res", type=int, default=10000)
    parser.add_argument("--heatmap_path", type=str, 
                        default= '/tf/hjlee/psg_image_codes/explainability_code/Clustering-Images/heatmap')
    parser.add_argument("--inf_res", type=str, 
                        default='/tf/hjlee/psg_image_codes/explainability_code/group_by_gt_1027.txt')
    parser.add_argument("--save_root", type=str, 
                        default='/tf/hjlee/psg_image_codes/explainability_code/res')
    
    args = parser.parse_args()   
    
    # inference result - select files
    df = pd.read_csv(args.inf_res, sep='\t')
    df['data'] = df['data'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['data'] = df.apply(lambda row: row['data'][:args.num_res] if row['count'] > args.num_res else row['data'], axis=1)

    labels_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    for _, group in df.iterrows():
        files = group['data']
        y_true = labels_map[group['y_true']]
        y_pred = labels_map[group['y_pred_model']]
        save_name = y_true + '_' + y_pred
        print(f'=========== {save_name} heatmap agg start ===========')
        file_paths = make_filepath(files, args.heatmap_path)
        _, _ = stack_sum(file_paths, save_name, args.save_root)
    
    
if __name__ == "__main__":
    main()
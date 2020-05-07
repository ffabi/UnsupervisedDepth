import os

import cv2
import numpy as np


def masked_mae(y_true, y_pred, weight = 1):
    mask = (y_pred != [0, 0, 0]).all(-1)


    y_pred = y_pred[mask]
    y_true = y_true[mask]
    y_pred = y_pred.astype(int)
    y_pred = y_pred.astype(int)
    return np.mean(np.mean(np.abs(y_pred - y_true), axis = -1)) * weight

def compare_orig_syn_plus_one():
    original_path = "/root/ffabi_shared_folder/datasets/_original_datasets/kitti_all/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000081.png"
    syn_path = "/root/ffabi_shared_folder/MSC/UnsupervisedDepth_data/saved_synthesized_images/synthesized_image_1.png"

    orig = cv2.imread(original_path)
    syn = cv2.imread(syn_path)

    print(masked_mae(orig, syn))

def compare_orig_syn_plus_zero():
    original_path = "/root/ffabi_shared_folder/datasets/_original_datasets/kitti_all/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000080.png"
    syn_path = "/root/ffabi_shared_folder/MSC/UnsupervisedDepth_data/saved_synthesized_images/synthesized_image_0.png"

    orig = cv2.imread(original_path)
    syn = cv2.imread(syn_path)

    print(masked_mae(orig, syn))

def compare_adjacents():
    f1_path = "/root/ffabi_shared_folder/datasets/_original_datasets/kitti_all/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000081.png"
    f2_path = "/root/ffabi_shared_folder/datasets/_original_datasets/kitti_all/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000082.png"

    f1 = cv2.imread(f1_path)
    f2 = cv2.imread(f2_path)

    print(masked_mae(f1, f2))


if __name__ == '__main__':
    compare_orig_syn_plus_one()
    compare_orig_syn_plus_zero()
    compare_adjacents()

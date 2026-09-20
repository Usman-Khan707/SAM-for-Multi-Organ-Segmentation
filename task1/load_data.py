import json
import torch
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

num_classes = 13  # except background


def get_center(label):
    """
    Input: label(H, W), np.array
    Output: A list of centers, each center is np.array [x, y] if the organ is present in the image, otherwise -1
    """
    centers = [-1]
    for i in range(1, num_classes + 1):
        if np.sum(label == i) == 0:
            centers.append(-1)
        else:
            tmp_label = np.where(label == i, 1, 0)
            dist = ndimage.distance_transform_edt(tmp_label)
            center = np.unravel_index(np.argmax(dist), dist.shape)
            centers.append(np.array([center[1], center[0]]))
    return centers


def get_bbox(label, centers, padding=1):
    """
    Input: label(H, W), np.array
    Output: A list of bboxes, each center is np.array [x1, y1, x2, y2] if the organ is present in the image,
    otherwise -1
    """
    bboxes = [-1]
    for i in range(1, num_classes + 1):
        if isinstance(centers[i], int):
            bboxes.append(-1)
        else:
            x, y = np.where(label == i)
            x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
            x1 = max(0, x1 - padding)
            x2 = min(label.shape[0], x2 + padding)
            y1 = max(0, y1 - padding)
            y2 = min(label.shape[1], y2 + padding)
            bboxes.append(np.array([y1, x1, y2, x2]))
    return bboxes


def random_selection(label, centers, num, except_center=True):
    """
    Input: label(H, W), np.array, num: (int), num of points to be selected, centers: list of np.array [x, y] or -1,
    center of the organs; except_center(bool): whether we want to select points except the center.
    Output: A list of lists, including the selected points (np.array) for each organ present in the image.
    """
    num_classes = 13
    points = [-1]
    for i in range(1, num_classes + 1):
        if centers[i].sum() == -1:
            points.append(-1)
        else:
            if except_center:
                label[centers[i][0][0], centers[i][0][1]] = 0
            x, y = np.where(label == i)
            num = min(num, len(x))
            idx = np.random.choice(len(x), num, replace=False)
            x, y = x[idx], y[idx]
            points.append(np.array([y, x]).transpose((1, 0)))
    return points


class CTDataset:
    def __init__(self, config, mode, horizontal_only=True):
        self.config = config
        self.mode = mode
        with open(config.dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        self.data_info = dataset_info['training']
        if mode == 'valid':
            self.data_info = dataset_info['validation']
        elif mode == 'test':
            self.data_info = dataset_info['test']
        if isinstance(self.data_info[0], str):
            self.img_lst = self.data_info
            self.label_lst = None
        else:
            self.img_lst = [data['image'] for data in self.data_info]
            self.label_lst = [data['label'] for data in self.data_info]
        self.total_images = []
        self.total_labels = []
        self.centers = []
        self.bboxes = []
        self.organs = []
        print(f"Loading {mode} data...")
        if mode == 'test':
            for idx in tqdm(range(len(self.img_lst))):
                self.img_lst[idx] = os.path.join(config.data_root_path, self.img_lst[idx])
                tmp_image_data = nib.load(self.img_lst[idx])
                tmp_images = tmp_image_data.get_fdata()
                num_slices_hori = tmp_images.shape[-1]
                img_size = tmp_images.shape[0]
                for i in range(num_slices_hori):
                    self.info_update(tmp_images[:, :, i])
                if not horizontal_only:
                    full_image = np.ones((img_size, img_size, img_size), dtype=tmp_images.dtype) * np.int16(-1024)
                    full_image[:, :, :num_slices_hori] = tmp_images
                    num_slices_vert = tmp_images.shape[0]
                    for i in range(num_slices_vert):
                        self.info_update(full_image[i, :, :])
                        self.info_update(full_image[:, i, :])
        else:
            for idx in tqdm(range(len(self.img_lst))):
                self.img_lst[idx] = os.path.join(config.data_root_path, self.img_lst[idx])
                self.label_lst[idx] = os.path.join(config.data_root_path, self.label_lst[idx])
                tmp_image_data = nib.load(self.img_lst[idx])
                tmp_images = tmp_image_data.get_fdata()
                tmp_label_data = nib.load(self.label_lst[idx])
                tmp_labels = tmp_label_data.get_fdata()
                num_slices_hori = tmp_images.shape[-1]
                img_size = tmp_images.shape[0]
                for i in range(num_slices_hori):
                    if np.sum(tmp_labels[:, :, i]) == 0:
                        continue
                    self.info_update(tmp_images[:, :, i], tmp_labels[:, :, i])
                if not horizontal_only:
                    full_image = np.ones((img_size, img_size, img_size), dtype=tmp_images.dtype) * np.int16(-1024)
                    full_label = np.zeros((img_size, img_size, img_size), dtype=tmp_labels.dtype)
                    full_image[:, :, :num_slices_hori] = tmp_images
                    full_label[:, :, :num_slices_hori] = tmp_labels
                    num_slices_vert = tmp_images.shape[0]
                    for i in range(num_slices_vert):
                        if np.sum(full_label[i, :, :]) != 0:
                            self.info_update(full_image[i, :, :], full_label[i, :, :])
                        if np.sum(full_label[:, i, :]) != 0:
                            self.info_update(full_image[:, i, :], full_label[:, i, :])
        print(f'Total number of {mode} CT slices: {len(self.total_images)}')
        # data = images.get_fdata()
        # num_slices = data.shape[-1]
        # fig, axes = plt.subplots(5, 5, figsize=(55, 55))
        # for i in range(5):
        #     for j in range(5):
        #         axes[i, j].imshow(data[:, :, 5 * (i + 16) + j], cmap='gray')
        #         axes[i, j].axis('off')
        # plt.show()

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        return self.total_images[idx], self.total_labels[idx], self.centers[idx], self.bboxes[idx], self.organs[idx]

    def info_update(self, img, label=None):
        self.total_images.append(img)
        if label is not None:
            self.total_labels.append(label)
            self.centers.append(get_center(label))
            self.bboxes.append(get_bbox(label, self.centers[-1]))
            organ = np.zeros((1 + self.config.num_classes))
            for i in range(1, self.config.num_classes + 1):
                if not isinstance(self.centers[-1][i], int):
                    organ[i] = 1
            self.organs.append(organ)

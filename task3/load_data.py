import json
import torch
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
import random
from segment_anything.utils.transforms import ResizeLongestSide


num_classes = 13  # except background


def get_center(label):
    """
    Input: label(H, W), np.array of 0-1 elements
    Output: np.array [x, y]
    """
    dist = ndimage.distance_transform_edt(label)
    center = np.unravel_index(np.argmax(dist), dist.shape)
    return np.array([center[1], center[0]])


def get_bbox(label, padding=1):
    """
    Input: label(H, W), np.array
    Output: np.array [x1, y1, x2, y2]
    """
    x, y = np.where(label == 1)
    x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
    x1 = max(0, x1 - padding)
    x2 = min(label.shape[0], x2 + padding)
    y1 = max(0, y1 - padding)
    y2 = min(label.shape[1], y2 + padding)
    return np.array([y1, x1, y2, x2])


def random_selection(label, center, num, choose=1, except_center=True):
    """
    Input: label(H, W), np.array, num: (int), num of points to be selected, center: np.array [x, y],
    center of the organ; except_center(bool): whether we want to select points except the center.
    Output: A list of lists, including the selected points (np.array) for each organ present in the image.
    """
    if except_center:
        label[center[0], center[1]] = 0
    x, y = np.where(label == choose)
    num = min(num, len(x))
    idx = np.random.choice(len(x), num, replace=False)
    x, y = x[idx], y[idx]
    return np.array([y, x]).transpose((1, 0))


def grid_selection(label, num):
    img_size = label.shape[-1]
    seq_x = np.linspace(0, img_size - 1, num, dtype=int)[1:-1]
    seq_y = np.linspace(0, img_size - 1, num, dtype=int)[1:-1]
    result = np.array([[y, x, label[x, y]] for x in seq_x for y in seq_y])
    return result


class CTDataset1:
    def __init__(self, config, mode, horizontal_only=True, model=None):
        self.config = config
        self.mode = mode
        with open(config['dataset_info_path'], 'r') as f:
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
            # Change here to train on a small subset
            # self.img_lst = [data['image'] for data in self.data_info[0:1]]
            # self.label_lst = [data['label'] for data in self.data_info[0:1]]
        self.total_images = []
        self.total_labels = []
        self.centers = []
        self.bboxes = []
        self.organs = []
        self.img_emb = []
        self.grid_emb = []
        self.prompt_type = config['prompt_type']
        self.prompt_choice = config['prompt_choice']
        self.prompt_num = config['prompt_num']
        if config['embedded']:
            self.transform = ResizeLongestSide(config['img_input_size'][0])
            self.model = model
        print(f"Loading {mode} data...")
        if mode == 'test':
            for idx in tqdm(range(len(self.img_lst))):
                self.img_lst[idx] = os.path.join(config['data_root_path'], self.img_lst[idx])
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
                self.img_lst[idx] = os.path.join(config['data_root_path'], self.img_lst[idx])
                self.label_lst[idx] = os.path.join(config['data_root_path'], self.label_lst[idx])
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
                if config['prompt_type'] == 'grid_points':
                    for i in range(len(self.total_labels)):
                        self.grid_emb.append(self.get_grid_points(i))
        print(f'Total number of {mode} CT slices: {len(self.total_images)}')


    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):

        if self.config['embedded']:
            return self.img_emb[idx], self.total_labels[idx], self.get_prompt(idx), self.organs[idx]
        else:
            return self.total_images[idx], self.total_labels[idx], self.get_prompt(idx), self.organs[idx]

    def info_update(self, img, label=None):
        if label is None:
            self.total_images.append(img)
        else:
            image_embedding = self.get_img_emb(img, self.model) if self.config['embedded'] else None
            cnt = 0
            for i in range(1, self.config['num_classes'] + 1):
                # delete small organs with pixels less than 10;
                if np.sum(label == i) <= 10 * i:
                    continue
                cnt += 1
                self.total_images.append(img)
                self.img_emb.append(image_embedding)
                tmp_label = np.where(label == i, 1, 0)
                self.total_labels.append(tmp_label)
                self.centers.append(get_center(tmp_label))
                self.bboxes.append(get_bbox(tmp_label, self.config['box_padding']))
                self.organs.append(i)


    def get_prompt(self, idx):
        prompt_type = self.prompt_type
        prompt_choice = self.prompt_choice
        prompt_num = self.prompt_num
        labels = self.total_labels[idx]
        center = self.centers[idx]
        bbox = self.bboxes[idx]
        if prompt_type == 'single_point':
            if prompt_choice == 'random':
                prompt = random_selection(labels, center, 1, except_center=False)
            elif prompt_choice == 'center':
                prompt = center.reshape((1, 2))
            else:
                raise ValueError('Invalid prompt choice.')
        elif prompt_type == 'multipoints':
            if prompt_choice == 'random':
                prompt_1 = random_selection(labels, center, prompt_num - prompt_num // 2, except_center=False)
                prompt_2 = random_selection(labels, center, prompt_num // 2, 0, except_center=False)
                prompt = np.append(prompt_1, prompt_2, axis=0)
            elif prompt_choice == 'center':
                random_prompt = random_selection(labels, center, prompt_num - prompt_num // 2 - 1, except_center=True)
                prompt_2 = random_selection(labels, center, prompt_num // 2, 0, except_center=False)
                prompt = np.append(random_prompt, center, axis=0)
                prompt = np.append(prompt, prompt_2, axis=0)
            else:
                raise ValueError('Invalid prompt choice.')
        elif prompt_type == 'bbox':
            prompt = bbox
        elif prompt_type == 'grid_points':
            prompt = self.grid_emb[idx]
        return prompt
    
    def get_img_emb(self, img, model):
        with torch.no_grad():
            img = torch.from_numpy(img).to(model.device)
            image = self.transform.apply_image_torch(img.reshape((1, 1, img.shape[-2], img.shape[-1])).float())
            image = image.repeat(1, 3, 1, 1)
            input_image = model.preprocess(image)
            image_embedding = model.image_encoder(input_image)
        return image_embedding.cpu().numpy()
    
    def get_grid_points(self, idx):
        with torch.no_grad():
            label = self.total_labels[idx]
            img_size = label.shape[-1]
            seq_x = np.linspace(0, img_size - 1, self.config['grid_num'], dtype=int)[1:-1]
            seq_y = np.linspace(0, img_size - 1, self.config['grid_num'], dtype=int)[1:-1]
            points_0, points_1 = [], []
            for x in seq_x:
                for y in seq_y:
                    if label[y, x] == 0:
                        points_0.append([x, y])
                    else:
                        points_1.append([x, y])
            random.shuffle(points_0)
            random.shuffle(points_1)
            if len(points_1) == 0:
                points_1 = get_center(label).reshape((1, 2)).tolist()
            num_1 = min((self.prompt_num + 1) // 2, len(points_1))
            num_0 = self.prompt_num - num_1
            points_2 = points_0[:num_0] + points_1[:num_1]
            points = np.array(points_2)
            point_labels = np.array([label[y, x] for x, y in points_2])
        return (points, point_labels)

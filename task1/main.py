import os
import torch
import numpy as np
import json
import argparse
import random
import yaml
from load_data import CTDataset, random_selection
from torch.utils.data import DataLoader
import nibabel as nib
from nibabel.imageglobals import LoggingOutputSuppressor
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
import sys
from datetime import datetime
logger = logging.getLogger()


def get_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=13080 if config.get('seed') is None else config['seed'])
    parser.add_argument('--use_gpu', type=bool, default=True if config.get('use_gpu') is None else config['use_gpu'])
    parser.add_argument('--gpu_idx', type=int, default=0 if config.get('gpu_idx') is None else config['gpu_idx'])
    parser.add_argument('--log_dir', type=str, default='../../autodl-tmp/ml_logs_1/' if config.get('log_dir') is None
                        else config['log_dir'])
    parser.add_argument('--logging', type=bool, default=True if config.get('logging') is None else config['logging'])
    parser.add_argument('--exp', type=str, default='temp' if config.get('exp') is None else config['exp'])
    parser.add_argument('--model_root_path', type=str,
                        default='../../autodl-tmp/ml_model' if config.get('model_root_path') is None else
                        config['model_root_path'])
    parser.add_argument('--model_type', type=str, default=None if config.get('model_type') is None else
                        config['model_type'])
    parser.add_argument('--model_checkpoint', type=str, default=None if config.get('model_checkpoint') is None else
                        config['model_checkpoint'])
    parser.add_argument('--data_root_path', type=str,
                        default='../../autodl-tmp/ml_data/RawData' if config.get('data_root_path') is None else
                        config['data_root_path'])
    parser.add_argument('--dataset_info_path', type=str,
                        default='../../autodl-tmp/ml_data/dataset_0.json' if config.get('dataset_info_path') is None
                        else config['dataset_info_path'])
    parser.add_argument('--num_classes', type=int, default=13 if config.get('num_classes') is None else
                        config['num_classes'])
    parser.add_argument('--batch_size', type=int, default=128 if config.get('batch_size') is None else
                        config['batch_size'])
    parser.add_argument('--num_workers', type=int, default=8 if config.get('num_workers') is None else
                        config['num_workers'])
    parser.add_argument('--prompt_type', type=str, default='single_point' if config.get('prompt_type') is None else
                        config['prompt_type'])  # 'single_point', 'multipoints', 'bbox', 'multiple'
    parser.add_argument('--prompt_choice', type=str, default=None if config.get('prompt_choice') is None else
                        config['prompt_choice'])  # 'random' or 'center' for point
    parser.add_argument('--prompt_num', type=int, default=5 if config.get('prompt_num') is None else
                        config['prompt_num'])  # for multipoints

    config, _ = parser.parse_known_args()
    print(config)
    return config


def get_time():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def prepare_dirs_loggers(config, script=""):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # if config.forward_only:
    #     return
    log_dir = config.log_dir + '/' + config.exp
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    config.time_stamp = get_time()
    config.script = script
    dir_name = "{}-{}".format(config.time_stamp, script) if script else config.time_stamp
    config.session_dir = os.path.join(log_dir, dir_name)
    os.mkdir(config.session_dir)

    fileHandler = logging.FileHandler(os.path.join(config.session_dir,
                                                   'session.log'))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # save config
    param_path = os.path.join(config.session_dir, "params.json")
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def get_dataset(config):
    valid_set = CTDataset(config, 'valid')
    return {'valid': valid_set}


def get_dataloader(config, dataset):

    valid_loader = DataLoader(dataset['valid'], batch_size=1, shuffle=False, num_workers=config.num_workers)
    return valid_loader, valid_loader, valid_loader


def show_mask(mask, ax, random_color=False):
    if random_color:
        # red color
        color = np.array([1, 0, 0, 0.6])
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def sam_predict(config, data_loader, mode='valid'):
    sam = sam_model_registry[config.model_type](checkpoint=os.path.join(config.model_root_path, config.model_checkpoint))
    prompt_type = config.prompt_type
    total_dice = np.zeros(1 + config.num_classes, dtype=np.float32)
    total_idx = np.zeros(1 + config.num_classes, dtype=np.int16)
    if config.logging:
        logger.info('Start {} prediction.'.format(mode))
    batch_cnt = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images, labels, centers, bboxes, organs = batch
            if prompt_type == 'single_point':
                if config.prompt_choice == 'random':
                    prompt = random_selection(labels, centers, 1, except_center=False)
                elif config.prompt_choice == 'center':
                    prompt = centers
                else:
                    raise ValueError('Invalid prompt choice.')
            elif prompt_type == 'multipoints':
                if config.prompt_choice == 'random':
                    prompt = random_selection(labels[0], centers, config.prompt_num, except_center=False)
                elif config.prompt_choice == 'center':
                    random_prompt = random_selection(labels[0], centers, config.prompt_num - 1, except_center=True)
                    prompt = [-1 if not organs[0][idx] else np.append(random_prompt[idx], centers[idx], axis=0)
                              for idx in range(config.num_classes + 1)]
                else:
                    raise ValueError('Invalid prompt choice.')
            elif prompt_type == 'bbox':
                prompt = bboxes
            tmp_dice = test(config, sam, (images, prompt), labels, organs)
            total_dice += tmp_dice
            total_idx += np.sum(organs.numpy().astype(np.int16), axis=0)
            if config.logging:
                logger.info('Batch {} dice: {}'.format(batch_cnt, tmp_dice))
            batch_cnt += 1
        dice_scores = total_dice / total_idx
    return dice_scores


def dice(pred, true, true_sum=None):
    if true_sum is None:
        true_sum = true.sum()
    _dice = 2 * ((pred * true).astype(np.int16).sum()) / (pred.sum() + true_sum)
    return _dice


def test(config, model, data, labels, organs, batched=False, visualize=True):
    """
        For batched or single images, labels and prompts, test the model.
    """
    prompt_type = config.prompt_type
    dice_scores = np.zeros((1 + config.num_classes), dtype=np.float32)
    if not batched:
        image, prompt = data
        image = np.repeat(image, 3, axis=0)
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        predictor = SamPredictor(model)
        predictor.set_image(image)
        cls_labels = [-1]
        for cls in range(1, 1 + config.num_classes):
            if not organs[0][cls]:
                cls_labels.append(-1)
                continue
            tmp_labels = np.zeros_like(labels)
            tmp_labels[labels == cls] = 1
            cls_labels.append(tmp_labels)
        if prompt_type == 'single_point':
            for cls in range(1 + config.num_classes):
                if not organs[0][cls]:
                    continue
                p_coord = np.array(prompt[cls])
                p_label = np.array([1])
                pred, _, _ = predictor.predict(point_coords=p_coord, point_labels=p_label, multimask_output=False)
                show_image = data[0].numpy().reshape((data[0].shape[1], data[0].shape[2], data[0].shape[0]))
                plt.figure(figsize=(20, 20))
                plt.imshow(show_image)
                show_mask(pred[0], plt.gca())
                show_mask(cls_labels[cls], plt.gca(), random_color=True)
                show_points(p_coord, p_label, plt.gca())
                plt.axis('off')
                plt.show()
                dice_scores[cls] = dice(pred, cls_labels[cls])
        elif prompt_type == 'multipoints':
            for cls in range(1 + config.num_classes):
                if not organs[0][cls]:
                    continue
                p_coord = np.array(prompt[cls])
                p_label = np.array([1] * prompt[cls].shape[0])

                pred, _, _ = predictor.predict(point_coords=p_coord, point_labels=p_label, multimask_output=False)

                dice_scores[cls] = dice(pred, cls_labels[cls])
        elif prompt_type == 'bbox':
            for cls in range(1 + config.num_classes):
                if not organs[0][cls]:
                    continue
                p_box = np.array(prompt[cls])
                pred, _, _ = predictor.predict(box=p_box, multimask_output=False)
                # show_image = data[0].numpy().reshape((data[0].shape[1], data[0].shape[2], data[0].shape[0]))
                # plt.figure(figsize=(20, 20))
                # plt.imshow(show_image)
                # show_mask(pred[0], plt.gca())
                # show_mask(cls_labels[cls], plt.gca(), random_color=True)
                # show_box(p_box[0], plt.gca())
                # plt.axis('off')
                # plt.show()
                dice_scores[cls] = dice(pred, cls_labels[cls])
    return dice_scores


if __name__ == '__main__':
    config = get_config('cfg.yaml')
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_gpu:
        torch.cuda.manual_seed(config.seed)
        torch.cuda.set_device(config.gpu_idx)
    if config.logging:
        prepare_dirs_loggers(config, os.path.basename(__file__).split('.')[0])
    with LoggingOutputSuppressor():
        dataset = get_dataset(config)
        train_loader, valid_loader, test_loader = get_dataloader(config, dataset)
    dice_scores = sam_predict(config, valid_loader, mode='valid')
    if config.logging:
        logger.info('Final dice scores: {}'.format(dice_scores))
    print('Finished.')

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_mask(mask, ax, random_color=False):
    if random_color:
        # color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
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


colors = [np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([0, 0, 255]), np.array([255, 255, 0])]


def draw_point(img, coord, color, size):
    x, y = coord
    img[y - size:y + size + 1, x - size:x + size + 1] = color
    return img


def visual(img, mask=None, coords=None, labels=None, box=None, random_color=True, path=None, size=0):
    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    img = img.astype(np.float32)
    plt.imshow(img, cmap='gray')
    plt.imsave(path, img, cmap='gray')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    if mask is not None:
        print('mask:', mask.shape)
        img[mask == 1] = np.array([255, 0, 0])
    if box is not None:
        x1, y1, x2, y2 = box
        for i in range(x1, x2 + 1):
            draw_point(img, (i, y1), colors[2], size)
            draw_point(img, (i, y2), colors[2], size)
        for i in range(y1, y2 + 1):
            draw_point(img, (x1, i), colors[2], size)
            draw_point(img, (x2, i), colors[2], size)
    if coords is not None and labels is not None:
        print('coords:', coords.shape)
        print('labels:', labels.shape)
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        for coord in pos_points:
            img = draw_point(img, coord, colors[2], size)
        for coord in neg_points:
            img = draw_point(img, coord, colors[0], size)
    plt.axis('off')
    print('img:', img.shape)
    if path is not None:
        new_path = path[:-4] + '_visualize.png'
        plt.imsave(new_path, img)
    else:
        plt.show()


def visual_4(img, mask=None, coords=None, labels=None, box=None, path0=None, size=0):
    colors = [np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([0, 0, 255]), np.array([255, 255, 0]), np.array([0, 255, 255])]
    # save_path = os.path.join(self.config['log_dir'], self.config['exp'], self.config['time_stamp'] + '-main', 'tmp.png')
    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    img = img.astype(np.float32)
    plt.imshow(img, cmap='gray')
    path = path0 + '/raw_img.png'
    plt.imsave(path, img, cmap='gray')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    for i in range(len(mask)):
        img[mask[i] == 1] = colors[i + 1]
    plt.imshow(img)
    path1 = path0 + '/all_mask_img.png'
    plt.imsave(path1, img)
    for i in range(len(mask)):
        img = cv2.imread(path)
        img[mask[i] == 1] = colors[i + 1]
        plt.imshow(img)
        path2 = path0 + '/mask_img_' + str(i) + '.png'
        plt.imsave(path2, img)
    img = cv2.imread(path)
    for j in range(len(mask)):
        x1, y1, x2, y2 = box[j]
        for i in range(x1, x2 + 1):
            draw_point(img, (i, y1), colors[j + 1], size)
            draw_point(img, (i, y2), colors[j + 1], size)
        for i in range(y1, y2 + 1):
            draw_point(img, (x1, i), colors[j + 1], size)
            draw_point(img, (x2, i), colors[j + 1], size)
    plt.imshow(img)
    path3 = path0 + '/box_img.png'
    plt.imsave(path3, img)
    img = cv2.imread(path)
    coords_tmp = coords[0]
    labels_tmp = labels[0]
    for i in range(len(mask)):
        coords_tmp[i] = coords_tmp[i].reshape(-1, 2)
    for j in range(len(mask)):
        pos_points = coords_tmp[j][labels_tmp[j] == 1]
        neg_points = coords_tmp[j][labels_tmp[j] == 0]
        for coord in pos_points:
            img = draw_point(img, coord, colors[j + 1], size)
        for coord in neg_points:
            img = draw_point(img, coord, colors[0], size)
    plt.imshow(img)
    path4 = path0 + '/single_point_img.png'
    plt.imsave(path4, img)
    img = cv2.imread(path)
    coords_tmp = coords[1]
    labels_tmp = labels[1]
    for j in range(len(mask)):
        pos_points = coords_tmp[j][labels_tmp[j] == 1]
        neg_points = coords_tmp[j][labels_tmp[j] == 0]
        for coord in pos_points:
            img = draw_point(img, coord, colors[j + 1], size)
        for coord in neg_points:
            img = draw_point(img, coord, colors[0], size)
    plt.imshow(img)
    path4 = path0 + '/multi_points_img.png'
    plt.imsave(path4, img)
    img = cv2.imread(path)
    coords_tmp = coords[2]
    labels_tmp = labels[2]
    neg_points = coords_tmp[0]
    for coord in neg_points:
        img = draw_point(img, coord, colors[0], size)
    for j in range(len(mask)):
        pos_points = coords_tmp[j][labels_tmp[j] == 1]
        for coord in pos_points:
            img = draw_point(img, coord, colors[j + 1], size)
    plt.imshow(img)
    path4 = path0 + '/grid_points_img.png'
    plt.imsave(path4, img)
    exit()
    # if box is not None:
    #     y1, x1, y2, x2 = box
    #     for i in range(x1, x2 + 1):
    #         draw_point(img, (i, y1), colors[2], size)
    #         draw_point(img, (i, y2), colors[2], size)
    #     for i in range(y1, y2 + 1):
    #         draw_point(img, (x1, i), colors[2], size)
    #         draw_point(img, (x2, i), colors[2], size)
    # if coords is not None and labels is not None:
    #     print('coords:', coords.shape)
    #     print('labels:', labels.shape)
    #     pos_points = coords[labels == 1]
    #     neg_points = coords[labels == 0]
    #     for coord in pos_points:
    #         img = draw_point(img, coord, colors[2], size)
    #     for coord in neg_points:
    #         img = draw_point(img, coord, colors[0], size)
    # plt.axis('off')
    # print('img:', img.shape)
    # if path is not None:
    #     new_path = path[:-4] + '_visualize.png'
    #     plt.imsave(new_path, img)
    # else:
    #     plt.show()

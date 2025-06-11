import cv2
import random

import matplotlib

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from random import choice
from scipy import ndimage
from matplotlib.patches import Rectangle
from skimage.util import random_noise
from torch.nn.functional import one_hot
import sys
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset

mnist_dir = '../data/mnist/MNIST/'
transform = transforms.Compose( [transforms.ToTensor()])

def make_dictionary():
    # indexes from 0 to 49
    # labels from 0 to 99
    dict_labels = {}
    index = 0
    for label in range(100):
        label_str = str(label)
        if len(label_str) == 1:
            label_str = '0' + label_str
        if not (int(label_str), int(label_str[::-1])) in dict_labels.values() and not (int(label_str[::-1]), int(
                label_str)) in dict_labels.values():
            dict_labels[index] = (int(label_str), int(label_str[::-1]))
            index += 1

    return dict_labels


dict_labels = {0: (0, 0), 1: (1, 10), 2: (2, 20), 3: (3, 30), 4: (4, 40), 5: (5, 50), 6: (6, 60), 7: (7, 70),
               8: (8, 80), 9: (9, 90), 10: (11, 11), 11: (12, 21), 12: (13, 31), 13: (14, 41), 14: (15, 51),
               15: (16, 61), 16: (17, 71), 17: (18, 81), 18: (19, 91), 19: (22, 22), 20: (23, 32), 21: (24, 42),
               22: (25, 52), 23: (26, 62), 24: (27, 72), 25: (28, 82), 26: (29, 92), 27: (33, 33), 28: (34, 43),
               29: (35, 53), 30: (36, 63), 31: (37, 73), 32: (38, 83), 33: (39, 93), 34: (44, 44), 35: (45, 54),
               36: (46, 64), 37: (47, 74), 38: (48, 84), 39: (49, 94), 40: (55, 55), 41: (56, 65), 42: (57, 75),
               43: (58, 85), 44: (59, 95), 45: (66, 66), 46: (67, 76), 47: (68, 86), 48: (69, 96), 49: (77, 77),
               50: (78, 87), 51: (79, 97), 52: (88, 88), 53: (89, 98), 54: (99, 99)}

for i in range(55):
    dict_labels[i] = (torch.tensor(dict_labels[i][0]), torch.tensor(dict_labels[i][1]))


def recursive_mkdir(save_dir):
    # recursive mkdir
    for i in range(len(save_dir.split('/')) - 1):
        if not os.path.exists('./' + '/'.join(save_dir.split('/')[1:i + 2])):
            os.mkdir('./' + '/'.join(save_dir.split('/')[1:i + 2]))


def compute_bounding_box(image, plot=False):
    image = torch.where(image == 0, 0, 1)

    # Find the location of all objects
    objs = ndimage.find_objects(image)

    height = int(objs[0][0].stop - objs[0][0].start)
    width = int(objs[0][1].stop - objs[0][1].start)

    if plot:
        plt.imshow(image, cmap='Greys')

        plt.gca().add_patch(Rectangle((objs[0][1].start - 1, objs[0][0].start - 1), width + 1, height + 1,
                                      edgecolor='red',
                                      facecolor='none',
                                      lw=4))
        plt.show()

    return (objs[0][0].start, objs[0][1].start), (objs[0][0].stop, objs[0][1].stop)


def pick_number(idx):
    d = mnist_imgs[idx]
    d_label = mnist_labels[idx]

    start_d, end_d = compute_bounding_box(d)

    return d, d_label, start_d, end_d


def place_d2(idx_d1, new_image, start_d1, end_d1, d2, start_d2, end_d2, start_d2_mask, end_d2_mask):
    position_d2 = (choice(positions), choice(positions))
    n_attempts = 0
    while (position_d2[0] + (end_d2[0] - start_d2[0]) > im_size or position_d2[1] + (
            end_d2[1] - start_d2[1]) > im_size):
        position_d2 = (choice(positions), choice(positions))
        n_attempts += 1
        if n_attempts == 100:
            print("max number of attempts reached to find a position, changing number. Also consider inscreasing the "
                  "size of the image")
            idx_d2 = choice(instances)

            while mnist_labels[idx_d2] == mnist_labels[idx_d1]:
                idx_d2 = choice(instances)
            d2, d2_label, start_d2, end_d2 = pick_number(idx_d2)
            start_d2_mask, end_d2_mask = start_d2, end_d2
            n_attempts = 0

    temp_image = new_image.clone()

    temp_image[position_d2[0]:position_d2[0] + (end_d2[0] - start_d2[0]),
    position_d2[1]:position_d2[1] + (end_d2[1] - start_d2[1])] += d2[start_d2[0]:end_d2[0], start_d2[1]:end_d2[1]]

    temp_image = torch.where(temp_image > 250, 255., temp_image.to(torch.double))

    end_d2 = position_d2[0] + (end_d2[0] - start_d2[0]), position_d2[1] + (end_d2[1] - start_d2[1])
    start_d2 = position_d2[0], position_d2[1]

    overlap_x = max(start_d1[0], start_d2[0]) - min(end_d1[0], end_d2[0])
    overlap_y = max(start_d1[1], start_d2[1]) - min(end_d1[1], end_d2[1])

    position_x = min(end_d1[0], end_d2[0])
    position_y = min(end_d1[1], end_d2[1])

    amount_overlap = overlap_x * overlap_y

    active_pixels = (end_d1[0] - start_d1[0]) * (end_d1[1] - start_d1[1]) + (end_d1[0] - start_d1[0]) * (
            end_d1[1] - start_d1[1])

    overlap = amount_overlap / active_pixels

    return temp_image, position_x, position_y, overlap_x, overlap_y, overlap, position_d2, start_d2, end_d2, start_d2_mask, end_d2_mask


def generate_masks(new_image, d1, d2, start_d1, start_d2, end_d1, end_d2, position_d2, start_point):
    mask = torch.zeros_like(new_image)

    mask_d1 = torch.where(d1[start_d1[0]:end_d1[0], start_d1[1]:end_d1[1]] > 0, 1, 0)
    mask_d2 = torch.where(d2[start_d2[0]:end_d2[0], start_d2[1]:end_d2[1]] > 0, 2, 0)

    mask[start_point[0]:start_point[0] + (end_d1[0] - start_d1[0]),
         start_point[1]:start_point[1] + (end_d1[1] - start_d1[1])] = mask_d1

    mask[position_d2[0]:position_d2[0] + (end_d2[0] - start_d2[0]),
         position_d2[1]:position_d2[1] + (end_d2[1] - start_d2[1])] += mask_d2

    return mask


def multi_mnist(save_dir, num_images=60000, kind='training', max_overlap=.1, plot=False, add_distractor=False, nb_objects=2):
    recursive_mkdir(save_dir)

    data = []
    labels = []
    masks = []

    list_lab = torch.zeros(10)

    for image in range(num_images):
        idx_d1 = choice(instances)

        d1, d1_label, start_d1, end_d1 = pick_number(idx_d1)
        start_d1_mask, end_d1_mask = start_d1, end_d1
        start_point = (choice([i for i in range(im_size - 28)]), choice([i for i in range(im_size - 28)]))

        new_image = torch.zeros(im_size, im_size)
        new_image[start_point[0]:start_point[0] + (end_d1[0] - start_d1[0]),
        start_point[1]:start_point[1] + (end_d1[1] - start_d1[1])] = d1[start_d1[0]:end_d1[0], start_d1[1]:end_d1[1]]

        end_d1 = start_point[0] + (end_d1[0] - start_d1[0]), start_point[1] + (end_d1[1] - start_d1[1])
        start_d1 = start_point[0], start_point[1]

        idx_d2 = choice(instances)

        while mnist_labels[idx_d2] == d1_label:
            idx_d2 = choice(instances)

        d2, d2_label, start_d2, end_d2 = pick_number(idx_d2)
        start_d2_mask, end_d2_mask = start_d2, end_d2

        temp_image, position_x, position_y, overlap_x, overlap_y, overlap, temp_position_d2, temp_start_d2, temp_end_d2, temp_start_d2_mask, temp_end_d2_mask = place_d2(
            idx_d1, new_image, start_d1, end_d1, d2, start_d2, end_d2, start_d2_mask, end_d2_mask)

        n_attempts_overlap = 0
        while overlap > max_overlap:
            temp_image, position_x, position_y, overlap_x, overlap_y, overlap, temp_position_d2, temp_start_d2, temp_end_d2, temp_start_d2_mask, temp_end_d2_mask = place_d2(
                idx_d1, new_image, start_d1, end_d1, d2, start_d2, end_d2, start_d2_mask, end_d2_mask)
            n_attempts_overlap += 1
            if n_attempts_overlap == 100:
                print(
                    "max number of attempts reached to fit overlap constraint, changing number. Also consider "
                    "inscreasing the size of the image or reduce the max amount of overlap")
                idx_d2 = choice(instances)

                while mnist_labels[idx_d2] == d1_label:
                    idx_d2 = choice(instances)
                d2, d2_label, start_d2, end_d2 = pick_number(idx_d2)
                start_d2_mask, end_d2_mask = start_d2, end_d2
                n_attempts_overlap = 0

        new_image = temp_image
        start_d2, end_d2 = temp_start_d2, temp_end_d2
        start_d2_mask, end_d2_mask = temp_start_d2_mask, temp_end_d2_mask
        position_d2 = temp_position_d2

        positions_x = np.arange(0, stop=im_size-8)
        positions_y = np.arange(0,stop=im_size-6)

        if add_distractor:
            x, y = choice(positions_x), choice(positions_y)
            cv2.rectangle(np.asarray(new_image), pt1=(x+10, y+8), pt2=(x, y), color=(255, 0, 0), thickness=2)


        mask = generate_masks(new_image, d1, d2, start_d1_mask, start_d2_mask, end_d1_mask, end_d2_mask, position_d2, start_point)
        new_image = torch.nn.functional.interpolate(new_image.unsqueeze(0).unsqueeze(0),
                                                    scale_factor=32 / im_size,
                                                    mode='bilinear').squeeze()


        if plot:
            plt.imshow(new_image.permute(1,2,0))
            plt.gca().add_patch(Rectangle((start_d1[1], start_d1[0]), end_d1[1] - start_d1[1], end_d1[0] - start_d1[0],
                                          edgecolor='blue',
                                          facecolor='none',
                                          lw=4))
            plt.gca().add_patch(Rectangle((start_d2[1], start_d2[0]), end_d2[1] - start_d2[1], end_d2[0] - start_d2[0],
                                          edgecolor='red',
                                          facecolor='none',
                                          lw=4))
            plt.gca().add_patch(Rectangle((position_y, position_x), overlap_y, overlap_x,
                                          edgecolor='green',
                                          facecolor='none',
                                          lw=4))
            plt.show()
            plt.close()

        data.append(new_image)
        labels.append(torch.tensor((d1_label, d2_label)).unsqueeze(0))
        list_lab[d1_label] += 1
        list_lab[d2_label] += 1
        masks.append(mask)

    print(list_lab)
    data = torch.stack(data)
    labels = torch.stack(labels)
    masks = torch.stack(masks)
    torch.save((data, labels), os.path.join(save_dir, 'processed.pt'))
    torch.save(masks, os.path.join(save_dir, 'masks.pt'))

def k_hot_encode(array, max_val):
    """k-hot encodes sparse vector of targets
    Array should be N x k for N samples, k targets for each sample"""
    if len(array.shape) == 1:
        array = array.long()
        array = one_hot(array, num_classes=10)
        return array.float()
    else:
        b = np.zeros((array.shape[0], max_val + 1))

        for col in range(array.shape[1]):
            b[np.arange(array.shape[0]), array[:, col]] = 1
        return b


def read_data():
    data, label = torch.load(os.path.join(save_dir, 'processed.pt'))
    masks = torch.load(os.path.join(save_dir, 'masks.pt'))
    print(masks.shape)

    for i in range(len(data)):
        inputs = data[i]
        print(inputs.shape)
        plt.imshow((inputs.numpy()))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.close()

        plt.imshow(masks[i])
        plt.colorbar()
        plt.show()
        plt.close()

        y = torch.Tensor(k_hot_encode(label[i], max_val=9))
        y = label[i]
        print(y)


if __name__ == '__main__':
    """random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)"""
    # TODO: CURRENTLY ONLY WORKS WITH 2 DIGITS!!!

    kind = sys.argv[1] #"training" #test
    mnist_imgs, mnist_labels = torch.load(os.path.join(mnist_dir, 'processed/{}.pt'.format(kind)))
    num_mnist = mnist_imgs.shape[0]
    im_size = int(mnist_imgs.shape[1] * 1.8)
    print(im_size)

    instances = [i for i in range(num_mnist)]
    positions = [i for i in range(im_size-20)]

    save_dir = './data/multi_mnist_cifar2/multi_mnist/{}'.format(kind)  # ,max_overlap,num_digits+1

    multi_mnist(save_dir, num_images=int(sys.argv[2]), kind=kind, max_overlap=0, plot=False, add_distractor=False)
    read_data()


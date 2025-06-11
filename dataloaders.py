import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
import torchvision as torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import pickle as pkl

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from opts import parser
args = parser.parse_args()


def Dataloader(path, batch_size, test_batch_size):
    data = torch.load(path + "_images.pt")#[:500]
    masks = torch.load(path + "_masks.pt")#[:500]

    shapes = torch.load(path + "_shapes.pt")
    dataset = TensorDataset(data, shapes, masks)

    train_set, train_eval_set = torch.utils.data.random_split(dataset, [int(0.7 * len(dataset)),
                                                                        int(0.3 * len(dataset))])

    train_eval_set, test_set = torch.utils.data.random_split(train_eval_set, [int(2 / 3 * len(train_eval_set)),
                                                                              int(1 / 3 * len(train_eval_set))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(train_eval_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

class MaskedDataset(Dataset):
    def __init__(self, images, labels, masks):
        super().__init__()
        self.data = images
        self.labels = labels
        self.masks = masks

    def __getitem__(self, i):
        return self.data[i], self.labels[i], self.masks[i]

    def __len__(self):
        return len(self.data)

def expand_mask(masks):
    masks_test = masks.clone()
    for im in range(masks.shape[0]):
        for i in range(1, 31):
            for j in range(1, 31):
                if (masks[im, i, j] == 1):
                    if masks[im, i + 1, j + 1] == 0:
                        masks_test[im, i + 1, j + 1] = 1
                    if masks[im, i - 1, j + 1] == 0:
                        masks_test[im, i - 1, j + 1] = 1
                    if masks[im, i - 1, j - 1] == 0:
                        masks_test[im, i - 1, j - 1] = 1
                    if masks[im, i + 1, j - 1] == 0:
                        masks_test[im, i + 1, j - 1] = 1
    for im in range(masks.shape[0]):
        for i in range(1, 31):
            for j in range(1, 31):
                if (masks[im, i, j] == 2):
                    if masks[im, i + 1, j + 1] == 0:
                        masks_test[im, i + 1, j + 1] = 2
                    if masks[im, i - 1, j + 1] == 0:
                        masks_test[im, i - 1, j + 1] = 2
                    if masks[im, i - 1, j - 1] == 0:
                        masks_test[im, i - 1, j - 1] = 2
                    if masks[im, i + 1, j - 1] == 0:
                        masks_test[im, i + 1, j - 1] = 2

    return masks_test

def MultiMNISTLoader(path):
    if args.in_repo == '23objects_0':
        train_set_2 = utils.load_dataset_v2("../../data/{}/2objects_0/".format(args.dataset), train=True)
        test_set_2 = utils.load_dataset_v2("../../data/{}/2objects_0/".format(args.dataset), train=False)
        train_loader_2, train_eval_loader_2 = torch.utils.data.random_split(train_set_2, [int(0.8 * len(train_set_2)),
                                                                                        int(0.2 * len(train_set_2))])
        train_set_3 = utils.load_dataset_v2("../../data/{}/3objects_0/".format(args.dataset), train=True)
        test_set_3 = utils.load_dataset_v2("../../data/{}/3objects_0/".format(args.dataset), train=False)
        train_loader_3, train_eval_loader_3 = torch.utils.data.random_split(train_set_3, [int(0.8 * len(train_set_3)),
                                                                                      int(0.2 * len(train_set_3))])
        train_loader = torch.utils.data.ConcatDataset([train_loader_2, train_loader_3])
        train_eval_loader = torch.utils.data.ConcatDataset([train_eval_loader_2, train_eval_loader_3])
        test_loader = torch.utils.data.ConcatDataset([test_set_2, test_set_3])
        train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(train_eval_loader, batch_size=args.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_loader, batch_size=args.test_batch_size, shuffle=False, drop_last=True)
    else:
        train_set = utils.load_dataset_v2(path, train=True)
        test_set = utils.load_dataset_v2(path, train=False)
        train_loader, train_eval_loader = torch.utils.data.random_split(train_set, [int(0.8 * len(train_set)),
                                                                                    int(0.2 * len(train_set))])
        train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(train_eval_loader, batch_size=args.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

def split_train_val(train_X, train_Y, val_fraction, train_fraction=None):
    """
    Input: training data as a torch.Tensor
    """
    # Shuffle
    idx = np.arange(train_X.shape[0])
    np.random.shuffle(idx)
    train_X = train_X[idx,:]
    train_Y = train_Y[idx,:]

    # Compute validation set size
    val_size = int(val_fraction*train_X.shape[0])

    # Downsample for sample complexity experiments
    if train_fraction is not None:
        train_size = int(train_fraction*train_X.shape[0])
        assert val_size + train_size <= train_X.shape[0]
    else:
        train_size = train_X.shape[0] - val_size

    # Shuffle X
    idx = np.arange(0, train_X.shape[0])
    np.random.shuffle(idx)

    train_idx = idx[0:train_size]
    val_idx = idx[-val_size:]
    val_X = train_X[val_idx, :]
    val_Y = train_Y[val_idx, :]
    train_X = train_X[train_idx, :]
    train_Y = train_Y[train_idx, :]


    return train_X, train_Y, val_X, val_Y

import numpy as np
import torch
from torch.utils.data import Sampler
from torchvision.datasets import SVHN, CIFAR10, CIFAR100, STL10, MNIST
import torchvision.transforms as tv_transforms
import os
from skimage.transform import resize


class InfiniteSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        epochs = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(epochs)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SequentialSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        epochs = num_sample // num_data + 1
        self.indices = torch.cat([torch.range(0, num_data-1).type(torch.int) for _ in range(epochs)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_svhn(root):
    train_data = SVHN(root, "train", download=True)
    test_data = SVHN(root, "test", download=True)
    train_data = {"images": np.transpose(train_data.data.astype(np.float32), (0, 2, 3, 1)),
                  "labels": train_data.labels.astype(np.int32)}
    test_data = {"images": np.transpose(test_data.data.astype(np.float32), (0, 2, 3, 1)),
                 "labels": test_data.labels.astype(np.int32)}
    return train_data, test_data


def get_cifar10(root):
    train_data = CIFAR10(root, download=True)
    test_data = CIFAR10(root, False)
    train_data = {"images": train_data.data.astype(np.float32),
                  "labels": np.asarray(train_data.targets).astype(np.int32)}
    test_data = {"images": test_data.data.astype(np.float32), 
                 "labels": np.asarray(test_data.targets).astype(np.int32)}
    return train_data, test_data


def get_cifar100(root):
    train_data = CIFAR100(root, download=True)
    test_data = CIFAR100(root, False)
    train_data = {"images": train_data.data.astype(np.float32),
                  "labels": np.asarray(train_data.targets).astype(np.int32)}
    test_data = {"images": test_data.data.astype(np.float32),
                 "labels": np.asarray(test_data.targets).astype(np.int32)}
    return train_data, test_data


def load_mnist(root):
    splits = {}
    #trans = tv_transforms.Compose([tv_transforms.ToPILImage(),tv_transforms.ToTensor(), tv_transforms.Normalize((0.5,), (1.0,))])
    for train in [True, False]:
        dataset = MNIST(root, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits['train' if train else 'test'] = data
    return splits.values()


def get_mnistOOD(root, ood_ratio=0.5):
    rng = np.random.RandomState(seed=1)
    train_data, test_data = load_mnist(root)
    # permute index of training set
    indices = rng.permutation(len(train_data['images']))
    train_data['images'] = train_data['images'][indices]
    train_data['labels'] = train_data['labels'][indices]
    test_data = split_test(test_data, tot_class=6)
    train_data, ul_train_data = split_l_u(train_data, n_labels=60, n_unlabels=30000, tot_class=6, ratio=ood_ratio)
    return train_data, ul_train_data, test_data


def split_l_u(train_set, n_labels, n_unlabels, tot_class=6, ratio = 0.5):
    # NOTE: this function assume that train_set is shuffled.
    rng = np.random.RandomState(seed=1)
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // tot_class
    n_unlabels_per_cls = int(n_unlabels*(1.0-ratio)) // tot_class
    if(tot_class < len(classes)):
        n_unlabels_shift = (n_unlabels - (n_unlabels_per_cls * tot_class)) // (len(classes) - tot_class)
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
        u_labels += [c_labels[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
    for c in classes[tot_class:]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        u_images += [c_images[:n_unlabels_shift]]
        u_labels += [c_labels[:n_unlabels_shift]]

    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}

    indices = rng.permutation(len(l_train_set["images"]))
    l_train_set["images"] = l_train_set["images"][indices]
    l_train_set["labels"] = l_train_set["labels"][indices]

    indices = rng.permutation(len(u_train_set["images"]))
    u_train_set["images"] = u_train_set["images"][indices]
    u_train_set["labels"] = u_train_set["labels"][indices]
    return l_train_set, u_train_set


def split_l_u_classimb(train_set, n_labels, n_unlabels, tot_class=5, ratio = 0.5):
    # NOTE: this function assume that train_set is shuffled.
    rng = np.random.RandomState(seed=1)
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    n_labels_per_cls = int(n_labels * (ratio/(1.0 + ratio))) // tot_class
    if (tot_class < len(classes)):
        n_labels_shift = (n_labels - (n_labels_per_cls * tot_class)) // (len(classes) - tot_class)

    n_unlabels_per_cls = int(n_unlabels * (ratio/(1.0 + ratio))) // tot_class
    if(tot_class < len(classes)):
        n_unlabels_shift = (n_unlabels - (n_unlabels_per_cls * tot_class)) // (len(classes) - tot_class)
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
        u_labels += [c_labels[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
    for c in classes[tot_class:]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_shift]]
        l_labels += [c_labels[:n_labels_shift]]
        u_images += [c_images[n_labels_shift: n_labels_shift + n_unlabels_shift]]
        u_labels += [c_labels[n_labels_shift: n_labels_shift + n_unlabels_shift]]

    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}

    indices = rng.permutation(len(l_train_set["images"]))
    l_train_set["images"] = l_train_set["images"][indices]
    l_train_set["labels"] = l_train_set["labels"][indices]

    indices = rng.permutation(len(u_train_set["images"]))
    u_train_set["images"] = u_train_set["images"][indices]
    u_train_set["labels"] = u_train_set["labels"][indices]
    return l_train_set, u_train_set


def split_test(test_set, tot_class=6):
    rng = np.random.RandomState(seed=1)
    images = test_set["images"]
    labels = test_set['labels']
    classes = np.unique(labels)
    l_images = []
    l_labels = []
    for c in classes[:tot_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:]]
        l_labels += [c_labels[:]]
    test_set = {"images": np.concatenate(l_images, 0), "labels":np.concatenate(l_labels,0)}
    indices = rng.permutation(len(test_set["images"]))
    test_set["images"] = test_set["images"][indices]
    test_set["labels"] = test_set["labels"][indices]
    return test_set

# def get_mnistOOD(root, ood_ratio=0.5):
#     train_data = np.load(os.path.join(root, "mnist_100", "l_train.npy"), allow_pickle=True).item()
#     #train_data['images'] = resize(train_data['images'], (len(train_data['images']), 32, 32))
#     ul_str = "u_train_fashion_ood_" + str(ood_ratio) + ".npy"
#     ul_train_data = np.load(os.path.join(root, "mnist_100", ul_str), allow_pickle=True).item()
#     #ul_train_data['images'] = resize(ul_train_data['images'], (len(ul_train_data['images']), 32, 32))
#     test_data = np.load(os.path.join(root, "mnist_100", "test.npy"), allow_pickle=True).item()
#     #test_data['images'] = resize(test_data['images'], (len(test_data['images']), 32, 32))
#     return train_data, ul_train_data, test_data

def get_cifarClassImb(root, classimb_ratio=0.5):
    rng = np.random.RandomState(seed=1)
    train_data = CIFAR10(root, download=True)
    test_data = CIFAR10(root, False)
    train_data = {"images": train_data.data.astype(np.float32),
                  "labels": np.asarray(train_data.targets).astype(np.int32)}
    test_data = {"images": test_data.data.astype(np.float32),
                 "labels": np.asarray(test_data.targets).astype(np.int32)}

    # permute index of training set
    indices = rng.permutation(len(train_data['images']))
    train_data['images'] = train_data['images'][indices]
    train_data['labels'] = train_data['labels'][indices]
    train_data, ul_train_data = split_l_u_classimb(train_data, n_labels=2400, n_unlabels=20000, tot_class=5, ratio=classimb_ratio)
    return train_data, ul_train_data, test_data


def get_cifarOOD(root, ood_ratio=0.5):
    rng = np.random.RandomState(seed=1)
    train_data = CIFAR10(root, download=True)
    test_data = CIFAR10(root, False)
    train_data = {"images": train_data.data.astype(np.float32),
                  "labels": np.asarray(train_data.targets).astype(np.int32)}
    test_data = {"images": test_data.data.astype(np.float32),
                 "labels": np.asarray(test_data.targets).astype(np.int32)}
    # move class "plane" and "car" to label 8 and 9
    train_data['labels'] -= 2
    test_data['labels'] -= 2
    train_data['labels'][np.where(train_data['labels'] == -2)] = 8
    train_data['labels'][np.where(train_data['labels'] == -1)] = 9
    test_data['labels'][np.where(test_data['labels'] == -2)] = 8
    test_data['labels'][np.where(test_data['labels'] == -1)] = 9
    # permute index of training set
    indices = rng.permutation(len(train_data['images']))
    train_data['images'] = train_data['images'][indices]
    train_data['labels'] = train_data['labels'][indices]
    test_data = split_test(test_data, tot_class=6)
    train_data, ul_train_data = split_l_u(train_data, n_labels=2400, n_unlabels=20000, tot_class=6, ratio=ood_ratio)
    return train_data, ul_train_data, test_data


def get_stl10(root):
    train_data = STL10(root, split="train", download=True)
    ul_train_data = STL10(root, split="unlabeled")
    test_data = STL10(root, split="test")
    train_data = {"images": np.transpose(train_data.data.astype(np.float32), (0, 2, 3, 1)),
                  "labels": train_data.labels}
    ul_train_data = {"images": np.transpose(ul_train_data.data.astype(np.float32), (0, 2, 3, 1)),
                    "labels": ul_train_data.labels}
    test_data = {"images": np.transpose(test_data.data.astype(np.float32), (0, 2, 3, 1)),
                 "labels": test_data.labels}
    return train_data, ul_train_data, test_data


def dataset_split(data, num_data, num_classes, random=False):
    """split dataset into two datasets
    
    Parameters
    -----
    data: dict with keys ["images", "labels"]
        each value is numpy.array
    num_data: int
        number of dataset1
    num_classes: int
        number of classes
    random: bool
        if True, dataset1 is randomly sampled from data.
        if False, dataset1 is uniformly sampled from data,
        which means that the dataset1 contains the same number of samples per class.

    Returns
    -----
    dataset1, dataset2: the same dict as data.
        number of data in dataset1 is num_data.
        number of data in dataset1 is len(data) - num_data.
    """
    dataset1 = {"images": [], "labels": []}
    dataset2 = {"images": [], "labels": []}
    images = data["images"]
    labels = data["labels"]

    # random sampling
    if random:
        dataset1["images"] = images[:num_data]
        dataset1["labels"] = labels[:num_data]
        dataset2["images"] = images[num_data:]
        dataset2["labels"] = labels[num_data:]

    else:
        data_per_class = num_data // num_classes
        for c in range(num_classes):
            c_idx = (labels == c)
            c_imgs = images[c_idx]
            c_lbls = labels[c_idx]
            dataset1["images"].append(c_imgs[:data_per_class])
            dataset1["labels"].append(c_lbls[:data_per_class])
            dataset2["images"].append(c_imgs[data_per_class:])
            dataset2["labels"].append(c_lbls[data_per_class:])
        for k in ("images", "labels"):
            dataset1[k] = np.concatenate(dataset1[k])
            dataset2[k] = np.concatenate(dataset2[k])
    return dataset1, dataset2


def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.transpose(0, 3, 1, 2)
    images = images.reshape(n_data, channels * height * width)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

if __name__ == "__main__":
    #get_mnistOOD(root='/home/krishnateja/PycharmProjects/EfficientSSL/data', ood_ratio=0.5)
    get_cifarOOD(root='/home/krishnateja/PycharmProjects/EfficientSSL/data', ood_ratio=0.5)

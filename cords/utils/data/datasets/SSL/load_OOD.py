import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.transform import resize
import torch.nn.functional as F
import random

class transform:
    def __init__(self, flip=True, r_crop=True, g_noise=True):
        self.flip = flip
        self.r_crop = r_crop
        self.g_noise = g_noise
        print("holizontal flip : {}, random crop : {}, gaussian noise : {}".format(
            self.flip, self.r_crop, self.g_noise
        ))

    def __call__(self, x):
        if self.flip and random.random() > 0.5:
            x = x.flip(-1)
        if self.r_crop:
            h, w = x.shape[-2:]
            x = F.pad(x, [2,2,2,2], mode="reflect")
            l, t = random.randint(0, 4), random.randint(0,4)
            x = x[:,:,t:t+h,l:l+w]
        if self.g_noise:
            n = torch.randn_like(x) * 0.15
            x = n + x
        return x

class CIFAR10:
    def __init__(self, root, split="l_train"):
        self.dataset = np.load(os.path.join(root, "cifar10_class6", split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        return image, label, idx

    def __len__(self):
        return len(self.dataset["images"])


class MNIST_idx:
    def __init__(self, root, split="l_train"):
        self.dataset = np.load(os.path.join(root, "mnist_100", split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        image = (image/255. - 0.5)/0.5
        image = resize(image, (1, 32, 32))
        return image, label, idx

    def __len__(self):
        return len(self.dataset["images"])



class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """

    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def load_minst(dataset, root, ood_ratio, batch_size, iteration):
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    if dataset == 'mnist':
        transform_fn = transform(False, False, False)  # transform function (flip, crop, noise)
        l_train_dataset = MNIST_idx(root, "l_train")
        u_train_dataset = MNIST_idx(root, "u_train_fashion_ood_{}".format(ood_ratio))
        val_dataset = MNIST_idx(root, "val")
        test_dataset = MNIST_idx(root, "test")
    else:
        transform_fn = transform(True, True, True)  # transform function (flip, crop, noise)
        l_train_dataset = CIFAR10(root, "l_train")
        u_train_dataset = CIFAR10(root, "u_train_ood_{}".format(ood_ratio))
        val_dataset = CIFAR10(root, "val")
        test_dataset = CIFAR10(root, "test")

    print("labeled data : {}, unlabeled data : {}, training data : {}, OOD rario : {}".format(
        len(l_train_dataset), len(u_train_dataset), len(l_train_dataset) + len(u_train_dataset), ood_ratio))
    print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))

    l_loader = DataLoader(
        l_train_dataset, batch_size, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), iteration * batch_size)
    )
    u_loader = DataLoader(
        u_train_dataset, batch_size, drop_last=True,
        sampler=RandomSampler(len(u_train_dataset), iteration * batch_size)
    )

    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)

    for l_data, u_data in zip(l_loader, u_loader):
        iteration += 1
        l_input, target, _ = l_data
        l_input, target = l_input.to(device).float(), target.to(device).long()

        u_input, dummy_target, idx = u_data
        u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()

    return

if __name__ == '__main__':
    root = "data"
    ood_ratio = 0.5
    batch_size = 64
    iteration = 1000
    dataset = 'mnist' # or cifar10
    load_minst(dataset, root, ood_ratio, batch_size, iteration)



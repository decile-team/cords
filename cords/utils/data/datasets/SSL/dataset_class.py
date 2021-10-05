import torch


class LabeledDataset:
    """
    For labeled dataset
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image = torch.from_numpy(self.dataset["images"][idx]).float()
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1).contiguous() / 255.
        else:
            image = image.permute(0, 1).contiguous() / 255.
            x = image.shape[0]
            image = image.view(1, x, -1)
        label = int(self.dataset["labels"][idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset["images"])


class UnlabeledDataset:
    """
    For unlabeled dataset
    """
    def __init__(self, dataset, weak_augmentation=None, strong_augmentation=None):
        self.dataset = dataset
        self.weak_augmentation = weak_augmentation
        self.strong_augmentation = strong_augmentation

    def __getitem__(self, idx):
        image = torch.from_numpy(self.dataset["images"][idx]).float()
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1).contiguous() / 255.
        else:
            image = image.permute(0, 1).contiguous() / 255.
        label = int(self.dataset["labels"][idx])
        w_aug_image = self.weak_augmentation(image)
        if self.strong_augmentation is not None:
            s_aug_image = self.strong_augmentation(image)
        else:
            s_aug_image = self.weak_augmentation(image)
        return w_aug_image, s_aug_image, label

    def __len__(self):
        return len(self.dataset["images"])


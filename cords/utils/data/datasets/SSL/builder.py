import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from . import utils
from . import dataset_class
from .augmentation.builder import gen_strong_augmentation, gen_weak_augmentation
from .augmentation.augmentation_pool import numpy_batch_gcn, ZCA, GCN


def __val_labeled_unlabeled_split(cfg, train_data, test_data, num_classes, ul_data=None):
    num_validation = int(np.round(len(train_data["images"]) * cfg.dataset.val_ratio))

    np.random.seed(cfg.train_args.seed)

    permutation = np.random.permutation(len(train_data["images"]))
    train_data["images"] = train_data["images"][permutation]
    train_data["labels"] = train_data["labels"][permutation]

    val_data, train_data = utils.dataset_split(train_data, num_validation, num_classes, cfg.dataset.random_split)
    l_train_data, ul_train_data = utils.dataset_split(train_data, cfg.dataset.num_labels, num_classes)

    if ul_data is not None:
        ul_train_data["images"] = np.concatenate([ul_train_data["images"], ul_data["images"]], 0)
        ul_train_data["labels"] = np.concatenate([ul_train_data["labels"], ul_data["labels"]], 0)

    return val_data, l_train_data, ul_train_data


def __labeled_unlabeled_split(cfg, train_data, test_data, num_classes, ul_data=None):
    np.random.seed(cfg.train_args.seed)

    permutation = np.random.permutation(len(train_data["images"]))
    train_data["images"] = train_data["images"][permutation]
    train_data["labels"] = train_data["labels"][permutation]

    l_train_data, ul_train_data = utils.dataset_split(train_data, cfg.dataset.num_labels, num_classes)

    if ul_data is not None:
        ul_train_data["images"] = np.concatenate([ul_train_data["images"], ul_data["images"]], 0)
        ul_train_data["labels"] = np.concatenate([ul_train_data["labels"], ul_data["labels"]], 0)

    return l_train_data, ul_train_data


def gen_dataloader(root, dataset, validation_split, cfg, logger=None):
    """
    generate train, val, and test dataloaders

    Parameters
    --------
    root: str
        root directory
    dataset: str
        dataset name, ['cifar10', 'cifar100', 'svhn', 'stl10']
    validation_split: bool
        if True, return validation loader.
        validation data is made from training data
    cfg: argparse.Namespace or something
    logger: logging.Logger
    """
    ul_train_data = None
    if dataset == "svhn":
        train_data, test_data = utils.get_svhn(root)
        num_classes = 10
        img_size = 32
    elif dataset == "stl10":
        train_data, ul_train_data, test_data = utils.get_stl10(root)
        num_classes = 10
        img_size = 96
    elif dataset == "cifar10":
        train_data, test_data = utils.get_cifar10(root)
        num_classes = 10
        img_size = 32
    elif dataset == "cifar100":
        train_data, test_data = utils.get_cifar100(root)
        num_classes = 100
        img_size = 32
    elif dataset == "cifarOOD":
        train_data, test_data = utils.get_cifarOOD(root, cfg.dataset.ood_ratio)
        num_classes = 6
        img_size = 32
    else:
        raise NotImplementedError

    if validation_split:
        val_data, l_train_data, ul_train_data = __val_labeled_unlabeled_split(
            cfg, train_data, test_data, num_classes, ul_train_data)
    else:
        l_train_data, ul_train_data = __labeled_unlabeled_split(
            cfg, train_data, test_data, num_classes, ul_train_data)
        val_data = None

    ul_train_data["images"] = np.concatenate([ul_train_data["images"], l_train_data["images"]], 0)
    ul_train_data["labels"] = np.concatenate([ul_train_data["labels"], l_train_data["labels"]], 0)

    if logger is not None:
        logger.info("number of :\n \
            training data: %d\n \
            labeled data: %d\n \
            unlabeled data: %d\n \
            validation data: %d\n \
            test data: %d",
            len(train_data["images"]),
            len(l_train_data["images"]),
            len(ul_train_data["images"]),
            0 if val_data is None else len(val_data["images"]),
            len(test_data["images"]))

    labeled_train_data = dataset_class.LabeledDataset(l_train_data)
    unlabeled_train_data = dataset_class.UnlabeledDataset(ul_train_data)

    train_data = np.concatenate([
        labeled_train_data.dataset["images"],
        unlabeled_train_data.dataset["images"]
        ], 0)

    if cfg.dataset.whiten:
        mean = train_data.mean((0, 1, 2)) / 255.
        scale = train_data.std((0, 1, 2)) / 255.
    elif cfg.dataset.zca:
        mean, scale = utils.get_zca_normalization_param(numpy_batch_gcn(train_data))
    else:
        # from [0, 1] to [-1, 1]
        mean = [0.5, 0.5, 0.5]
        scale = [0.5, 0.5, 0.5]

    # set augmentation
    # RA: RandAugment, WA: Weak Augmentation
    randauglist = "fixmatch" if cfg.ssl_args.alg == "pl" else "uda"

    flags = [True if b == "t" else False for b in cfg.dataset.wa.split(".")]

    if cfg.dataset.labeled_aug == "RA":
        labeled_augmentation = gen_strong_augmentation(
            img_size, mean, scale, flags[0], flags[1], randauglist, cfg.dataset.zca)
    elif cfg.dataset.labeled_aug == "WA":
        labeled_augmentation = gen_weak_augmentation(img_size, mean, scale, *flags, cfg.dataset.zca)
    else:
        raise NotImplementedError

    labeled_train_data.transform = labeled_augmentation

    if cfg.dataset.unlabeled_aug == "RA":
        unlabeled_augmentation = gen_strong_augmentation(
            img_size, mean, scale, flags[0], flags[1], randauglist, cfg.dataset.zca)
    elif cfg.dataset.unlabeled_aug == "WA":
        unlabeled_augmentation = gen_weak_augmentation(img_size, mean, scale, *flags, cfg.dataset.zca)
    else:
        raise NotImplementedError

    if logger is not None:
        logger.info("labeled augmentation")
        logger.info(labeled_augmentation)
        logger.info("unlabeled augmentation")
        logger.info(unlabeled_augmentation)

    unlabeled_train_data.weak_augmentation = unlabeled_augmentation

    if cfg.dataset.strong_aug:
        strong_augmentation = gen_strong_augmentation(
            img_size, mean, scale, flags[0], flags[1], randauglist, cfg.dataset.zca)
        unlabeled_train_data.strong_augmentation = strong_augmentation
        if logger is not None:
            logger.info(strong_augmentation)

    if cfg.dataset.zca:
        test_transform = transforms.Compose([GCN(), ZCA(mean, scale)])
    else:
        test_transform = transforms.Compose([transforms.Normalize(mean, scale, True)])

    test_data = dataset_class.LabeledDataset(test_data, test_transform)

    l_train_loader = DataLoader(
        labeled_train_data,
        cfg.dataset.l_batch_size,
        sampler=utils.InfiniteSampler(len(labeled_train_data), cfg.train_args.iteration * cfg.dataloader.l_batch_size),
        num_workers=cfg.dataloader.num_workers
    )
    ul_train_loader = DataLoader(
        unlabeled_train_data,
        cfg.dataloader.ul_batch_size,
        sampler=utils.InfiniteSampler(len(unlabeled_train_data), cfg.train_args.iteration * cfg.dataloader.ul_batch_size),
        num_workers=cfg.dataloader.num_workers
    )
    test_loader = DataLoader(
        test_data,
        1,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.dataloader.num_workers
    )

    if validation_split:
        validation_data = dataset_class.LabeledDataset(val_data, test_transform)
        val_loader = DataLoader(
            validation_data,
            1,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.dataloader.num_workers
        )

        return (
            l_train_loader,
            ul_train_loader,
            val_loader,
            test_loader,
            num_classes,
            img_size
        )

    else:
        return (
            l_train_loader,
            ul_train_loader,
            test_loader,
            num_classes,
            img_size
        )


def gen_dataset(root, dataset, validation_split, cfg, logger=None):
    """
    generate train, val, and test datasets

    Parameters
    -----------
        root: str
            root directory in which data is present or needs to be downloaded
        dataset: str
            dataset name,
            Existing dataset choices: ['cifar10', 'cifar100', 'svhn', 'stl10', 'cifarOOD', 'mnistOOD', 'cifarImbalance']
        validation_split: bool
            if True, return validation loader.
            We use 10% random split of training data as validation data
        cfg: argparse.Namespace or dict
            Dictionary containing necessary arguments for generating the dataset
        logger: logging.Logger
            Logger class for logging the information
    """
    ul_train_data = None
    if dataset == "svhn":
        train_data, test_data = utils.get_svhn(root)
        num_classes = 10
        img_size = 32
    elif dataset == "stl10":
        train_data, ul_train_data, test_data = utils.get_stl10(root)
        num_classes = 10
        img_size = 96
    elif dataset == "cifar10":
        train_data, test_data = utils.get_cifar10(root)
        num_classes = 10
        img_size = 32
    elif dataset == "cifar100":
        train_data, test_data = utils.get_cifar100(root)
        num_classes = 100
        img_size = 32
    elif dataset == "cifarOOD":
        train_data, ul_train_data, test_data = utils.get_cifarOOD(root, cfg.dataset.ood_ratio)
        num_classes = 6
        img_size = 32
    elif dataset == "mnistOOD":
        train_data, ul_train_data, test_data = utils.get_mnistOOD(root, cfg.dataset.ood_ratio)
        num_classes = 6
        img_size = 28
    elif dataset == "cifarImbalance":
        train_data, ul_train_data, test_data = utils.get_cifarClassImb(root, cfg.dataset.ood_ratio)
        num_classes = 10
        img_size = 32
    else:
        raise NotImplementedError

    if validation_split:
        val_data, l_train_data, ul_train_data = __val_labeled_unlabeled_split(
            cfg, train_data, test_data, num_classes, ul_train_data)
    else:
        l_train_data, ul_train_data = __labeled_unlabeled_split(
            cfg, train_data, test_data, num_classes, ul_train_data)
        val_data = None

    ul_train_data["images"] = np.concatenate([ul_train_data["images"], l_train_data["images"]], 0)
    ul_train_data["labels"] = np.concatenate([ul_train_data["labels"], l_train_data["labels"]], 0)

    if logger is not None:
        logger.info("number of :\n \
            training data: %d\n \
            labeled data: %d\n \
            unlabeled data: %d\n \
            validation data: %d\n \
            test data: %d",
            len(train_data["images"]),
            len(l_train_data["images"]),
            len(ul_train_data["images"]),
            0 if val_data is None else len(val_data["images"]),
            len(test_data["images"]))

    labeled_train_data = dataset_class.LabeledDataset(l_train_data)
    unlabeled_train_data = dataset_class.UnlabeledDataset(ul_train_data)

    train_data = np.concatenate([
        labeled_train_data.dataset["images"],
        unlabeled_train_data.dataset["images"]
        ], 0)

    if cfg.dataset.whiten:
        mean = train_data.mean((0, 1, 2)) / 255.
        scale = train_data.std((0, 1, 2)) / 255.
    elif cfg.dataset.zca:
        mean, scale = utils.get_zca_normalization_param(numpy_batch_gcn(train_data))
    elif dataset == 'mnistOOD':
        mean = [0.5]
        scale = [0.5]
    else:
        # from [0, 1] to [-1, 1]
        mean = [0.5, 0.5, 0.5]
        scale = [0.5, 0.5, 0.5]

    # set augmentation
    # RA: RandAugment, WA: Weak Augmentation
    randauglist = "fixmatch" if cfg.ssl_args.alg == "pl" else "uda"

    flags = [True if b == "t" else False for b in cfg.dataset.wa.split(".")]

    if cfg.dataset.labeled_aug == "RA":
        labeled_augmentation = gen_strong_augmentation(
            img_size, mean, scale, flags[0], flags[1], randauglist, cfg.dataset.zca)
    elif cfg.dataset.labeled_aug == "WA":
        labeled_augmentation = gen_weak_augmentation(img_size, mean, scale, *flags, cfg.dataset.zca)
    else:
        raise NotImplementedError

    labeled_train_data.transform = labeled_augmentation

    if cfg.dataset.unlabeled_aug == "RA":
        unlabeled_augmentation = gen_strong_augmentation(
            img_size, mean, scale, flags[0], flags[1], randauglist, cfg.dataset.zca)
    elif cfg.dataset.unlabeled_aug == "WA":
        unlabeled_augmentation = gen_weak_augmentation(img_size, mean, scale, *flags, cfg.dataset.zca)
    else:
        raise NotImplementedError

    if logger is not None:
        logger.info("labeled augmentation")
        logger.info(labeled_augmentation)
        logger.info("unlabeled augmentation")
        logger.info(unlabeled_augmentation)

    unlabeled_train_data.weak_augmentation = unlabeled_augmentation

    if cfg.dataset.strong_aug:
        strong_augmentation = gen_strong_augmentation(
            img_size, mean, scale, flags[0], flags[1], randauglist, cfg.dataset.zca)
        unlabeled_train_data.strong_augmentation = strong_augmentation
        if logger is not None:
            logger.info(strong_augmentation)

    if cfg.dataset.zca:
        test_transform = transforms.Compose([GCN(), ZCA(mean, scale)])
    else:
        test_transform = transforms.Compose([transforms.Normalize(mean, scale, True)])

    test_data = dataset_class.LabeledDataset(test_data, test_transform)

    if validation_split:
        validation_data = dataset_class.LabeledDataset(val_data, test_transform)

        return (
            labeled_train_data,
            unlabeled_train_data,
            validation_data,
            test_data,
            num_classes,
            img_size
        )

    else:
        return (
            labeled_train_data,
            unlabeled_train_data,
            test_data,
            num_classes,
            img_size
        )

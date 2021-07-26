from torch.nn import CrossEntropyLoss

from cords.utils import load_dataset_custom
from cords.utils.data.data_loader import GLISTERDataLoader
from cords.utils.models import TwoLayerNet
import torch
import torch.optim as optim

from train import TrainClassifier


def test_glister_dataloader_mnist():
    model = TwoLayerNet(784, 10, 2048)
    datadir = './data'
    trainset, validset, testset, num_cls = load_dataset_custom(datadir, 'mnist', 'dss')
    train_queue = torch.utils.data.DataLoader(trainset)
    valid_queue = torch.utils.data.DataLoader(validset)
    criterion = CrossEntropyLoss()
    loader = GLISTERDataLoader(train_queue, valid_queue, select_ratio=0.01, select_every=3,
                               model=model, loss=criterion, eta=1, device='cpu', num_cls=10,
                               linear_layer=False, selection_type='Stochastic', verbose=True, r=1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for i in range(0, 4):
        for batch_idx, (inputs, targets) in enumerate(loader):
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def test_glister_mnist():
    config_file = "test_config/config_glister.py"
    classifier = TrainClassifier(config_file)
    classifier.train()
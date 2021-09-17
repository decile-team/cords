# Please run this script at the root dir of cords
# Efficient text classification algorithms
import sys

from cords.utils.models.embedding_bag import EmbeddingBagModel
from cords.utils.models.lstm import LSTMModel

sys.path.append("../")
sys.path.append("./")

import os
import pickle
import argparse
import torch.nn as nn
import torch.optim as optim
from cords.utils.models import EmbeddingBagModel, LSTMModel
from cords.utils.dataloader import *

filepaths = {"corona": "data/corona.pickle",
             "news": "data/news.pickle",
             "twitter": "data/twitter.pickle"}

_adaptive_methods = ["glister", "random-ol"]
_nonadaptive_methods = ["full", "random", "facloc", "graphcut", "sumredun", "satcov", "CRAIG"]
# _nonadaptive_methods = ["random", "facloc", "graphcut", "sumredun", "satcov", "CRAIG"]
_nlp_models = ["LSTM", "bag"]

parser = argparse.ArgumentParser()
# Dataset name
parser.add_argument("--dataset", type=str, choices=list(filepaths.keys()))

# Model selection
parser.add_argument("--model", type=str, default="LSTM", choices=_nlp_models)

# Model arguments
parser.add_argument("--hidden_units", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--embed_dim", type=int, default=16)

# Training arguments
parser.add_argument("--n_epochs", type=int, default=80)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--report_every_batch", type=int, default=50)

# DSS type
parser.add_argument("--dss_strategy", type=str, choices=_adaptive_methods + _nonadaptive_methods, default="full")

# DSS arguments
parser.add_argument("--select_ratio", type=float, default=0.2)
parser.add_argument("--select_every", type=int, default=10)
parser.add_argument("--device", type=str)
parser.add_argument("--r_ratio", type=float, default=1)

# CRAIG argument
parser.add_argument("--linear_layer", type=bool, default=False)
parser.add_argument("--if_convex", type=bool, default=False)
parser.add_argument("--selection_type", type=str, default="PerClass")
parser.add_argument("--optimizer", type=str, default="lazy")

args = parser.parse_args()
if args.dss_strategy in _adaptive_methods:
    args.is_adaptive = True
elif args.dss_strategy in _nonadaptive_methods:
    args.is_adaptive = False
else:
    raise Exception("DSS strategy %s does not exist. " % args.dss_strategy)

criterion = nn.CrossEntropyLoss()
criterion_nored = nn.CrossEntropyLoss(reduction="none")


def validate(model, queue):
    _valid_loss, _valid_tot, _valid_correct = .0, 0, 0
    with torch.no_grad():
        for i_batch, (X, y) in enumerate(queue):
            X, y = X.to(args.device), y.to(args.device)
            _logits = model(X)
            _y = _logits.argmax(1)
            _valid_loss += criterion(_logits, y).sum().item()
            _valid_tot += X.size(0)
            _valid_correct += _y.eq(y).sum().item()

            if i_batch % args.report_every_batch == 0:
                print(
                    "Epoch: %s, validation batch: %s, loss: %10.5f, accuracy: %10.5f. " % (
                        i_epoch, i_batch, _valid_loss / _valid_tot, _valid_correct / _valid_tot))
    return _valid_loss / _valid_tot, _valid_correct / _valid_tot


if __name__ == "__main__":
    assert args.dataset in filepaths, "Dataset %s does not exist. Available datasets: %s" % \
                                      (args.dataset, list(filepaths.keys()))
    filepath = filepaths[args.dataset]

    with open(filepath, 'rb') as handle:
        train, valid, test, vocab_size, n_classes = pickle.load(handle)
    print("vocab_size: %s, n_classes: %s. " % (vocab_size, n_classes))
    n_train, n_valid, n_test = len(train), len(valid), len(test)
    n_epochs, batch_size = args.n_epochs, args.batch_size
    train_queue = DataLoader(train, batch_size=args.batch_size)
    valid_queue = DataLoader(valid, batch_size=args.batch_size)
    test_queue = DataLoader(test, batch_size=args.batch_size)
    if args.model == "bag":
        model = EmbeddingBagModel(vocab_size, args.embed_dim, n_classes)
    elif args.model == "LSTM":
        model = LSTMModel(vocab_size, args.hidden_units, args.num_layers, args.embed_dim, n_classes)
    else:
        raise Exception("Model %s is not supported, supported models are: %s. " % (args.model, str(_nlp_models)))
    model = model.to(args.device)
    print("model: %s, number of parameters: %s" % (args.model, sum(param.numel() for param in model.parameters())))
    lr, momentum, weight_decay = args.lr, args.momentum, args.weight_decay
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    budget = int(args.select_ratio * n_train)

    # DSS dataloader
    if args.dss_strategy == "full":
        dss_train_queue = train_queue
    elif args.dss_strategy == "glister":
        dss_train_queue = GLISTERDataLoader(train_queue, valid_queue, budget=budget, select_every=args.select_every,
                                            model=model, loss=criterion_nored, eta=lr, device=args.device,
                                            num_cls=n_classes, linear_layer=False, selection_type="Stochastic",
                                            r=budget * args.r_ratio, batch_size=args.batch_size)
    elif args.dss_strategy == "random-ol":
        dss_train_queue = OnlineRandomDataLoader(train_queue, valid_queue, budget=budget,
                                                 select_every=args.select_every, model=model, loss=criterion_nored,
                                                 device=args.device, batch_size=args.batch_size, verbose=True)
    elif args.dss_strategy == "random":
        dss_train_queue = RandomDataLoader(train_queue, valid_queue, budget=budget, model=model, loss=criterion_nored,
                                           device=args.device, batch_size=args.batch_size, verbose=True)
    elif args.dss_strategy == "facloc":
        dss_train_queue = FacLocDataLoader(train_queue, valid_queue, budget=budget, model=model, loss=criterion_nored,
                                           device=args.device, batch_size=args.batch_size, verbose=True)
    elif args.dss_strategy == "graphcut":
        dss_train_queue = GraphCutDataLoader(train_queue, valid_queue, budget=budget, model=model, loss=criterion_nored,
                                             device=args.device, batch_size=args.batch_size, verbose=True)
    elif args.dss_strategy == "sumredun":
        dss_train_queue = SumRedundancyDataLoader(train_queue, valid_queue, budget=budget, model=model,
                                                  loss=criterion_nored, device=args.device, batch_size=args.batch_size,
                                                  verbose=True)
    elif args.dss_strategy == "satcov":
        dss_train_queue = SaturatedCoverageDataLoader(train_queue, valid_queue, budget=budget, model=model,
                                                      loss=criterion_nored, device=args.device,
                                                      batch_size=args.batch_size, verbose=True)
    elif args.dss_strategy == "CRAIG":
        dss_train_queue = CRAIGDataLoader(train_queue, valid_queue, budget=budget, model=model, loss=criterion_nored,
                                          device=args.device, num_cls=n_classes, linear_layer=args.linear_layer,
                                          if_convex=args.if_convex, selection_type=args.selection_type,
                                          optimizer=args.optimizer, batch_size=batch_size, verbose=True)
    else:
        raise Exception("Strategy %s does not exist. " % args.dss_strategy)

    start = time.time()
    train_loss, valid_loss, train_elapsed = [], [], []
    train_accu, valid_accu = [], []

    for i_epoch in range(n_epochs):
        _train_loss, _train_tot, _train_correct = .0, 0, 0
        for i_batch, (X, y) in enumerate(dss_train_queue):
            X, y = X.to(args.device), y.to(args.device)
            logits = model(X)
            _y = logits.argmax(1)
            _loss = criterion(logits, y).sum()
            # Optimize step
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            # Record metrics
            _train_loss += _loss.item()
            _train_tot += X.size(0)
            _train_correct += _y.eq(y).sum().item()
            if i_batch % args.report_every_batch == 0:
                print(
                    "Epoch idx: %s, batch idx: %s, training loss: %10.5f, training accuracy: %10.5f. " % (
                        i_epoch, i_batch, _train_loss / _train_tot, _train_correct / _train_tot))
        train_elapsed.append(time.time() - start)
        _valid_loss, _valid_accu = validate(model, valid_queue)
        train_loss.append(_train_loss / _train_tot)
        train_accu.append(_train_correct / _train_tot)
        valid_loss.append(_valid_loss)
        valid_accu.append(_valid_accu)
        print(
            "Epoch: %s, "
            "training elapsed: %5.2f, "
            "training loss: %10.5f, "
            "training accuracy: %10.5f, "
            "validation loss: %10.5f, "
            "validation accuracy: %10.5f " % (
                i_epoch, train_elapsed[-1], train_loss[-1], train_accu[-1], valid_loss[-1], valid_accu[-1]))

    test_loss, test_accu = validate(model, test_queue)

    print("Test loss: %10.5f" % test_loss)
    print("Test accu: %10.5f" % test_accu)

    # Save results
    save_path = os.path.join(".", "scripts", "RESULTS", "EXP_%s" % start)
    print("Saving to path: %s" % save_path)
    os.makedirs(save_path)
    with open(os.path.join(save_path, 'save_dict.pickle'), 'wb') as handle:
        save_dict = {"args": args,
                     "train_loss": train_loss, "train_accu": train_accu,
                     "valid_loss": valid_loss, "valid_accu": valid_accu,
                     "epochs": range(n_epochs), "train_elapsed": train_elapsed,
                     "test_loss": test_loss, "test_accu": test_accu}
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

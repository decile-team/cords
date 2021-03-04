import yaml
import argparse
import time
import datetime
import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from cords.utils.models import *
from cords.utils.custom_dataset import load_dataset_custom
from torch.utils.data import Subset
from math import floor
import subprocess
"""
Argument Parsing
"""
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config_dir', type=str, default="configs/default_config.yaml",
                    help='Config File Location')
args = parser.parse_args()

"""
Parsing Config file
"""
with open(args.config_dir, 'r') as config_file:
    data = yaml.load(config_file)

if data.setting == 'supervisedlearning':
    from cords.selectionstrategies.supervisedlearning import *
elif data.setting == 'general':
    from cords.selectionstrategies.general import *

"""
Loss Evaluation
"""

def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(data.training_args.device), targets.to(data.training_args.device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss

"""
#Model Creation
"""

def create_model(name, num_cls, device):
    if name == 'ResNet18':
        model = ResNet18(num_cls)
    elif name == 'MnistNet':
        model = MnistNet()
    elif name == 'ResNet164':
        model = ResNet164(num_cls)
    model = model.to(device)
    return model


"""#Loss Type, Optimizer and Learning Rate Scheduler"""
def loss_function():
    if data.loss.name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
        criterion_nored = nn.CrossEntropyLoss(reduction='none')
    return criterion, criterion_nored


def optimizer_with_scheduler(model, num_epochs, learning_rate, m=0.9, wd=5e-4):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=m, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler


def generate_cumulative_timing(mod_timing):
    tmp = 0
    mod_cum_timing = np.zeros(len(mod_timing))
    for i in range(len(mod_timing)):
        tmp += mod_timing[i]
        mod_cum_timing[i] = tmp
    return mod_cum_timing / 3600


from scipy.signal import lfilter


def filter(y):
    n = 1  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b, a, y)
    return yy


"""
#General Training Loop with Data Selection Strategies
"""


def train_model(num_epochs, dataset_name, datadir, feature, model_name, fraction, select_every, learning_rate, run,
                device, strategy):

    # Loading the Dataset
    trainset, validset, testset, num_cls = load_dataset_custom(datadir, dataset_name, feature)
    N = len(trainset)
    trn_batch_size = 20
    val_batch_size = 1000
    tst_batch_size = 1000

    # Creating the Data Loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                              shuffle=False, pin_memory=True)

    valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                            shuffle=False, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                             shuffle=False, pin_memory=True)

    # Budget for subset selection
    bud = int(fraction * N)
    print("Budget, fraction and N:", bud, fraction, N)

    # Subset Selection and creating the subset data loader
    start_idxs = np.random.choice(N, size=bud, replace=False)
    idxs = start_idxs
    data_sub = Subset(trainset, idxs)
    subset_trnloader = torch.utils.data.DataLoader(data_sub, batch_size=trn_batch_size,
                                                   shuffle=False, pin_memory=True)

    # Variables to store accuracies
    gammas = torch.ones(len(idxs)).to(device)
    substrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    subtrn_acc = np.zeros(num_epochs)

    # Results logging file
    print_every = 3
    all_logs_dir = 'results/' + strategy + '/' + dataset_name + '/warmstart/' + str(
        fraction) + '/' + str(select_every) + '/' + str(run)
    print(all_logs_dir)
    subprocess.run(["mkdir", "-p", all_logs_dir])
    path_logfile = os.path.join(all_logs_dir, dataset_name + '.txt')
    logfile = open(path_logfile, 'w')
    exp_name = dataset_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
               '_selEvery:' + str(select_every) + '_variant' + '_runs' + str(run)
    print(exp_name)

    # Model Creation
    model = create_model(model_name, num_cls, device)
    model1 = create_model(model_name, num_cls, device)
    # Loss Functions
    criterion, criterion_nored = loss_function()

    # Getting the optimizer and scheduler
    optimizer, scheduler = optimizer_with_scheduler(model, num_epochs, learning_rate)

    if strategy == 'GradMatch':
        # OMPGradMatch Selection strategy
        setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion,
                                          learning_rate, device, num_cls, True, 'PerClassPerGradient',
                                          False, lam=0.5, eps=1e-100)
    elif strategy == 'GradMatchPB':
        setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion,
                                          learning_rate, device, num_cls, True, 'PerBatch',
                                          False, lam=0, eps=1e-100)
    elif strategy == 'GLISTER':
        # GLISTER Selection strategy
        setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
                                     learning_rate, device, num_cls, False, 'Stochastic', r=int(bud))

    elif strategy == 'CRAIG':
        # CRAIG Selection strategy
        setf_model = CRAIGStrategy(trainloader, valloader, model1, criterion,
                                   device, num_cls, False, False, 'PerClass')

    elif strategy == 'CRAIGPB':
        # CRAIG Selection strategy
        setf_model = CRAIGStrategy(trainloader, valloader, model1, criterion,
                                   device, num_cls, False, False, 'PerBatch')

    elif strategy == 'CRAIG-Explore':
        # CRAIG Selection strategy
        setf_model = CRAIGStrategy(trainloader, valloader, model1, criterion,
                                   device, num_cls, False, False, 'PerClass')
        # Random-Online Selection strategy
        rand_setf_model = RandomStrategy(trainloader, online=True)

    elif strategy == 'CRAIGPB-Explore':
        # CRAIG Selection strategy
        setf_model = CRAIGStrategy(trainloader, valloader, model1, criterion,
                                   device, num_cls, False, False, 'PerBatch')
        # Random-Online Selection strategy
        rand_setf_model = RandomStrategy(trainloader, online=True)

    elif strategy == 'Random':
        # Random Selection strategy
        setf_model = RandomStrategy(trainloader, online=False)

    elif strategy == 'Random-Online':
        # Random-Online Selection strategy
        setf_model = RandomStrategy(trainloader, online=True)

    elif strategy == 'GLISTER-Explore':
        # GLISTER Selection strategy
        setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion,
                                     learning_rate, device, num_cls, False, 'Stochastic', r=int(bud))
        # Random-Online Selection strategy
        rand_setf_model = RandomStrategy(trainloader, online=True)

    elif strategy == 'GradMatch-Explore':
        # OMPGradMatch Selection strategy
        setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion,
                                          learning_rate, device, num_cls, True, 'PerClassPerGradient',
                                          False, lam=0.5, eps=1e-100)
        # Random-Online Selection strategy
        rand_setf_model = RandomStrategy(trainloader, online=True)

    elif strategy == 'GradMatchPB-Explore':
        # OMPGradMatch Selection strategy
        setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion,
                                          learning_rate, device, num_cls, True, 'PerBatch',
                                          False, lam=0, eps=1e-100)
        # Random-Online Selection strategy
        rand_setf_model = RandomStrategy(trainloader, online=True)

    print("=======================================", file=logfile)
    kappa_epochs = int(0.5 * num_epochs)
    full_epochs = floor(kappa_epochs/int(fraction*100))

    for i in range(num_epochs):
        subtrn_loss = 0
        subtrn_correct = 0
        subtrn_total = 0
        subset_selection_time = 0

        if strategy in ['Random-Online']:
            start_time = time.time()
            subset_idxs, gammas = setf_model.select(int(bud))
            idxs = subset_idxs
            subset_selection_time += (time.time() - start_time)
            gammas = gammas.to(device)

        elif strategy in ['Random']:
            pass

        elif (strategy in ['GLISTER', 'GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']) and (
                ((i + 1) % select_every) == 0):
            start_time = time.time()
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            if strategy in ['CRAIG', 'CRAIGPB']:
                subset_idxs, gammas = setf_model.select(int(bud), clone_dict, 'lazy')
            else:
                subset_idxs, gammas = setf_model.select(int(bud), clone_dict)
            model.load_state_dict(cached_state_dict)
            idxs = subset_idxs
            if strategy in ['GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']:
                gammas = torch.from_numpy(np.array(gammas)).to(device).to(torch.float32)
            subset_selection_time += (time.time() - start_time)

        elif (strategy in ['GLISTER-Explore', 'GradMatch-Explore', 'GradMatchPB-Explore', 'CRAIG-Explore',
                           'CRAIGPB-Explore']):
            start_time = time.time()
            if i < full_epochs:
                subset_idxs, gammas = rand_setf_model.select(int(bud))
                idxs = subset_idxs
                gammas = gammas.to(device)
            elif ((i % select_every == 0) and (i >= kappa_epochs)):
                cached_state_dict = copy.deepcopy(model.state_dict())
                clone_dict = copy.deepcopy(model.state_dict())
                if strategy in ['CRAIG-Explore', 'CRAIGPB-Explore']:
                    subset_idxs, gammas = setf_model.select(int(bud), clone_dict, 'lazy')
                else:
                    subset_idxs, gammas = setf_model.select(int(bud), clone_dict)
                model.load_state_dict(cached_state_dict)
                idxs = subset_idxs
                if strategy in ['GradMatch-Explore', 'GradMatchPB-Explore', 'CRAIG-Explore', 'CRAIGPB-Explore']:
                    gammas = torch.from_numpy(np.array(gammas)).to(device).to(torch.float32)
            subset_selection_time += (time.time() - start_time)

        print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
        data_sub = Subset(trainset, idxs)
        subset_trnloader = torch.utils.data.DataLoader(data_sub, batch_size=trn_batch_size, shuffle=False,
                                                       pin_memory=True)

        model.train()
        batch_wise_indices = list(subset_trnloader.batch_sampler)
        if strategy in ['CRAIG', 'CRAIGPB', 'GradMatch', 'GradMatchPB']:
            start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                inputs, targets = inputs.to(device), targets.to(device,
                                                                non_blocking=True)  # targets can have non_blocking=True.
                optimizer.zero_grad()
                outputs = model(inputs)
                losses = criterion_nored(outputs, targets)
                loss = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (gammas[batch_wise_indices[batch_idx]].sum())
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        elif strategy in ['CRAIGPB-Explore', 'CRAIG-Explore', 'GradMatch-Explore', 'GradMatchPB-Explore']:
            start_time = time.time()
            if i < full_epochs:
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device,
                                                                    non_blocking=True)  # targets can have non_blocking=True.
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    subtrn_loss += loss.item()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()

            elif i >= kappa_epochs:
                for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                    inputs, targets = inputs.to(device), targets.to(device,
                                                                    non_blocking=True)  # targets can have non_blocking=True.
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    losses = criterion_nored(outputs, targets)
                    loss = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (
                        gammas[batch_wise_indices[batch_idx]].sum())
                    loss.backward()
                    subtrn_loss += loss.item()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        elif strategy in ['GLISTER', 'Random', 'Random-Online']:
            start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                inputs, targets = inputs.to(device), targets.to(device,
                                                                non_blocking=True)  # targets can have non_blocking=True.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        elif strategy in ['GLISTER-Explore']:
            start_time = time.time()
            if i < full_epochs:
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device,
                                                                    non_blocking=True)  # targets can have non_blocking=True.
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    subtrn_loss += loss.item()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
            elif i >= kappa_epochs:
                for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                    inputs, targets = inputs.to(device), targets.to(device,
                                                                    non_blocking=True)  # targets can have non_blocking=True.
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    subtrn_loss += loss.item()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        elif strategy in ['Full']:
            start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device,
                                                                non_blocking=True)  # targets can have non_blocking=True.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time
        scheduler.step()
        timing[i] = train_time + subset_selection_time
        # print("Epoch timing is: " + str(timing[i]))

        val_loss = 0
        val_correct = 0
        val_total = 0
        tst_correct = 0
        tst_total = 0
        tst_loss = 0
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                # print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

            for batch_idx, (inputs, targets) in enumerate(testloader):
                # print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += loss.item()
                _, predicted = outputs.max(1)
                tst_total += targets.size(0)
                tst_correct += predicted.eq(targets).sum().item()

        val_acc[i] = val_correct / val_total
        tst_acc[i] = tst_correct / tst_total
        subtrn_acc[i] = subtrn_correct / subtrn_total
        substrn_losses[i] = subtrn_loss
        val_losses[i] = val_loss
        print('Epoch:', i + 1, 'Validation Accuracy: ', val_acc[i], 'Test Accuracy: ', tst_acc[i], 'Time: ', timing[i])
    print(strategy + " Selection Run---------------------------------")
    print("Final SubsetTrn:", subtrn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc.max())
    print("Test Data Loss and Accuracy:", tst_loss, tst_acc.max())
    print('-----------------------------------')

    # Results logging into the file
    print(strategy, file=logfile)
    print('---------------------------------------------------------------------', file=logfile)
    val = "Validation Accuracy, "
    tst = "Test Accuracy, "
    time_str = "Time, "

    for i in range(num_epochs):
        time_str = time_str + "," + str(timing[i])
        val = val + "," + str(val_acc[i])
        tst = tst + "," + str(tst_acc[i])

    print(timing, file=logfile)
    print(val, file=logfile)
    print(tst, file=logfile)

    omp_timing = np.array(timing)
    omp_cum_timing = list(generate_cumulative_timing(omp_timing))
    omp_tst_acc = list(filter(tst_acc))
    print("Total time taken by " + strategy + " = " + str(omp_cum_timing[-1]))
    logfile.close()



print()
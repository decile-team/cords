import time
import copy
import datetime
import numpy as np
import os
import subprocess
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from selectionstrategies.supervisedlearning.glisterstrategy import GLISTERStrategy as Strategy
from utils.models.mnist_net import MnistNet
from utils.models.resnet import ResNet18
from utils.models.simpleNN_net import TwoLayerNet
from utils.custom_dataset import load_mnist_cifar, load_dataset_custom
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler
from torch.autograd import Variable
from selectionstrategies.supervisedlearning.craigstrategy import CRAIGStrategy as CRAIG
import math

def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss


def write_knndata(datadir, x_trn, y_trn, x_val, y_val, x_tst, y_tst, dset_name):
    ## Create VAL data
    subprocess.run(["mkdir", "-p", datadir])
    # x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
    trndata = np.c_[x_trn.cpu().numpy(), y_trn.cpu().numpy()]
    valdata = np.c_[x_val.cpu().numpy(), y_val.cpu().numpy()]
    tstdata = np.c_[x_tst.cpu().numpy(), y_tst.cpu().numpy()]
    # Write out the trndata
    trn_filepath = os.path.join(datadir, 'knn_' + dset_name + '.trn')
    val_filepath = os.path.join(datadir, 'knn_' + dset_name + '.val')
    tst_filepath = os.path.join(datadir, 'knn_' + dset_name + '.tst')
    np.savetxt(trn_filepath, trndata, fmt='%.6f')
    np.savetxt(val_filepath, valdata, fmt='%.6f')
    np.savetxt(tst_filepath, tstdata, fmt='%.6f')
    return


def perform_knnsb_selection(datadir, dset_name, budget, selUsing):
    trn_filepath = os.path.join(datadir, 'knn_' + dset_name + '.trn')
    if selUsing == 'val':
        val_filepath = os.path.join(datadir, 'knn_' + dset_name + '.val')
    else:
        val_filepath = trn_filepath

    run_path = './run_data/'
    output_dir = run_path + 'KNNSubmod_' + dset_name + '/'
    indices_file = output_dir + 'KNNSubmod_' + str((int)(budget*100)) + ".subset"
    subprocess.call(["mkdir", output_dir])
    knnsb_args = []
    knnsb_args.append('../build/KNNSubmod')
    knnsb_args.append(trn_filepath)
    knnsb_args.append(val_filepath)
    knnsb_args.append(" ")  # File delimiter!!
    knnsb_args.append(str(budget))
    knnsb_args.append(indices_file)
    knnsb_args.append("1")  # indicates cts data. Deprecated.
    print("Obtaining the subset")
    subprocess.run(knnsb_args)
    print("finished selection")
    # Can make it return the indices_file if using with other function.
    idxs_knnsb = np.genfromtxt(indices_file, delimiter=',', dtype=int) # since they are indices!
    return idxs_knnsb

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print("Using Device:", device)

## Convert to this argparse
datadir = '../../data/dna/'
data_name = 'dna'
fraction = float(0.1)
num_epochs = int(200)
select_every = int(1)
feature = 'dss'# 70
warm_method = 0  # whether to use warmstart-onestep (1) or online (0)
num_runs = 1  # number of random runs
learning_rate = 0.05
all_logs_dir = './results/' + data_name +'/' + feature +'/' + str(fraction) + '/' + str(select_every)
print(all_logs_dir)
subprocess.run(["mkdir", "-p", all_logs_dir])
path_logfile = os.path.join(all_logs_dir, data_name + '.txt')
logfile = open(path_logfile, 'w')
exp_name = data_name + '_fraction:' + str(fraction) + '_epochs:' + str(num_epochs) + \
           '_selEvery:' + str(select_every) + '_variant' + str(warm_method) + '_runs' + str(num_runs)
print(exp_name)
exp_start_time = datetime.datetime.now()
print("=======================================", file=logfile)
print(exp_name, str(exp_start_time), file=logfile)
fullset, valset, testset, M, num_cls = load_dataset_custom(datadir, data_name, feature, False)
#fullset, valset, testset, num_cls = load_mnist_cifar(datadir, data_name, feature)
# Validation Data set is 10% of the Entire Trainset.
validation_set_fraction = 0.1
num_fulltrn = len(fullset)
num_val = int(num_fulltrn * validation_set_fraction)
num_trn = num_fulltrn - num_val
trainset, validset = random_split(fullset, [num_trn, num_val])
N = len(trainset)
trn_batch_size = 20
val_batch_size = 1000
tst_batch_size = 1000

trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                          shuffle=False, pin_memory=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, shuffle=False,
                                               sampler=SubsetRandomSampler(validset.indices),
                                               pin_memory=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                         shuffle=False, pin_memory=True)

bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)
# Transfer all the data to GPU
print_every = 3


def train_model_craig(start_rand_idxs, bud):
    torch.manual_seed(42)
    np.random.seed(42)
    if data_name == 'mnist':
        model = MnistNet()
        num_channels = 1
    elif data_name == 'cifar10':
        model = ResNet18(num_cls)
        num_channels = 3
    else:
        model = TwoLayerNet(M, num_cls, 50)
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate,
    #                            momentum=0.9, weight_decay=0.1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    full_trn_acc = np.zeros(num_epochs)
    subtrn_acc = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                   shuffle=False, sampler=SubsetRandomSampler(idxs), pin_memory=True)

    # cosine learning rate
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, math.ceil(len(idxs)/trn_batch_size) * num_epochs)
    setf_model = CRAIG(trainloader, valloader, model, 'CrossEntropy', device, num_cls, True, True, 'Supervised')
    print("Starting CRAIG Run")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    for i in range(0, num_epochs):
        start_time = time.time()
        print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
        #if ((i) % select_every) == 0:
        cached_state_dict = copy.deepcopy(model.state_dict())
        clone_dict = copy.deepcopy(model.state_dict())
        subset_idxs, gammas = setf_model.select(int(bud), clone_dict, 'stochastic')
        model.load_state_dict(cached_state_dict)
        gammas = np.array(gammas)
        idxs = subset_idxs
        #print(gammas)
        print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
        actual_idxs = np.array(trainset.indices)[idxs]
        batch_wise_indices = list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))
        subtrn_loss = 0
        subtrn_total = 0
        subtrn_correct = 0
        model.train()
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [fullset[actual_idxs[x]][0].view(-1, num_channels, fullset[actual_idxs[x]][0].shape[1], fullset[actual_idxs[x]][0].shape[2]) for x in batch_idx],
                dim=0).type(torch.float)
            targets = torch.tensor([fullset[actual_idxs[x]][1] for x in batch_idx])
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True) # targets can have non_blocking=True.
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion_nored(outputs, targets)
            batch_gammas = (1/N) * gammas[batch_idx]
            loss = torch.dot(torch.from_numpy(batch_gammas).to(device).type(torch.float), losses)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
            subtrn_accu = 100 * subtrn_correct / subtrn_total

        timing[i] = time.time() - start_time
        # print("Epoch timing is: " + str(timing[i]))
        val_loss = 0
        val_correct = 0
        val_total = 0
        tst_correct = 0
        tst_total = 0
        tst_loss = 0
        full_trn_loss = 0
        # subtrn_loss = 0
        full_trn_correct = 0
        full_trn_total = 0
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

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()
                _, predicted = outputs.max(1)
                full_trn_total += targets.size(0)
                full_trn_correct += predicted.eq(targets).sum().item()

        val_acc[i] = val_correct / val_total
        tst_acc[i] = tst_correct / tst_total
        subtrn_acc[i] = subtrn_correct / subtrn_total
        full_trn_acc[i] = full_trn_correct / full_trn_total
        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss,Time:', subtrn_loss, full_trn_loss, val_loss, timing[i])

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc[-1])
    print("Test Data Loss and Accuracy:", tst_loss, tst_acc[-1])
    print('-----------------------------------')

    print("CRAIG Every Epoch", file=logfile)
    print('---------------------------------------------------------------------', file=logfile)
    val = "Validation Accuracy,"
    tst = "Test Accuracy,"
    time_str = "Time,"
    for i in range(num_epochs):
        time_str = time_str + "," + str(timing[i])
        val = val + "," + str(val_acc[i])
        tst = tst + "," + str(tst_acc[i])
    print(timing, file=logfile)
    print(val, file=logfile)
    print(tst, file=logfile)
    return val_acc[-1], tst_acc[-1], subtrn_acc[-1], full_trn_acc[
        -1], val_loss, tst_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, np.sum(
        timing), timing, val_acc, tst_acc


def train_model_glister_closed(start_rand_idxs, bud):
    torch.manual_seed(42)
    np.random.seed(42)
    if data_name == 'mnist':
        model = MnistNet()
        #model = Net()
    elif data_name == 'cifar10':
        model = ResNet18(num_cls)
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    #criterion_nored = nn.CrossEntropyLoss(reduction='none')
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    if data_name == 'mnist':
        setf_model = Strategy(trainloader, valloader, model, criterion,
                              learning_rate, device, num_cls, True, 'RModular')
        num_channels = 1
    elif data_name == 'cifar10':
        setf_model = Strategy(trainloader, valloader, model, criterion,
                              learning_rate, device, num_cls, True, 'RModular')
        num_channels = 3
    print("Starting Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    full_trn_acc = np.zeros(num_epochs)
    subtrn_acc = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
            			shuffle=False, sampler=SubsetRandomSampler(idxs), pin_memory=True)
    for i in range(0, num_epochs):
        subtrn_loss = 0
        subtrn_correct = 0
        subtrn_total = 0
        start_time = time.time()
        if (((i+1) % select_every) == 0):
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_start_time = time.time()
            subset_idxs, grads_idxs = setf_model.select(int(bud), clone_dict)
            subset_end_time = time.time() - subset_start_time
            print("Subset Selection Time is:" + str(subset_end_time))
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            #actual_idxs = np.array(trainset.indices)[idxs]
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
            			shuffle=False, sampler=SubsetRandomSampler(idxs), pin_memory=True)
            #batch_wise_indices = [actual_idxs[x] for x in list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))]
        model.train()
        #for batch_idx in batch_wise_indices:
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            #inputs = torch.cat(
            #    [fullset[x][0].view(-1, num_channels, fullset[x][0].shape[1], fullset[x][0].shape[2]) for x in batch_idx],
            #    dim=0).type(torch.float)
            #targets = torch.tensor([fullset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True) # targets can have non_blocking=True.
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
        scheduler.step()
        timing[i] = time.time() - start_time
        #print("Epoch timing is: " + str(timing[i]))
        val_loss = 0
        val_correct = 0
        val_total = 0
        tst_correct = 0
        tst_total = 0
        tst_loss = 0
        full_trn_loss = 0
        #subtrn_loss = 0
        full_trn_correct = 0
        full_trn_total = 0
        model.eval()
        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(valloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

            for batch_idx, (inputs, targets) in enumerate(testloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += loss.item()
                _, predicted = outputs.max(1)
                tst_total += targets.size(0)
                tst_correct += predicted.eq(targets).sum().item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()
                _, predicted = outputs.max(1)
                full_trn_total += targets.size(0)
                full_trn_correct += predicted.eq(targets).sum().item()

        val_acc[i] = val_correct/val_total
        tst_acc[i] = tst_correct/tst_total
        subtrn_acc[i] = subtrn_correct/subtrn_total
        full_trn_acc[i] = full_trn_correct/full_trn_total
        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss,Time:', subtrn_loss, full_trn_loss, val_loss, timing[i])


    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc[-1])
    print("Test Data Loss and Accuracy:", tst_loss, tst_acc[-1])
    print('-----------------------------------')

    print("GLISTER", file=logfile)
    print('---------------------------------------------------------------------', file=logfile)
    val = "Validation Accuracy,"
    tst = "Test Accuracy,"
    time_str = "Time,"
    for i in range(num_epochs):
        time_str = time_str + "," + str(timing[i])
        val = val + "," + str(val_acc[i])
        tst = tst + "," + str(tst_acc[i])
    print(timing, file=logfile)
    print(val, file=logfile)
    print(tst, file=logfile)
    return val_acc[-1], tst_acc[-1],  subtrn_acc[-1], full_trn_acc[-1], val_loss, tst_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, np.sum(timing), timing, val_acc, tst_acc


def train_model_random_online(start_rand_idxs, bud):
    torch.manual_seed(42)
    np.random.seed(42)
    if data_name == 'mnist':
        model = MnistNet()
        #model = Net()
    elif data_name == 'cifar10':
        model = ResNet18(num_cls)
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    print("Starting Random Online Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    full_trn_acc = np.zeros(num_epochs)
    subtrn_acc = np.zeros(num_epochs)

    for i in range(0, num_epochs):
        subtrn_loss = 0
        subtrn_correct = 0
        subtrn_total = 0
        start_time = time.time()
        if (((i) % select_every) == 0):
            idxs = np.random.choice(N, size=bud, replace=False)
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
            			shuffle=False, sampler=SubsetRandomSampler(idxs), pin_memory=True)
        model.train()
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True) # targets can have non_blocking=True.
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
        scheduler.step()
        timing[i] = time.time() - start_time
        #print("Epoch timing is: " + str(timing[i]))
        val_loss = 0
        val_correct = 0
        val_total = 0
        tst_correct = 0
        tst_total = 0
        tst_loss = 0
        full_trn_loss = 0
        #subtrn_loss = 0
        full_trn_correct = 0
        full_trn_total = 0
        model.eval()
        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(valloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

            for batch_idx, (inputs, targets) in enumerate(testloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += loss.item()
                _, predicted = outputs.max(1)
                tst_total += targets.size(0)
                tst_correct += predicted.eq(targets).sum().item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()
                _, predicted = outputs.max(1)
                full_trn_total += targets.size(0)
                full_trn_correct += predicted.eq(targets).sum().item()

        val_acc[i] = val_correct/val_total
        tst_acc[i] = tst_correct/tst_total
        subtrn_acc[i] = subtrn_correct/subtrn_total
        full_trn_acc[i] = full_trn_correct/full_trn_total
        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss,Time:', subtrn_loss, full_trn_loss, val_loss, timing[i])


    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc[-1])
    print("Test Data Loss and Accuracy:", tst_loss, tst_acc[-1])
    print('-----------------------------------')

    print("GLISTER", file=logfile)
    print('---------------------------------------------------------------------', file=logfile)
    val = "Validation Accuracy,"
    tst = "Test Accuracy,"
    time_str = "Time,"
    for i in range(num_epochs):
        time_str = time_str + "," + str(timing[i])
        val = val + "," + str(val_acc[i])
        tst = tst + "," + str(tst_acc[i])
    print(timing, file=logfile)
    print(val, file=logfile)
    print(tst, file=logfile)
    return val_acc[-1], tst_acc[-1],  subtrn_acc[-1], full_trn_acc[-1], val_loss, tst_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, np.sum(timing), timing, val_acc, tst_acc


def train_model_mod_online(start_rand_idxs, bud):
    torch.manual_seed(42)
    np.random.seed(42)
    if data_name == 'mnist':
        model = MnistNet()
    elif data_name == 'cifar10':
        model = ResNet18(num_cls)
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    print("Starting Modified Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)

    timing = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    tst_acc = np.zeros(num_epochs)
    full_trn_acc = np.zeros(num_epochs)
    subtrn_acc = np.zeros(num_epochs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                              shuffle=True)

    for i in range(0, num_epochs):
        start_time = time.time()
        model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            # Variables in Pytorch are differentiable.
            inputs, target = Variable(inputs), Variable(inputs)
            # This will zero out the gradients for this batch.
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        timing[i] = time.time() - start_time
        val_loss = 0
        val_correct = 0
        val_total = 0
        tst_correct = 0
        tst_total = 0
        tst_loss = 0
        full_trn_loss = 0
        subtrn_loss = 0
        full_trn_correct = 0
        full_trn_total = 0
        subtrn_correct = 0
        subtrn_total = 0
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

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()
                _, predicted = outputs.max(1)
                full_trn_total += targets.size(0)
                full_trn_correct += predicted.eq(targets).sum().item()

            for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                subtrn_loss += loss.item()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()

        val_acc[i] = val_correct / val_total
        tst_acc[i] = tst_correct / tst_total
        subtrn_acc[i] = subtrn_correct / subtrn_total
        full_trn_acc[i] = full_trn_correct / full_trn_total
        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss,Time:', subtrn_loss, full_trn_loss, val_loss, timing[i])
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc[-1])
    print("Test Data Loss and Accuracy:", tst_loss, tst_acc[-1])
    print('-----------------------------------')

    print("Full Training", file=logfile)
    print('---------------------------------------------------------------------', file=logfile)
    val = "Validation Accuracy,"
    tst = "Test Accuracy,"
    time_str = "Time,"
    for i in range(num_epochs):
        time_str = time_str + "," + str(timing[i])
        val = val + "," + str(val_acc[i])
        tst = tst + "," + str(tst_acc[i])
    print(timing, file=logfile)
    print(val, file=logfile)
    print(tst, file=logfile)
    return val_acc[-1], tst_acc[-1], subtrn_acc[-1], full_trn_acc[-1], val_loss, tst_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, \
           np.sum(timing), timing, val_acc, tst_acc



start_idxs = np.random.choice(N, size=bud, replace=False)
random_subset_idx = [trainset.indices[x] for x in start_idxs]

craig_val_valacc, craig_val_tstacc, craig_val_subtrn_acc, craig_val_full_trn_acc, craig_val_valloss, craig_val_tstloss,  craig_val_subtrnloss, \
craig_val_full_trn_loss, craig_fval_val_losses, craig_fval_substrn_losses, craig_fval_fulltrn_losses, craig_subset_idxs, \
craig_step_time, craig_timing, craig_val_accuracies, craig_tst_accuracies= \
train_model_craig(start_idxs, bud)

"""
# Modified OneStep Runs
#mod_val_valacc, mod_val_tstacc, mod_val_subtrn_acc, mod_val_full_trn_acc, mod_val_valloss, mod_val_tstloss, \
#mod_val_subtrnloss, mod_val_full_trn_loss, mod_val_val_losses, mod_val_substrn_losses, mod_val_fulltrn_losses,\
#mod_subset_idxs, mod_one_step_time, mod_timing, mod_val_accuracies, mod_tst_accuracies = \
#train_model_mod_online(start_idxs, bud)

# Online algo run
closed_val_valacc, closed_val_tstacc, closed_val_subtrn_acc, closed_val_full_trn_acc, closed_val_valloss, closed_val_tstloss,  closed_val_subtrnloss, \
closed_val_full_trn_loss, closed_fval_val_losses, closed_fval_substrn_losses, closed_fval_fulltrn_losses, closed_subset_idxs, \
closed_step_time, closed_timing, closed_val_accuracies, closed_tst_accuracies= \
train_model_glister_closed(start_idxs, bud)

#mod_cum_timing = np.zeros(num_epochs)
closed_cum_timing = np.zeros(num_epochs)

#tmp = 0
#for i in range(len(mod_timing)):
#    tmp += mod_timing[i]
#    mod_cum_timing[i] = tmp

tmp = 0
for i in range(len(closed_timing)):
    tmp += closed_timing[i]
    closed_cum_timing[i] = tmp

###### Test accuray #############
plt.figure()
# plt.plot(craig_timing, craig_tstacc,'g-' , label='CRAIG')
#plt.plot(mod_cum_timing, mod_tst_accuracies, 'orange', label='full training')
plt.plot(closed_cum_timing, closed_tst_accuracies, 'b-', label='GLISTER')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Test accuracy')
plt.title('Test Accuracy vs Time ' + data_name + '_' + str(fraction))
plt_file = path_logfile + '_' + str(fraction) + 'tst_accuracy_v=VAL.png'
plt.savefig(plt_file)
plt.clf()

########################################################################
###### Validation #############

plt.figure()
# plt.plot(craig_timing, craig_valacc,'g-' , label='CRAIG')
#plt.plot(mod_cum_timing, mod_val_accuracies, 'orange', label='full training')
plt.plot(closed_cum_timing, closed_val_accuracies, 'b-', label='GLISTER')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Validation accuracy')
plt.title('Validation Accuracy vs Time ' + data_name + '_' + str(fraction))
plt_file = path_logfile + '_' + str(fraction) + 'val_accuracy_v=VAL.png'
plt.savefig(plt_file)
plt.clf()


print("CRAIG",file=logfile)
print('---------------------------------------------------------------------',file=logfile)


val = "Validation Accuracy,"
tst = "Test Accuracy,"
timing = "Time,"

for i in range(num_epochs):
    timing = timing+"," +str(craig_timing[i])
    val = val+"," +str(craig_valacc[i])
    tst = tst+"," +str(craig_tstacc[i])

print(time,file=logfile)
print(val,file=logfile)
print(tst,file=logfile)
"""
logfile.close()
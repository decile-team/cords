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
from models.set_function_all import SetFunctionFacLoc
from torch.utils.data.sampler import SubsetRandomSampler
# from models.simpleNN_net import ThreeLayerNet
#from models.set_function_craig import DeepSetFunction as CRAIG
from models.set_function_craig import SetFunctionCRAIG_Super_MNIST as CRAIG
from models.set_function_grad_computation_taylor import GlisterSetFunction as SetFunction
from models.set_function_grad_computation_taylor import WeightedSetFunctionLoader as WtSetFunction
import math
from models.mnist_net import MnistNet
from utils.data_utils import load_dataset_pytorch
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler

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

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:0"
print("Using Device:", device)

## Convert to this argparse
datadir = sys.argv[1]
data_name = sys.argv[2]
fraction = float(sys.argv[3])
num_epochs = int(sys.argv[4])
select_every = int(sys.argv[5])
feature = sys.argv[6]# 70
warm_method = 0  # whether to use warmstart-onestep (1) or online (0)
num_runs = 1  # number of random runs
learning_rate = 0.05
all_logs_dir = './results/debugging/' + data_name +'_grad/' + feature +'/' + str(fraction) + '/' + str(select_every)
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
if data_name == 'mnist':
    fullset, valset, testset, num_cls = load_dataset_pytorch(datadir, data_name, feature)
    # Validation Data set is 10% of the Entire Trainset.
    validation_set_fraction = 0.1
    num_fulltrn = len(fullset)
    num_val = int(num_fulltrn * validation_set_fraction)
    num_trn = num_fulltrn - num_val
    trainset, validset = random_split(fullset, [num_trn, num_val])
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

    trainset_idxs = np.array(trainset.indices)
    batch_wise_indices = [trainset_idxs[x] for x in list(BatchSampler(SequentialSampler(trainset_idxs), 1000, drop_last=False))]
    cnt = 0
    for batch_idx in batch_wise_indices:
        inputs = torch.cat([fullset[x][0].view(1, -1) for x in batch_idx],
                           dim=0).type(torch.float)
        targets = torch.tensor([fullset[x][1] for x in batch_idx])
        if cnt == 0:
            x_trn = inputs
            y_trn = targets
            cnt = cnt + 1
        else:
            x_trn = torch.cat([x_trn, inputs], dim=0)
            y_trn = torch.cat([y_trn, targets], dim=0)
            cnt = cnt + 1

    for batch_idx, (inputs, targets) in enumerate(valloader):
        if batch_idx == 0:
            x_val = inputs
            y_val = targets
            x_val_new = inputs.view(val_batch_size, -1)
        else:
            x_val = torch.cat([x_val, inputs], dim=0)
            y_val = torch.cat([y_val, targets], dim=0)
            x_val_new = torch.cat([x_val_new, inputs.view(len(inputs), -1)], dim=0)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == 0:
            x_tst = inputs
            y_tst = targets
            x_tst_new = inputs.view(tst_batch_size, -1)
        else:
            x_tst = torch.cat([x_tst, inputs], dim=0)
            y_tst = torch.cat([y_tst, targets], dim=0)
            x_tst_new = torch.cat([x_tst_new, inputs.view(len(inputs), -1)], dim=0)

write_knndata(datadir, x_trn, y_trn, x_val_new, y_val, x_tst_new, y_tst, data_name)
print('-----------------------------------------')
print(exp_name, str(exp_start_time))
print("Data sizes:", x_trn.shape, x_val.shape, x_tst.shape)
# print(y_trn.shape, y_val.shape, y_tst.shape)

N, M = x_trn.shape
n_val = x_val_new.shape[0]
bud = int(fraction * N)
print("Budget, fraction and N:", bud, fraction, N)
# Transfer all the data to GPU
d_t = time.time()
x_trn, y_trn = x_trn.to('cpu'), y_trn.to('cpu')
x_val, y_val = x_val.to('cpu'), y_val.to('cpu')
print("Transferred data to device in time:", time.time() - d_t)
print_every = 50


def train_model_craig(start_rand_idxs, bud):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #setf_model = CRAIG(device, model, trainset, N_trn=N, batch_size=1000, if_convex=False)
    setf_model = CRAIG(device ,x_trn, y_trn,if_convex)
    print("Starting CRAIG Run")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    for i in range(0, num_epochs):
        print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
        if (i%select_every) == 0:
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            #subset_idxs, gammas = setf_model.lazy_greedy_max(int(bud), clone_dict)
            subset_idxs, gammas = setf_model.class_wise(int(bud), clone_dict)
            model.load_state_dict(cached_state_dict)
            gammas = np.array(gammas)
            idxs = subset_idxs
        #print(gammas)
        print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
        subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                       sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                       pin_memory=True)
        actual_idxs = np.array(trainset.indices)[idxs]
        batch_wise_indices = list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))
        subtrn_loss = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [fullset[actual_idxs[x]][0].view(-1, 1, fullset[actual_idxs[x]][0].shape[1], fullset[actual_idxs[x]][0].shape[2]) for x in batch_idx],
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
        val_loss = 0
        full_trn_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)

    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total
    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


def random_greedy_train_model_online_taylor(start_rand_idxs, bud, lam):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    idxs = start_rand_idxs
    total_idxs = list(np.arange(len(y_trn)))
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = SetFunction(trainset, x_val, y_val, model, criterion,
                             criterion_nored, learning_rate, device, num_cls, 1000)
    print("Starting Randomized Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        actual_idxs = np.array(trainset.indices)[idxs]
        batch_wise_indices = [actual_idxs[x] for x in list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))]
        subtrn_loss = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [fullset[x][0].view(-1, 1, fullset[x][0].shape[1], fullset[x][0].shape[2]) for x in batch_idx],
                dim=0).type(torch.float)
            targets = torch.tensor([fullset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss = 0
        full_trn_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)
        if ((i + 1) % select_every) == 0:
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(lam * bud), clone_dict)
            rem_idxs = list(set(total_idxs).difference(set(subset_idxs)))
            subset_idxs.extend(list(np.random.choice(rem_idxs, size=int((1 - lam) * bud), replace=False)))
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total

    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


def train_model_online(start_rand_idxs, bud):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    idxs = start_rand_idxs

    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = SetFunction(trainset, x_val, y_val, model, criterion,
                             criterion_nored, learning_rate, device, num_cls, 1000)
    print("Starting Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        actual_idxs = np.array(trainset.indices)[idxs]
        batch_wise_indices = [actual_idxs[x] for x in list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))]
        subtrn_loss = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [fullset[x][0].view(-1, 1, fullset[x][0].shape[1], fullset[x][0].shape[2]) for x in batch_idx],
                dim=0).type(torch.float)
            targets = torch.tensor([fullset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True) # targets can have non_blocking=True.
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()
        val_loss = 0
        full_trn_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)
        if ((i + 1) % select_every) == 0:
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            prev_idxs = idxs
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(bud), clone_dict)
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
            print(len(list(set(prev_idxs).difference(set(idxs)))) + len(list(set(idxs).difference(set(prev_idxs)))))
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total
    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


def facloc_reg_train_model_online_taylor(start_rand_idxs, facloc_idxs, bud, lam):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    val_plus_facloc_idxs = [trainset.indices[x] for x in facloc_idxs]
    val_plus_facloc_idxs.extend(validset.indices)
    cmb_set = torch.utils.data.Subset(fullset, val_plus_facloc_idxs)
    cmbloader = torch.utils.data.DataLoader(cmb_set, batch_size=1000,
                                            shuffle=False, pin_memory=True)
    for batch_idx, (inputs, targets) in enumerate(cmbloader):
        if batch_idx == 0:
            x_cmb = inputs
            y_cmb = targets
        else:
            x_cmb = torch.cat([x_cmb, inputs], dim=0)
            y_cmb = torch.cat([y_cmb, targets], dim=0)
    model = model.to(device)
    idxs = start_rand_idxs
    #total_idxs = list(np.arange(len(y_trn)))
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = WtSetFunction(trainset, x_cmb, y_cmb, len(facloc_idxs), lam, model, criterion,
                             criterion_nored, learning_rate, device, num_cls, 1000)
    print("Starting Facloc regularized Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        actual_idxs = np.array(trainset.indices)[idxs]
        batch_wise_indices = [actual_idxs[x] for x in list(BatchSampler(RandomSampler(actual_idxs), trn_batch_size, drop_last=False))]
        subtrn_loss = 0
        for batch_idx in batch_wise_indices:
            inputs = torch.cat(
                [fullset[x][0].view(-1, 1, fullset[x][0].shape[1], fullset[x][0].shape[2]) for x in batch_idx],
                dim=0).type(torch.float)
            targets = torch.tensor([fullset[x][1] for x in batch_idx])
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss = 0
        full_trn_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)
        if ((i + 1) % select_every) == 0:
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(bud), clone_dict)
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            model.load_state_dict(cached_state_dict)
            ### Change the subset_trnloader according to new found indices: subset_idxs
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs), num_workers=1,
                                                           pin_memory=True)
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total

    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


def train_model_mod_online(start_rand_idxs, bud):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    setf_model = SetFunction(trainset, x_val, y_val, model, criterion,
                             criterion_nored, learning_rate, device, 10, 1000)
    print("Starting Modified Greedy Online OneStep Run with taylor!")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #print(batch_idx)
            # targets can have non_blocking=True.
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end)/1000
    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total

    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


def train_model_random(start_rand_idxs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting Random Run")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)

    for i in range(0, num_epochs):
        subtrn_loss = 0
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            # targets can have non_blocking=True.
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss = 0
        full_trn_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end) / 1000

    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total

    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, time


def train_model_random_online(start_rand_idxs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    idxs = start_rand_idxs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting Random Run")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)
    for i in range(0, num_epochs):
        subtrn_loss = 0
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            # targets can have non_blocking=True.
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss = 0
        full_trn_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)

        if ((i + 1) % select_every) == 0:
            idxs = np.random.choice(N, size=bud, replace=False)
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                           sampler=SubsetRandomSampler(idxs),
                                                           pin_memory=True)

    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end) / 1000

    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total

    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total

    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, time


def run_stochastic_Facloc(data, targets, budget):
    model = MnistNet()
    model = model.to(device='cpu')
    approximate_error = 0.01
    per_iter_bud = 10
    num_iterations = int(budget/10)
    facloc_indices = []
    trn_indices = list(np.arange(len(data)))
    sample_size = int(len(data) / num_iterations * math.log(1 / approximate_error))
    #greedy_batch_size = 1200
    for i in range(num_iterations):
        rem_indices = list(set(trn_indices).difference(set(facloc_indices)))
        sub_indices = np.random.choice(rem_indices, size=sample_size, replace=False)
        data_subset = data[sub_indices].cpu()
        targets_subset = targets[sub_indices].cpu()
        train_loader_greedy = []
        train_loader_greedy.append((data_subset, targets_subset))
        setf_model = SetFunctionFacLoc(device, train_loader_greedy)
        idxs = setf_model.lazy_greedy_max(per_iter_bud, model)
        facloc_indices.extend([sub_indices[idx] for idx in idxs])
    return facloc_indices


def train_model_Facloc(idxs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.manual_seed(42)
    np.random.seed(42)
    model = MnistNet()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting Facility Location Run")
    substrn_losses = np.zeros(num_epochs)
    fulltrn_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)

    for i in range(0, num_epochs):
        subtrn_loss = 0
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            # targets can have non_blocking=True.
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss = 0
        full_trn_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                full_trn_loss += loss.item()

        substrn_losses[i] = subtrn_loss
        fulltrn_losses[i] = full_trn_loss
        val_losses[i] = val_loss
        if i % print_every == 0:  # Print Training and Validation Loss
            print('Epoch:', i + 1, 'SubsetTrn,FullTrn,ValLoss:', subtrn_loss, full_trn_loss, val_loss)
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end) / 1000
    subtrn_loss = 0
    subtrn_correct = 0
    subtrn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            subtrn_loss += loss.item()
            _, predicted = outputs.max(1)
            subtrn_total += targets.size(0)
            subtrn_correct += predicted.eq(targets).sum().item()
    subtrn_acc = subtrn_correct / subtrn_total

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = val_correct / val_total

    full_trn_loss = 0
    full_trn_correct = 0
    full_trn_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            full_trn_loss += loss.item()
            _, predicted = outputs.max(1)
            full_trn_total += targets.size(0)
            full_trn_correct += predicted.eq(targets).sum().item()
    full_trn_acc = full_trn_correct / full_trn_total

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    tst_acc = correct / total
    print("SelectionRun---------------------------------")
    print("Final SubsetTrn and FullTrn Loss:", subtrn_loss, full_trn_loss)
    print("Validation Loss and Accuracy:", val_loss, val_acc)
    print("Test Data Loss and Accuracy:", test_loss, tst_acc)
    print('-----------------------------------')
    return val_acc, tst_acc,  subtrn_acc, full_trn_acc, val_loss, test_loss, subtrn_loss, full_trn_loss, val_losses, substrn_losses, fulltrn_losses, idxs, time


start_idxs = np.random.choice(N, size=bud, replace=False)
random_subset_idx = [trainset.indices[x] for x in start_idxs]
## KnnSB selection with Flag = TRN and FLAG = VAL
#knn_idxs_flag_trn = perform_knnsb_selection(datadir, data_name, fraction, selUsing='trn')
#knn_idxs_flag_val = perform_knnsb_selection(datadir, data_name, fraction, selUsing='val')

#facloc_idxs = [trainset.indices[x] for x in knn_idxs_flag_trn]
#facloc_val_idxs = [trainset.indices[x] for x in knn_idxs_flag_val]
# Online algo run
craig_valacc, craig_tstacc, craig_subtrn_acc, craig_full_trn_acc, craig_valloss, \
craig_tstloss,  craig_subtrnloss, craig_full_trn_loss, craig_val_losses, \
craig_substrn_losses, craig_fulltrn_losses, craig_subset_idxs, craig_time= \
    train_model_craig(start_idxs, bud)
print("One step run time: ", craig_time)
craig_subset_idxs = [trainset.indices[x] for x in craig_subset_idxs]


# Online algo run
one_val_valacc, one_val_tstacc, one_val_subtrn_acc, one_val_full_trn_acc, one_val_valloss, \
one_val_tstloss,  one_val_subtrnloss, one_val_full_trn_loss, one_fval_val_losses, \
one_fval_substrn_losses, one_fval_fulltrn_losses, one_subset_idxs, one_step_time= \
    train_model_online(start_idxs, bud)
print("One step run time: ", one_step_time)
one_subset_idxs = [trainset.indices[x] for x in one_subset_idxs]
facloc_idxs = run_stochastic_Facloc(x_trn, y_trn, bud)

# Facility Location OneStep Runs
facloc_reg_val_valacc, facloc_reg_val_tstacc, facloc_reg_val_subtrn_acc, facloc_reg_val_full_trn_acc, facloc_reg_val_valloss, \
facloc_reg_val_tstloss,  facloc_reg_val_subtrnloss, facloc_reg_val_full_trn_loss, facloc_reg_fval_val_losses, \
facloc_reg_fval_substrn_losses, facloc_reg_fval_fulltrn_losses, facloc_reg_subset_idxs, facloc_one_step_time = \
facloc_reg_train_model_online_taylor(start_idxs, facloc_idxs, bud, 100)
print("Facility Location One step run time: ", facloc_one_step_time)
facloc_reg_subset_idxs = [trainset.indices[x] for x in facloc_reg_subset_idxs]

# Modified OneStep Runs
mod_val_valacc, mod_val_tstacc, mod_val_subtrn_acc, mod_val_full_trn_acc, mod_val_valloss, mod_val_tstloss, \
mod_val_subtrnloss, mod_val_full_trn_loss, mod_val_val_losses, mod_val_substrn_losses, mod_val_fulltrn_losses,\
mod_subset_idxs, mod_one_step_time = train_model_mod_online(start_idxs, bud)
print("Mod One Step run time: ", mod_one_step_time)

# Random Run
rand_valacc, rand_tstacc, rand_subtrn_acc, rand_full_trn_acc, rand_valloss, rand_tstloss, rand_subtrnloss,\
rand_full_trn_loss,rand_val_losses, rand_substrn_losses, rand_fulltrn_losses, \
random_run_time = train_model_random(start_idxs)
print("Random Run Time: ", random_run_time)

#Online Random Run
ol_rand_valacc, ol_rand_tstacc, ol_rand_subtrn_acc, ol_rand_full_trn_acc, ol_rand_valloss, ol_rand_tstloss, ol_rand_subtrnloss,\
ol_rand_full_trn_loss, ol_rand_val_losses, ol_rand_substrn_losses, ol_rand_fulltrn_losses, \
ol_random_run_time = train_model_random_online(start_idxs)
print("Online Random Run Time: ", ol_random_run_time)

# Randomized Greedy Taylor OneStep Runs

rand_reg_val_valacc, rand_reg_val_tstacc, rand_reg_val_subtrn_acc, rand_reg_val_full_trn_acc, rand_reg_val_valloss, \
rand_reg_val_tstloss,  rand_reg_val_subtrnloss,rand_reg_val_full_trn_loss, rand_reg_fval_val_losses, \
rand_reg_fval_substrn_losses, rand_reg_fval_fulltrn_losses, rand_reg_subset_idxs, \
random_reg_one_step_time = random_greedy_train_model_online_taylor(start_idxs, bud, 0.9)
print("Random Reg One Step run time: ", random_reg_one_step_time)
rand_reg_subset_idxs = [trainset.indices[x] for x in rand_reg_subset_idxs]

# Facility Location Run
facloc_valacc, facloc_tstacc, facloc_subtrn_acc, facloc_full_tran_acc, facloc_valloss, facloc_tstloss, \
facloc_subtrnloss,facloc_full_trn_loss,facloc_val_losses, facloc_substrn_losses, facloc_fulltrn_losses, \
facloc_idxs, facloc_time = train_model_Facloc(facloc_idxs)
print("Facility location run time: ", facloc_time)
facloc_idxs = [trainset.indices[x] for x in facloc_idxs]

"""
# Facility Location Run
facloc_val_valacc, facloc_val_tstacc, facloc_val_subtrn_acc, facloc_val_full_tran_acc, facloc_val_valloss, facloc_val_tstloss, \
facloc_val_subtrnloss,facloc_val_full_trn_loss,facloc_val_val_losses, facloc_val_substrn_losses, facloc_val_fulltrn_losses, \
facloc_val_idxs, facloc_val_time = train_model_Facloc(facloc_val_idxs)
print("Facility location run time: ", facloc_time)
facloc_idxs = [trainset.indices[x] for x in facloc_idxs]
"""

plot_start_epoch = 0
###### Subset Trn loss with val = VAL #############
plt.figure()
plt.plot(np.arange(plot_start_epoch, num_epochs), one_fval_substrn_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), craig_substrn_losses[plot_start_epoch:], 'r', label='craig')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_substrn_losses[plot_start_epoch:], 'g-', label='random')
plt.plot(np.arange(plot_start_epoch, num_epochs), ol_rand_substrn_losses[plot_start_epoch:], 'g+', label='Online random')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_substrn_losses[plot_start_epoch:], 'pink', label='FacLoc')
#plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_val_substrn_losses[plot_start_epoch:], '#750D86',
#         label='facloc_val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_reg_fval_substrn_losses[plot_start_epoch:], 'k-',
         label='rand_reg_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_fval_substrn_losses[plot_start_epoch:], 'y',
         label='facloc_reg_tay_v=val')



plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Subset trn loss')
plt.title('Subset Training Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'substrn_loss_v=VAL.png'
plt.savefig(plt_file)
plt.clf()

########################################################################
###### Full Trn loss with val = VAL #############
plt.figure()
plt.plot(np.arange(plot_start_epoch, num_epochs), one_fval_fulltrn_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), craig_fulltrn_losses[plot_start_epoch:], 'r', label='craig')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_fulltrn_losses[plot_start_epoch:], 'g-', label='random')
plt.plot(np.arange(plot_start_epoch, num_epochs), ol_rand_fulltrn_losses[plot_start_epoch:], 'g+', label='Online random')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_fulltrn_losses[plot_start_epoch:], 'pink', label='FacLoc')
#plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_val_fulltrn_losses[plot_start_epoch:], '#750D86',
#         label='facloc_val')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_reg_fval_fulltrn_losses[plot_start_epoch:], 'k-',
         label='rand_reg_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_fval_fulltrn_losses[plot_start_epoch:], 'y',
         label='facloc_reg_tay_v=val')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Full trn loss')
plt.title('Full Training Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'fulltrn_loss_v=VAL.png'
plt.savefig(plt_file)
plt.clf()

########################################################################
###### Validation loss with val = VAL #############
plt.figure()
plt.plot(np.arange(plot_start_epoch, num_epochs), one_fval_val_losses[plot_start_epoch:], 'b-', label='tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), craig_val_losses[plot_start_epoch:], 'r', label='craig')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_val_losses[plot_start_epoch:], 'g-', label='random')
plt.plot(np.arange(plot_start_epoch, num_epochs), ol_rand_val_losses[plot_start_epoch:], 'g+', label='Online random')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_val_losses[plot_start_epoch:], 'pink', label='FacLoc')
plt.plot(np.arange(plot_start_epoch, num_epochs), rand_reg_fval_val_losses[plot_start_epoch:], 'k-',
         label='rand_reg_tay_v=val')
plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_reg_fval_val_losses[plot_start_epoch:], 'y',
         label='facloc_reg_tay_v=val')
#plt.plot(np.arange(plot_start_epoch, num_epochs), facloc_val_val_losses[plot_start_epoch:], '#750D86',
#         label='facloc_val')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.title('Validation Loss vs Epochs ' + data_name + '_' + str(fraction) + '_' + 'val=VAL')
plt_file = path_logfile + '_' + str(fraction) + 'valloss_v=VAL.png'
plt.savefig(plt_file)
plt.clf()

print(data_name, ":Budget = ", fraction, file=logfile)
print('---------------------------------------------------------------------', file=logfile)
print('|Algo                            | Val Acc       |   Test Acc       |', file=logfile)
print('| -------------------------------|:-------------:| ----------------:|', file=logfile)
print('*| Facility Location             |', facloc_valacc, '  | ', facloc_tstacc, ' |', file=logfile)
#print('*| Facility Location using val set|', facloc_val_valacc, '  | ', facloc_val_tstacc, ' |', file=logfile)
print('*| Taylor with Validation=VAL     |', one_val_valacc, '  | ', one_val_tstacc, ' |', file=logfile)
print('*| Random Selection               |', rand_valacc, '  | ', rand_tstacc, ' |', file=logfile)
print('*| Online Random Selection               |', ol_rand_valacc, '  | ', ol_rand_tstacc, ' |', file=logfile)
print('*| CRAIG Selection               |', craig_valacc, '  | ', craig_tstacc, ' |', file=logfile)
print('*| Taylor after training               |', mod_val_valacc, '  | ', mod_val_tstacc, ' |', file=logfile)
print('*| random regularized Taylor after training               |', rand_reg_val_valacc, '  | ', rand_reg_val_tstacc, ' |',
      file=logfile)
print('*| facloc regularizec Taylor after training               |', facloc_reg_val_valacc, '  | ',
      facloc_reg_val_tstacc, ' |', file=logfile)
print('---------------------------------------------------', file=logfile)
print('|Algo                            | Run Time       |', file=logfile)
print('| -------------------------------|:-------------:|', file=logfile)
print('*| Facility Location             |', facloc_time, '  | ', file=logfile)
print('*| Taylor with Validation=VAL     |', one_step_time, '  | ', file=logfile)
print('*| Random Selection               |', random_run_time, '  | ',file=logfile)
print('*| Online Random Selection               |', ol_random_run_time, '  | ',file=logfile)
print('*| Taylor after training               |', mod_one_step_time, '  | ', file=logfile)
print('*| random regularized Taylor after training               |', random_reg_one_step_time,' |',
      file=logfile)
print('*| facloc regularizec Taylor after training               |', facloc_one_step_time, '  | ',
       file=logfile)

print("\n", file=logfile)

print("=========Random Results==============", file=logfile)
print("*Rand Validation LOSS:", rand_valloss, file=logfile)
print("*Rand Test Data LOSS:", rand_tstloss, file=logfile)
print("*Rand Full Trn Data LOSS:", rand_fulltrn_losses[-1], file=logfile)

print("=========Online Random Results==============", file=logfile)
print("*Rand Validation LOSS:", ol_rand_valloss, file=logfile)
print("*Rand Test Data LOSS:", ol_rand_tstloss, file=logfile)
print("*Rand Full Trn Data LOSS:", ol_rand_fulltrn_losses[-1], file=logfile)

print("=========FacLoc Results==============", file=logfile)
print("*Facloc Validation LOSS:", facloc_valloss, file=logfile)
print("*Facloc Test Data LOSS:", facloc_tstloss, file=logfile)
print("*Facloc Full Trn Data LOSS:", facloc_fulltrn_losses[-1], file=logfile)

print("=========Online Selection Taylor with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", one_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", one_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", one_fval_fulltrn_losses[-1], file=logfile)

print("=========Random Regularized Online Selection Taylor with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", rand_reg_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", rand_reg_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", rand_reg_fval_fulltrn_losses[-1], file=logfile)

print("=========Facility Location Loss regularized Online Selection Taylor with Validation Set===================",
      file=logfile)
print("*Taylor v=VAL Validation LOSS:", facloc_reg_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", facloc_reg_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", facloc_reg_fval_fulltrn_losses[-1], file=logfile)

print("=========Online Selection Taylor after model training with Validation Set===================", file=logfile)
print("*Taylor v=VAL Validation LOSS:", mod_val_valloss, file=logfile)
print("*Taylor v=VAL Test Data LOSS:", mod_val_tstloss, file=logfile)
print("*Taylor v=VAL Full Trn Data LOSS:", mod_val_fulltrn_losses[-1], file=logfile)
print("=============================================================================================", file=logfile)
print("---------------------------------------------------------------------------------------------", file=logfile)
print("\n", file=logfile)

mod_subset_idxs = list(mod_subset_idxs)
with open(all_logs_dir + '/mod_one_step_subset_selected.txt', 'w') as log_file:
    print(mod_subset_idxs, file=log_file)

subset_idxs = list(one_subset_idxs)
with open(all_logs_dir + '/one_step_subset_selected.txt', 'w') as log_file:
    print(subset_idxs, file=log_file)

rand_subset_idxs = list(rand_reg_subset_idxs)
with open(all_logs_dir + '/rand_reg_one_step_subset_selected.txt', 'w') as log_file:
    print(rand_subset_idxs, file=log_file)

facloc_reg_subset_idxs = list(facloc_reg_subset_idxs)
with open(all_logs_dir + '/facloc_reg_one_step_subset_selected.txt', 'w') as log_file:
    print(facloc_reg_subset_idxs, file=log_file)

random_subset_idx = list(random_subset_idx)
with open(all_logs_dir + '/random_subset_selected.txt', 'w') as log_file:
    print(random_subset_idx, file=log_file)

facloc_idxs = list(facloc_idxs)
with open(all_logs_dir + '/facloc_subset_selected.txt', 'w') as log_file:
    print(facloc_idxs, file=log_file)


craig_subset_idxs = list(craig_subset_idxs)
with open(all_logs_dir + '/craig_subset_selected.txt', 'w') as log_file:
    print(craig_subset_idxs, file=log_file)
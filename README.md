# CORDS: COResets and Data Subset selection

## Outerloop skeleton (PyTorch)

```python
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
#from models.set_function_grad_computation_taylor import GlisterSetFunction as SetFunction, GlisterSetFunction_Closed as ClosedSetFunction
#from models.mnist_net1 import Net
#from models.mnist_net import MnistNet
#from models.resnet import ResNet18
#from utils.custom_dataset import load_mnist_cifar
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler
from torch.autograd import Variable
#from models.set_function_craig import PerClassDeepSetFunction as CRAIG
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


# Data PreProcessing
fullset, valset, testset, num_cls = load_dataset(datadir, data_name, feature)
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


# Model Definition
torch.manual_seed(42)
np.random.seed(42)
model = ResNet18(num_cls)
model = model.to(device)


# loss criterion, optimizer and scheduler definitions
criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Subset Selection Set Function Arguments and Initialization
# All the Set Functions should be overloaded from a base class
setf_name = 'GLISTER'
setf_model = SetFunction(trainset, x_val, y_val, model, criterion,
                             criterion_nored, learning_rate, device, 1, num_cls, 1000)


# Training Loop
for i in range(0, num_epochs):
        subtrn_loss = 0
        subtrn_correct = 0
        subtrn_total = 0
        start_time = time.time()
        if (((i+1) % select_every) == 0):
            print("selEpoch: %d, Starting Selection:" % i, str(datetime.datetime.now()))
            subset_start_time = time.time()
            subset_idxs, grads_idxs = setf_model.naive_greedy_max(int(bud), model.state_dict())
            subset_end_time = time.time() - subset_start_time
            print("Subset Selection Time is:" + str(subset_end_time))
            idxs = subset_idxs
            print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            subset_trnloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
            			shuffle=False, sampler=SubsetRandomSampler(idxs), pin_memory=True)
        model.train()
        #for batch_idx in batch_wise_indices:
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
            #Validation Loss and Accuracy Computation
            for batch_idx, (inputs, targets) in enumerate(valloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
            #Test Loss and Accuracy Computation
            for batch_idx, (inputs, targets) in enumerate(testloader):
                #print(batch_idx)
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += loss.item()
                _, predicted = outputs.max(1)
                tst_total += targets.size(0)
                tst_correct += predicted.eq(targets).sum().item()
            #Train Loss and Accuracy Computation
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
```

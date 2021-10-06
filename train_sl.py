import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from cords.utils.models import *
from cords.utils.data.datasets.SL import gen_dataset
from torch.utils.data import Subset
from cords.utils.config_utils import load_config_data
import os.path as osp
from cords.utils.data._utils import WeightedSubset
from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, OLRandomDataLoader, \
    CRAIGDataLoader, GradMatchDataLoader, RandomDataLoader
from ray import tune


class TrainClassifier:
    def __init__(self, config_file):
        self.config_file = config_file
        self.configdata = load_config_data(self.config_file)
        print()

    """
    ############################## Loss Evaluation ##############################
    """
    def model_eval_loss(self, data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.configdata.train_args.device), \
                                  targets.to(self.configdata.train_args.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    """
    ############################## Model Creation ##############################
    """
    def create_model(self):
        if self.configdata.model.architecture == 'ResNet18':
            model = ResNet18(self.configdata.model.numclasses)
        elif self.configdata.model.architecture == 'MnistNet':
            model = MnistNet()
        elif self.configdata.model.architecture == 'ResNet164':
            model = ResNet164(self.configdata.model.numclasses)
        elif self.configdata.model.architecture == 'MobileNet':
            model = MobileNet(self.configdata.model.numclasses)
        elif self.configdata.model.architecture == 'MobileNetV2':
            model = MobileNetV2(self.configdata.model.numclasses)
        elif self.configdata.model.architecture == 'MobileNet2':
            model = MobileNet2(output_size=self.configdata.model.numclasses)
        elif self.configdata.model.architecture == 'HyperParamNet':
            model = HyperParamNet(self.configdata.model.l1, self.configdata.model.l2)
        model = model.to(self.configdata.train_args.device)
        return model

    """
    ############################## Loss Type, Optimizer and Learning Rate Scheduler ##############################
    """
    def loss_function(self):
        if self.configdata.loss.type == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction='none')
        return criterion, criterion_nored

    def optimizer_with_scheduler(self, model):
        if self.configdata.optimizer.type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.configdata.optimizer.lr,
                                  momentum=self.configdata.optimizer.momentum,
                                  weight_decay=self.configdata.optimizer.weight_decay)
        elif self.configdata.optimizer.type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.configdata.optimizer.lr)
        elif self.configdata.optimizer.type == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.configdata.optimizer.lr)

        if self.configdata.scheduler.type == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.configdata.scheduler.T_max)
        return optimizer, scheduler

    def generate_cumulative_timing(self, mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing / 3600

    def save_ckpt(self, state, ckpt_path):
        torch.save(state, ckpt_path)

    def load_ckp(self, ckpt_path, model, optimizer):
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        metrics = checkpoint['metrics']
        return start_epoch, model, optimizer, loss, metrics

    def train(self):
        """
        ############################## General Training Loop with Data Selection Strategies ##############################
        """
        # Loading the Dataset
        if self.configdata.dataset.feature == 'classimb':
            trainset, validset, testset, num_cls = gen_dataset(self.configdata.dataset.datadir,
                                                               self.configdata.dataset.name,
                                                               self.configdata.dataset.feature,
                                                               classimb_ratio=self.configdata.dataset.classimb_ratio)
        else:
            trainset, validset, testset, num_cls = gen_dataset(self.configdata.dataset.datadir,
                                                               self.configdata.dataset.name,
                                                               self.configdata.dataset.feature)

        N = len(trainset)
        trn_batch_size = self.configdata.dataloader.batch_size
        val_batch_size = self.configdata.dataloader.batch_size
        tst_batch_size = 1000

        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                  shuffle=False, pin_memory=True)

        valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                                shuffle=False, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                                 shuffle=False, pin_memory=True)

        substrn_losses = list()  # np.zeros(configdata['train_args']['num_epochs'])
        trn_losses = list()
        val_losses = list()  # np.zeros(configdata['train_args']['num_epochs'])
        tst_losses = list()
        subtrn_losses = list()
        timing = list()
        trn_acc = list()
        val_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        tst_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        subtrn_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])

        # Results logging file
        print_every = self.configdata.train_args.print_every
        results_dir = osp.abspath(osp.expanduser(self.configdata.train_args.results_dir))
        all_logs_dir = os.path.join(results_dir, self.configdata.dss_args.type,
                                    self.configdata.dataset.name, 
                                    str(self.configdata.dss_args.fraction), 
                                    str(self.configdata.dss_args.select_every))

        os.makedirs(all_logs_dir, exist_ok=True)
        path_logfile = os.path.join(all_logs_dir, self.configdata.dataset.name + '.txt')
        logfile = open(path_logfile, 'w')

        checkpoint_dir = osp.abspath(osp.expanduser(self.configdata.ckpt.dir))
        ckpt_dir = os.path.join(checkpoint_dir, self.configdata.dss_args.type,
                                self.configdata.dataset.name, 
                                str(self.configdata.dss_args.fraction), 
                                str(self.configdata.dss_args.select_every))
        checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model Creation
        model = self.create_model()
        # model1 = self.create_model()

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(model)

        """
        ############################## Custom Dataloader Creation ##############################
        """

        if self.configdata.dss_args.type in ['GradMatch', 'GradMatchPB', 'GradMatch-Warm', 'GradMatchPB-Warm']:
            """
            ############################## GradMatch Dataloader Additional Arguments ##############################
            """
            self.configdata.dss_args.model = model
            self.configdata.dss_args.loss = criterion_nored
            self.configdata.dss_args.eta = self.configdata.optimizer.lr
            self.configdata.dss_args.num_classes = self.configdata.model.numclasses 
            self.configdata.dss_args.num_epochs = self.configdata.train_args.num_epochs
            self.configdata.dss_args.device = self.configdata.train_args.device

            dataloader = GradMatchDataLoader(trainloader, valloader, self.configdata.dss_args, verbose=True, 
                                             batch_size=self.configdata.dataloader.batch_size, 
                                             shuffle=self.configdata.dataloader.shuffle,
                                             pin_memory=self.configdata.dataloader.pin_memory)

        
        elif self.configdata.dss_args.type in ['GLISTER', 'GLISTER-Warm', 'GLISTERPB', 'GLISTERPB-Warm']:
            """
            ############################## GLISTER Dataloader Additional Arguments ##############################
            """
            self.configdata.dss_args.model = model
            self.configdata.dss_args.loss = criterion_nored
            self.configdata.dss_args.eta = self.configdata.optimizer.lr
            self.configdata.dss_args.num_classes = self.configdata.model.numclasses 
            self.configdata.dss_args.num_epochs = self.configdata.train_args.num_epochs
            self.configdata.dss_args.device = self.configdata.train_args.device

            dataloader = GLISTERDataLoader(trainloader, valloader, self.configdata.dss_args, verbose=True, 
                                             batch_size=self.configdata.dataloader.batch_size, 
                                             shuffle=self.configdata.dataloader.shuffle,
                                             pin_memory=self.configdata.dataloader.pin_memory)

        
        elif self.configdata.dss_args.type in ['CRAIG', 'CRAIG-Warm', 'CRAIGPB', 'CRAIGPB-Warm']:
            """
            ############################## CRAIG Dataloader Additional Arguments ##############################
            """
            self.configdata.dss_args.model = model
            self.configdata.dss_args.loss = criterion_nored
            self.configdata.dss_args.num_classes = self.configdata.model.numclasses 
            self.configdata.dss_args.num_epochs = self.configdata.train_args.num_epochs
            self.configdata.dss_args.device = self.configdata.train_args.device

            dataloader = CRAIGDataLoader(trainloader, valloader, self.configdata.dss_args, verbose=True, 
                                             batch_size=self.configdata.dataloader.batch_size,
                                             shuffle=self.configdata.dataloader.shuffle,
                                             pin_memory=self.configdata.dataloader.pin_memory)

    
        elif self.configdata.dss_args.type in ['Random', 'Random-Warm']:
            """
            ############################## Random Dataloader Additional Arguments ##############################
            """
            self.configdata.dss_args.device = self.configdata.train_args.device
            self.configdata.dss_args.num_epochs = self.configdata.train_args.num_epochs

            dataloader = RandomDataLoader(trainloader, self.configdata.dss_args, verbose=True,
                                         batch_size=self.configdata.dataloader.batch_size,
                                         shuffle=self.configdata.dataloader.shuffle,
                                         pin_memory=self.configdata.dataloader.pin_memory)
           

        elif self.configdata.dss_args.type == ['OLRandom', 'OLRandom-Warm']:
            """
            ############################## OLRandom Dataloader Additional Arguments ##############################
            """
            self.configdata.dss_args.device = self.configdata.train_args.device
            self.configdata.dss_args.num_epochs = self.configdata.train_args.num_epochs

            dataloader = OLRandomDataLoader(trainloader, self.configdata.dss_args, verbose=True,
                                         batch_size=self.configdata.dataloader.batch_size,
                                         shuffle=self.configdata.dataloader.shuffle,
                                         pin_memory=self.configdata.dataloader.pin_memory)
        
        elif self.configdata.dss_args.type == 'Full':
            ############################## Full Dataloader Additional Arguments ##############################
            wt_trainset = WeightedSubset(trainset, list(range(len(trainset))), [1]*len(trainset))

            dataloader = torch.utils.data.DataLoader(wt_trainset,
                                         batch_size=self.configdata.dataloader.batch_size,
                                         shuffle=self.configdata.dataloader.shuffle,
                                         pin_memory=self.configdata.dataloader.pin_memory)

        print("=======================================", file=logfile)

        """
        ################################################# Checkpoint Loading #################################################
        """

        if self.configdata.ckpt.is_load == True:
            start_epoch, model, optimizer, ckpt_loss, load_metrics = self.load_ckp(checkpoint_path, model, optimizer)
            print("Loading saved checkpoint model at epoch " + str(start_epoch))
            for arg in load_metrics.keys():
                if arg == "val_loss":
                    val_losses = load_metrics['val_loss']
                if arg == "val_acc":
                    val_acc = load_metrics['val_acc']
                if arg == "tst_loss":
                    tst_losses = load_metrics['tst_loss']
                if arg == "tst_acc":
                    tst_acc = load_metrics['tst_acc']
                if arg == "trn_loss":
                    trn_losses = load_metrics['trn_loss']
                if arg == "trn_acc":
                    trn_acc = load_metrics['trn_acc']
                if arg == "subtrn_loss":
                    subtrn_losses = load_metrics['subtrn_loss']
                if arg == "subtrn_acc":
                    subtrn_acc = load_metrics['subtrn_acc']
                if arg == "time":
                    timing = load_metrics['time']
        else:
            start_epoch = 0

        """
        ################################################# Training Loop #################################################
        """

        for epoch in range(start_epoch, self.configdata.train_args.num_epochs):
            subtrn_loss = 0
            subtrn_correct = 0
            subtrn_total = 0
            model.train()
            start_time = time.time()
            for _, (inputs, targets, weights) in enumerate(dataloader):
                inputs = inputs.to(self.configdata.train_args.device)
                targets = targets.to(self.configdata.train_args.device, non_blocking=True)
                weights = weights.to(self.configdata.train_args.device)  
                optimizer.zero_grad()
                outputs = model(inputs)
                losses = criterion_nored(outputs, targets)
                loss = torch.dot(losses, weights/(weights.sum()))
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            epoch_time = time.time() - start_time
            scheduler.step()
            timing.append(epoch_time)
            print_args = self.configdata.train_args.print_args
            
            """
            ################################################# Evaluation Loop #################################################
            """

            if ((epoch + 1) % self.configdata.train_args.print_every == 0):
                trn_loss = 0
                trn_correct = 0
                trn_total = 0
                val_loss = 0
                val_correct = 0
                val_total = 0
                tst_correct = 0
                tst_total = 0
                tst_loss = 0
                model.eval()

                if (("trn_loss" in print_args) or ("trn_acc" in print_args)):
                    with torch.no_grad():
                        for _, (inputs, targets) in enumerate(trainloader):
                            inputs, targets = inputs.to(self.configdata.train_args.device),  \
                                              targets.to(self.configdata.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            trn_loss += loss.item()
                            if "trn_acc" in print_args:
                                _, predicted = outputs.max(1)
                                trn_total += targets.size(0)
                                trn_correct += predicted.eq(targets).sum().item()
                        trn_losses.append(trn_loss)
                        
                    if "trn_acc" in print_args:
                        trn_acc.append(trn_correct / trn_total)

                if (("val_loss" in print_args) or ("val_acc" in print_args)):
                    with torch.no_grad():
                        for _, (inputs, targets) in enumerate(valloader):
                            inputs, targets = inputs.to(self.configdata.train_args.device), \
                                              targets.to(self.configdata.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                            if "val_acc" in print_args:
                                _, predicted = outputs.max(1)
                                val_total += targets.size(0)
                                val_correct += predicted.eq(targets).sum().item()
                        val_losses.append(val_loss)

                    if "val_acc" in print_args:
                        val_acc.append(val_correct / val_total)

                if (("tst_loss" in print_args) or ("tst_acc" in print_args)):
                    with torch.no_grad():
                        for _, (inputs, targets) in enumerate(testloader):
                            inputs, targets = inputs.to(self.configdata.train_args.device),  \
                                              targets.to(self.configdata.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            tst_loss += loss.item()
                            if "tst_acc" in print_args:
                                _, predicted = outputs.max(1)
                                tst_total += targets.size(0)
                                tst_correct += predicted.eq(targets).sum().item()
                        tst_losses.append(tst_loss)
                    
                    if "tst_acc" in print_args:
                        tst_acc.append(tst_correct / tst_total)

                if "subtrn_acc" in print_args:
                    subtrn_acc.append(subtrn_correct / subtrn_total)

                if "subtrn_losses" in print_args:
                    subtrn_losses.append(subtrn_loss)

                print_str = "Epoch: " + str(epoch + 1)

                """
                ################################################# Results Printing #################################################
                """

                for arg in print_args:

                    if arg == "val_loss":
                        print_str += " , " + "Validation Loss: " + str(val_losses[-1])

                    if arg == "val_acc":
                        print_str += " , " + "Validation Accuracy: " + str(val_acc[-1])

                    if arg == "tst_loss":
                        print_str += " , " + "Test Loss: " + str(tst_losses[-1])

                    if arg == "tst_acc":
                        print_str += " , " + "Test Accuracy: " + str(tst_acc[-1])

                    if arg == "trn_loss":
                        print_str += " , " + "Training Loss: " + str(trn_losses[-1])

                    if arg == "trn_acc":
                        print_str += " , " + "Training Accuracy: " + str(trn_acc[-1])

                    if arg == "subtrn_loss":
                        print_str += " , " + "Subset Loss: " + str(subtrn_losses[-1])

                    if arg == "subtrn_acc":
                        print_str += " , " + "Subset Accuracy: " + str(subtrn_acc[-1])

                    if arg == "time":
                        print_str += " , " + "Timing: " + str(timing[-1])

                # report metric to ray for hyperparameter optimization
                if 'report_tune' in self.configdata and self.configdata.report_tune:
                    tune.report(mean_accuracy=val_acc[-1])

                print(print_str)

            ################################################# Checkpoint Saving #################################################
            if ((epoch + 1) % self.configdata.ckpt.save_every == 0) and self.configdata.ckpt.is_save == True:

                metric_dict = {}

                for arg in print_args:
                    if arg == "val_loss":
                        metric_dict['val_loss'] = val_losses
                    if arg == "val_acc":
                        metric_dict['val_acc'] = val_acc
                    if arg == "tst_loss":
                        metric_dict['tst_loss'] = tst_losses
                    if arg == "tst_acc":
                        metric_dict['tst_acc'] = tst_acc
                    if arg == "trn_loss":
                        metric_dict['trn_loss'] = trn_losses
                    if arg == "trn_acc":
                        metric_dict['trn_acc'] = trn_acc
                    if arg == "subtrn_loss":
                        metric_dict['subtrn_loss'] = subtrn_losses
                    if arg == "subtrn_acc":
                        metric_dict['subtrn_acc'] = subtrn_acc
                    if arg == "time":
                        metric_dict['time'] = timing

                ckpt_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': self.loss_function(),
                    'metrics': metric_dict
                }

                # save checkpoint
                self.save_ckpt(ckpt_state, checkpoint_path)
                print("Model checkpoint saved at epoch " + str(epoch + 1))

        """
        ################################################# Results Summary #################################################
        """

        print(self.configdata.dss_args.type + " Selection Run---------------------------------")
        print("Final SubsetTrn:", subtrn_loss)
        if "val_loss" in print_args:
            if "val_acc" in print_args:
                print("Validation Loss and Accuracy: ", val_loss, val_acc[-1])
            else:
                print("Validation Loss: ", val_loss)

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                print("Test Data Loss and Accuracy: ", tst_loss, tst_acc[-1])
            else:
                print("Test Data Loss: ", tst_loss)
        print('-----------------------------------')
        print(self.configdata.dss_args.type, file=logfile)
        print('---------------------------------------------------------------------', file=logfile)

        """
        ################################################# Final Results Logging #################################################
        """

        if "val_acc" in print_args:
            val_str = "Validation Accuracy, "
            for val in val_acc:
                val_str = val_str + " , " + str(val)
            print(val_str, file=logfile)

        if "tst_acc" in print_args:
            tst_str = "Test Accuracy, "
            for tst in tst_acc:
                tst_str = tst_str + " , " + str(tst)
            print(tst_str, file=logfile)

        if "time" in print_args:
            time_str = "Time, "
            for t in timing:
                time_str = time_str + " , " + str(t)
            print(timing, file=logfile)

        omp_timing = np.array(timing)
        omp_cum_timing = list(self.generate_cumulative_timing(omp_timing))
        print("Total time taken by " + self.configdata.dss_args.type + " = " + str(omp_cum_timing[-1]))
        logfile.close()        

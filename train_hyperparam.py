import time
import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from ray.tune import Trainable
from torch.utils.data.sampler import SubsetRandomSampler
from cords.utils.models import *
from cords.utils.custom_dataset import load_dataset_custom
from torch.utils.data import Subset
from cords.utils.config_utils import load_config_data
import os.path as osp
from cords.selectionstrategies.supervisedlearning import OMPGradMatchStrategy, GLISTERStrategy, RandomStrategy, \
    CRAIGStrategy
from ray import tune

from cords.utils.models.simpleNN_net import FourLayerNet


class SSLTrainClassifier(Trainable):
    def __init__(self, config_file):
        self.config_file = config_file
        self.configdata = load_config_data(self.config_file)

    """
    Loss Evaluation
    """

    def model_eval_loss(self, data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                    self.configdata['train_args']['device'], non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    """
    #Model Creation
    """

    def create_model(self):

        if self.configdata['model']['architecture'] == 'ResNet18':
            model = ResNet18(self.configdata['model']['numclasses'])
        elif self.configdata['model']['architecture'] == 'MnistNet':
            model = MnistNet()
        elif self.configdata['model']['architecture'] == 'ResNet164':
            model = ResNet164(self.configdata['model']['numclasses'])
        elif self.configdata['model']['architecture'] == 'MobileNet':
            model = MobileNet(self.configdata['model']['numclasses'])
        elif self.configdata['model']['architecture'] == 'MobileNetV2':
            model = MobileNetV2(self.configdata['model']['numclasses'])
        elif self.configdata['model']['architecture'] == 'MobileNet2':
            model = MobileNet2(output_size=self.configdata['model']['numclasses'])
        elif self.configdata['model']['architecture'] == 'TwoLayerNet':
            input_dim = self.configdata["model"]["input_dim"]
            num_classes = self.configdata["model"]["numclasses"]
            hidden_units = self.configdata["model"]["hidden_units"]
            model = TwoLayerNet(input_dim, num_classes, hidden_units)
        elif self.configdata['model']['architecture'] == 'FourLayerNet':
            input_dim = self.configdata["model"]["input_dim"]
            num_classes = self.configdata["model"]["numclasses"]
            h1 = self.configdata["model"]["hidden_units1"]
            h2 = self.configdata["model"]["hidden_units2"]
            h3 = self.configdata["model"]["hidden_units3"]
            # model = FourLayerNet(input_dim, num_classes, hidden_units)
            model = FourLayerNet(input_dim, num_classes, h1, h2, h3)
        elif self.configdata['model']['architecture'] == 'LSTM':
            # vocab_size = self.configdata["model"]["vocab_size"]
            vocab_size = self.configdata["model"]["input_dim"]
            hidden_units = self.configdata["model"]["hidden_units"]
            num_layers = self.configdata["model"]["num_layers"]
            embed_dim = self.configdata["model"]["embed_dim"]
            num_classes = self.configdata["model"]["numclasses"]
            dataset_embedding = {"twitter": "glove_twitter_200", "corona": "glove_twitter_200",
                                 "news": "glove.6B.300d", "ag_news": "glove.6B.200d"}
            model = LSTMModel(vocab_size, hidden_units, num_layers, embed_dim, num_classes,
                              pretrained_embedding=dataset_embedding[self.configdata['dataset']['name']],
                              dataset=self.configdata['dataset']['name'])
        else:
            raise Exception("model %s does not exist. " % self.configdata['model']['architecture'])
        model = model.to(self.configdata['train_args']['device'])
        return model

    """#Loss Type, Optimizer and Learning Rate Scheduler"""

    def loss_function(self):
        if self.configdata['loss']['type'] == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction='none')
        return criterion, criterion_nored

    def optimizer_with_scheduler(self, model):

        if self.configdata['optimizer']['type'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.configdata['optimizer']['lr'],
                                  momentum=self.configdata['optimizer']['momentum'],
                                  weight_decay=self.configdata['optimizer']['weight_decay'], nesterov=True)
        elif self.configdata['optimizer']['type'] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.configdata['optimizer']['lr'])
        elif self.configdata['optimizer']['type'] == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.configdata['optimizer']['lr'])

        if self.configdata['scheduler']['type'] == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.configdata['scheduler']['T_max'])
        elif self.configdata['scheduler']['type'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.configdata['scheduler']['step_size'])
        else:
            scheduler = None
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

    def setup(self):
        self.timestep = 0

        if self.configdata['dataset']['feature'] == 'classimb':
            self.trainset, self.validset, self.testset, self.num_cls = load_dataset_custom(
                self.configdata['dataset']['datadir'],
                self.configdata['dataset']['name'],
                self.configdata['dataset']['feature'],
                classimb_ratio=self.configdata['dataset'][
                    'classimb_ratio'])
        else:
            self.trainset, self.validset, self.testset, self.num_cls = load_dataset_custom(
                self.configdata['dataset']['datadir'],
                self.configdata['dataset']['name'],
                self.configdata['dataset']['feature'])

        self.N = len(self.trainset)

        self.trn_batch_size = self.configdata["dataloader"]["batch_size"]

        self.val_batch_size = 1000
        self.tst_batch_size = 1000

        self.train_start_time = round(time.time())
        print("train_start_time: %s" % self.train_start_time)

        # print("--------------------------------------------------------")
        print("trn_batch_size: %s. " % self.trn_batch_size)
        print("Using dss_strategy: %s" % self.configdata["dss_strategy"]["type"])
        # print("--------------------------------------------------------")

        # Creating the Data Loaders
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.trn_batch_size,
                                                       shuffle=False, pin_memory=True)

        self.valloader = torch.utils.data.DataLoader(self.validset, batch_size=self.val_batch_size,
                                                     shuffle=False, pin_memory=True)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.tst_batch_size,
                                                      shuffle=False, pin_memory=True)

        # Budget for subset selection
        bud = int(self.configdata['dss_strategy']['fraction'] * self.N)
        print("Budget, fraction and N:", self.bud, self.configdata['dss_strategy']['fraction'], self.N)

        # Subset Selection and creating the subset data loader
        self.start_idxs = np.random.choice(self.N, size=self.bud, replace=False)
        self.idxs = self.start_idxs
        self.data_sub = Subset(self.trainset, self.idxs)
        self.subset_trnloader = torch.utils.data.DataLoader(self.data_sub,
                                                            batch_size=self.configdata['dataloader']['batch_size'],
                                                            shuffle=self.configdata['dataloader']['shuffle'],
                                                            pin_memory=self.configdata['dataloader']['pin_memory'])

        # Variables to store accuracies
        self.gammas = torch.ones(len(self.idxs)).to(self.configdata['train_args']['device'])
        self.substrn_losses = list()  # np.zeros(configdata['train_args']['num_epochs'])
        self.trn_losses = list()
        self.val_losses = list()  # np.zeros(configdata['train_args']['num_epochs'])
        self.tst_losses = list()
        self.subtrn_losses = list()
        self.timing = list()
        self.trn_acc = list()
        self.val_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        self.tst_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        self.subtrn_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])

        # Results logging file
        self.print_every = self.configdata['train_args']['print_every']
        self.results_dir = osp.abspath(osp.expanduser(self.configdata['train_args']['results_dir']))
        # all_logs_dir = os.path.join(results_dir, self.configdata['dss_strategy']['type'],
        #                             self.configdata['dataset']['name'], str(
        #         self.configdata['dss_strategy']['fraction']), str(self.configdata['dss_strategy']['select_every']))
        self.all_logs_dir = os.path.join(self.results_dir, self.configdata['dss_strategy']['type'],
                                         self.configdata['dataset']['name'],
                                         str(self.configdata['dss_strategy']['fraction']),
                                         str(self.configdata['dss_strategy']['select_every']),
                                         str(self.train_start_time))

        os.makedirs(self.all_logs_dir, exist_ok=True)
        self.path_logfile = os.path.join(self.all_logs_dir, self.configdata['dataset']['name'] + '.txt')
        self.logfile = open(self.path_logfile, 'w')

        self.checkpoint_dir = osp.abspath(osp.expanduser(self.configdata['ckpt']['dir']))
        if "save_dir" in self.configdata:
            self.ckpt_dir = os.path.join(self.checkpoint_dir, self.configdata["save_dir"])
            self.checkpoint_path = os.path.join(self.ckpt_dir, 'model.pt')
        else:
            self.ckpt_dir = os.path.join(self.checkpoint_dir, self.configdata['dss_strategy']['type'],
                                         self.configdata['dataset']['name'], str(
                    self.configdata['dss_strategy']['fraction']), str(self.configdata['dss_strategy']['select_every']),
                                         str(self.train_start_time))
            checkpoint_path = os.path.join(self.ckpt_dir, 'model.pt')
        print("making dir: %s" % self.ckpt_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Model Creation
        self.model = self.create_model()
        self.model1 = self.create_model()

        self.n_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        print(
            "Training model: %s, number of parameters: %d" % (self.configdata['model']['architecture'], self.n_params))

        # Loss Functions
        self.criterion, self.criterion_nored = self.loss_function()

        # Getting the optimizer and scheduler
        self.optimizer, self.scheduler = self.optimizer_with_scheduler(self.model)

        if self.configdata['dss_strategy']['type'] == 'GradMatch':
            # OMPGradMatch Selection strategy
            self.setf_model = OMPGradMatchStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                                   self.configdata['optimizer']['lr'],
                                                   self.configdata['train_args']['device'], self.num_cls, True,
                                                   'PerClassPerGradient',
                                                   valid=self.configdata['dss_strategy']['valid'],
                                                   lam=self.configdata['dss_strategy']['lam'], eps=1e-100)
        elif self.configdata['dss_strategy']['type'] == 'GradMatchPB':
            self.setf_model = OMPGradMatchStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                                   self.configdata['optimizer']['lr'],
                                                   self.configdata['train_args']['device'], self.num_cls, True,
                                                   'PerBatch',
                                                   valid=self.configdata['dss_strategy']['valid'],
                                                   lam=self.configdata['dss_strategy']['lam'], eps=1e-100)
        elif self.configdata['dss_strategy']['type'] == 'GLISTER':
            # GLISTER Selection strategy
            self.setf_model = GLISTERStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'],
                                              self.num_cls, True, 'Supervised', 'Stochastic', r=int(bud))
        elif self.configdata['dss_strategy']['type'] == 'GLISTERPB':
            self.setf_model = GLISTERStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'],
                                              self.num_cls, True, 'PerBatch', 'Stochastic', r=int(bud))
        elif self.configdata['dss_strategy']['type'] == 'CRAIG':
            # CRAIG Selection strategy
            self.setf_model = CRAIGStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                            self.configdata['train_args']['device'], self.num_cls, False, False,
                                            'PerClass')

        elif self.configdata['dss_strategy']['type'] == 'CRAIGPB':
            # CRAIG Selection strategy
            self.setf_model = CRAIGStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                            self.configdata['train_args']['device'], self.num_cls, False, False,
                                            'PerBatch')

        elif self.configdata['dss_strategy']['type'] == 'CRAIG-Warm':
            # CRAIG Selection strategy
            self.setf_model = CRAIGStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                            self.configdata['train_args']['device'], self.num_cls, False, False,
                                            'PerClass')
            # Random-Online Selection strategy
            # rand_setf_model = RandomStrategy(trainloader, online=True)
            if 'kappa' in self.configdata['dss_strategy']:
                self.kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                self.full_epochs = round(self.kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'CRAIGPB-Warm':
            # CRAIG Selection strategy
            self.setf_model = CRAIGStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                            self.configdata['train_args']['device'], self.num_cls, False, False,
                                            'PerBatch')
            # Random-Online Selection strategy
            # rand_setf_model = RandomStrategy(trainloader, online=True)
            if 'kappa' in self.configdata['dss_strategy']:
                self.kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                self.full_epochs = round(self.kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'Random':
            # Random Selection strategy
            self.setf_model = RandomStrategy(self.trainloader, online=False)

        elif self.configdata['dss_strategy']['type'] == 'Random-Online':
            # Random-Online Selection strategy
            self.setf_model = RandomStrategy(self.trainloader, online=True)

        elif self.configdata['dss_strategy']['type'] == 'GLISTER-Warm':
            # GLISTER Selection strategy
            self.setf_model = GLISTERStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'],
                                              self.num_cls, True, 'Supervised', 'Stochastic', r=int(self.bud))
            # Random-Online Selection strategy
            if 'kappa' in self.configdata['dss_strategy']:
                self.kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                self.full_epochs = round(self.kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'GradMatch-Warm':
            # OMPGradMatch Selection strategy
            self.setf_model = OMPGradMatchStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                                   self.configdata['optimizer']['lr'],
                                                   self.configdata['train_args']['device'],
                                                   self.num_cls, True, 'PerClassPerGradient',
                                                   valid=self.configdata['dss_strategy']['valid'],
                                                   lam=self.configdata['dss_strategy']['lam'], eps=1e-100)
            # Random-Online Selection strategy
            if 'kappa' in self.configdata['dss_strategy']:
                self.kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                self.full_epochs = round(self.kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'GradMatchPB-Warm':
            # OMPGradMatch Selection strategy
            self.setf_model = OMPGradMatchStrategy(self.trainloader, self.valloader, self.model1, self.criterion_nored,
                                                   self.configdata['optimizer']['lr'],
                                                   self.configdata['train_args']['device'],
                                                   self.num_cls, True, 'PerBatch',
                                                   valid=self.configdata['dss_strategy']['valid'],
                                                   lam=self.configdata['dss_strategy']['lam'], eps=1e-100)
            # Random-Online Selection strategy
            if 'kappa' in self.configdata['dss_strategy']:
                self.kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                self.full_epochs = round(self.kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'Random-Warm':
            if 'kappa' in self.configdata['dss_strategy']:
                self.kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                self.full_epochs = round(self.kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        print("=======================================", file=self.logfile)

        if self.configdata['ckpt']['is_load'] == True:
            self.start_epoch, self.model, self.optimizer, self.ckpt_loss, self.load_metrics = self.load_ckp(
                self.checkpoint_path, self.model, self.optimizer)
            print("Loading saved checkpoint model at epoch " + str(self.start_epoch))
            for arg in self.load_metrics.keys():
                if arg == "val_loss":
                    self.val_losses = self.load_metrics['val_loss']
                if arg == "val_acc":
                    self.val_acc = self.load_metrics['val_acc']
                if arg == "tst_loss":
                    self.tst_losses = self.load_metrics['tst_loss']
                if arg == "tst_acc":
                    self.tst_acc = self.load_metrics['tst_acc']
                if arg == "trn_loss":
                    self.trn_losses = self.load_metrics['trn_loss']
                if arg == "trn_acc":
                    self.trn_acc = self.load_metrics['trn_acc']
                if arg == "subtrn_loss":
                    self.subtrn_losses = self.load_metrics['subtrn_loss']
                if arg == "subtrn_acc":
                    self.subtrn_acc = self.load_metrics['subtrn_acc']
                if arg == "time":
                    self.timing = self.load_metrics['time']
        else:
            self.start_epoch = 0

    def step(self):

        subtrn_loss = 0
        subtrn_correct = 0
        subtrn_total = 0
        subset_selection_time = 0

        if self.configdata['dss_strategy']['type'] in ['Random-Online']:
            start_time = time.time()
            subset_idxs, gammas = self.setf_model.select(int(self.bud))
            idxs = subset_idxs
            subset_selection_time += (time.time() - start_time)
            gammas = gammas.to(self.configdata['train_args']['device'])

        elif self.configdata['dss_strategy']['type'] in ['Random']:
            pass

        elif (self.configdata['dss_strategy']['type'] in ['GLISTER', 'GLISTERPB', 'GradMatch',
                                                          'GradMatchPB', 'CRAIG', 'CRAIGPB', 'R-GLISTER']) and (
                ((self.timestep + 1) % self.configdata['dss_strategy']['select_every']) == 0):
            start_time = time.time()
            cached_state_dict = copy.deepcopy(self.model.state_dict())
            clone_dict = copy.deepcopy(self.model.state_dict())
            subset_idxs, gammas = self.setf_model.select(int(self.bud), clone_dict)
            self.model.load_state_dict(cached_state_dict)
            idxs = subset_idxs
            if self.configdata['dss_strategy']['type'] in ['GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']:
                gammas = torch.from_numpy(np.array(gammas)).to(self.configdata['train_args']['device']).to(
                    torch.float32)
            subset_selection_time += (time.time() - start_time)

        elif (self.configdata['dss_strategy']['type'] in ['GLISTER-Warm', 'GradMatch-Warm', 'GradMatchPB-Warm',
                                                          'CRAIG-Warm',
                                                          'CRAIGPB-Warm']):
            start_time = time.time()
            if ((self.timestep % self.configdata['dss_strategy']['select_every'] == 0) and (
                    self.timestep >= self.kappa_epochs)):
                cached_state_dict = copy.deepcopy(self.model.state_dict())
                clone_dict = copy.deepcopy(self.model.state_dict())
                subset_idxs, gammas = self.setf_model.select(int(self.bud), clone_dict)
                self.model.load_state_dict(cached_state_dict)
                idxs = subset_idxs
                if self.configdata['dss_strategy']['type'] in ['GradMatch-Warm', 'GradMatchPB-Warm', 'CRAIG-Warm',
                                                               'CRAIGPB-Warm']:
                    gammas = torch.from_numpy(np.array(gammas)).to(self.configdata['train_args']['device']).to(
                        torch.float32)
            subset_selection_time += (time.time() - start_time)

        elif self.configdata['dss_strategy']['type'] in ['Random-Warm']:
            pass

        data_sub = Subset(self.trainset, idxs)
        subset_trnloader = torch.utils.data.DataLoader(data_sub, batch_size=self.trn_batch_size, shuffle=False,
                                                       pin_memory=True)

        self.model.train()
        # print(model)
        batch_wise_indices = list(subset_trnloader.batch_sampler)
        if self.configdata['dss_strategy']['type'] in ['CRAIG', 'CRAIGPB', 'GradMatch', 'GradMatchPB']:
            start_time = time.time()
            total_targets = []
            total_preds = []
            for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                    self.configdata['train_args']['device'],
                    non_blocking=True)  # targets can have non_blocking=True.
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                losses = self.criterion_nored(outputs, targets)
                loss = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (
                    gammas[batch_wise_indices[batch_idx]].sum())
                loss.backward()
                subtrn_loss += loss.item()
                self.optimizer.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        elif self.configdata['dss_strategy']['type'] in ['CRAIGPB-Warm', 'CRAIG-Warm', 'GradMatch-Warm',
                                                         'GradMatchPB-Warm']:
            start_time = time.time()
            if self.timestep < self.full_epochs:
                for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'],
                        non_blocking=True)  # targets can have non_blocking=True.
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    subtrn_loss += loss.item()
                    self.optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()

            elif self.timestep >= self.kappa_epochs:
                for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'],
                        non_blocking=True)  # targets can have non_blocking=True.
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    losses = self.criterion_nored(outputs, targets)
                    loss = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (
                        gammas[batch_wise_indices[batch_idx]].sum())
                    loss.backward()
                    subtrn_loss += loss.item()
                    self.optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        elif self.configdata['dss_strategy']['type'] in ['GLISTER', 'GLISTERPB', 'Random', 'Random-Online',
                                                         'R-GLISTER']:
            start_time = time.time()
            print("len(subset_trnloader): %s" % len(subset_trnloader))
            for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                    self.configdata['train_args']['device'],
                    non_blocking=True)  # targets can have non_blocking=True.
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                subtrn_loss += loss.item()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        elif self.configdata['dss_strategy']['type'] in ['GLISTER-Warm', 'Random-Warm']:
            start_time = time.time()
            if self.timestep < self.full_epochs:
                for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'],
                        non_blocking=True)  # targets can have non_blocking=True.
                    self.optimizer.zero_grad()
                    # print(model.linear1.weight.dtype)
                    # print(inputs.dtype)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    subtrn_loss += loss.item()
                    self.optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
            elif self.timestep >= self.kappa_epochs:
                for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'],
                        non_blocking=True)  # targets can have non_blocking=True.
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    subtrn_loss += loss.item()
                    self.optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        elif self.configdata['dss_strategy']['type'] in ['Full']:
            start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                    self.configdata['train_args']['device'],
                    non_blocking=True)  # targets can have non_blocking=True.
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                subtrn_loss += loss.item()
                self.optimizer.step()
                _, predicted = outputs.max(1)
                subtrn_total += targets.size(0)
                subtrn_correct += predicted.eq(targets).sum().item()
            train_time = time.time() - start_time

        self.scheduler.step()
        self.timing.append(train_time + subset_selection_time)
        print_args = self.configdata['train_args']['print_args']

        if ((self.timestep + 1) % self.configdata['train_args']['print_every'] == 0):
            trn_loss = 0
            trn_correct = 0
            trn_total = 0
            val_loss = 0
            val_correct = 0
            val_total = 0
            tst_correct = 0
            tst_total = 0
            tst_loss = 0
            self.model.eval()

            if (("trn_loss" in print_args) or ("trn_acc" in print_args)):
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                        # print(batch_idx)
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'], non_blocking=True)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        trn_loss += loss.item()
                        self.trn_losses.append(trn_loss)
                        if "trn_acc" in print_args:
                            _, predicted = outputs.max(1)
                            trn_total += targets.size(0)
                            trn_correct += predicted.eq(targets).sum().item()

                if "trn_acc" in print_args:
                    self.trn_acc.append(trn_correct / trn_total)

            if (("val_loss" in print_args) or ("val_acc" in print_args)):
                with torch.no_grad():
                    total_targets = []
                    total_preds = []
                    for batch_idx, (inputs, targets) in enumerate(self.valloader):
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'], non_blocking=True)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        val_loss += loss.item()
                        if "val_acc" in print_args:
                            _, predicted = outputs.max(1)
                            val_total += targets.size(0)
                            val_correct += predicted.eq(targets).sum().item()
                        total_targets.extend(targets.cpu().tolist())
                        total_preds.extend(predicted.cpu().tolist())
                    self.val_losses.append(val_loss)

                targets_array = np.array(total_targets)
                preds_array = np.array(total_preds)

                if "val_acc" in print_args:
                    self.val_acc.append(val_correct / val_total)
                    print("adding val_acc, length: %s" % len(self.val_acc))

            if (("tst_loss" in print_args) or ("tst_acc" in print_args)):
                with torch.no_grad():
                    total_targets = []
                    total_preds = []
                    for batch_idx, (inputs, targets) in enumerate(self.testloader):
                        # print(batch_idx)
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'], non_blocking=True)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        tst_loss += loss.item()
                        if "tst_acc" in print_args:
                            _, predicted = outputs.max(1)
                            tst_total += targets.size(0)
                            tst_correct += predicted.eq(targets).sum().item()
                        total_targets.extend(targets.cpu().tolist())
                        total_preds.extend(predicted.cpu().tolist())
                    self.tst_losses.append(tst_loss)
                if "tst_acc" in print_args:
                    self.tst_acc.append(tst_correct / tst_total)

            if "subtrn_acc" in print_args:
                self.subtrn_acc.append(subtrn_correct / subtrn_total)

            if "subtrn_losses" in print_args:
                self.subtrn_losses.append(subtrn_loss)

            print_str = "Epoch: " + str(self.timestep + 1)

            for arg in print_args:

                if arg == "val_loss":
                    print_str += " , " + "Validation Loss: " + str(self.val_losses[-1])

                if arg == "val_acc":
                    print_str += " , " + "Validation Accuracy: " + str(self.val_acc[-1])

                if arg == "tst_loss":
                    print_str += " , " + "Test Loss: " + str(self.tst_losses[-1])

                if arg == "tst_acc":
                    print_str += " , " + "Test Accuracy: " + str(self.tst_acc[-1])

                if arg == "trn_loss":
                    print_str += " , " + "Training Loss: " + str(self.trn_losses[-1])

                if arg == "trn_acc":
                    print_str += " , " + "Training Accuracy: " + str(self.trn_acc[-1])

                if arg == "subtrn_loss":
                    print_str += " , " + "Subset Loss: " + str(self.subtrn_losses[-1])

                if arg == "subtrn_acc":
                    print_str += " , " + "Subset Accuracy: " + str(self.subtrn_acc[-1])

                if arg == "time":
                    print_str += " , " + "Timing: " + str(self.timing[-1])

            # report metric to ray for hyperparameter optimization
            if 'report_tune' in self.configdata and self.configdata['report_tune']:
                tune.report(mean_accuracy=self.val_acc[-1])

            # breakpoint()
            # import pdb; pdb.set_trace()
            print(print_str)

        if ((self.timestep + 1) % self.configdata['ckpt']['save_every'] == 0) and self.configdata['ckpt'][
            'is_save'] == True:

            metric_dict = {}

            for arg in print_args:
                if arg == "val_loss":
                    metric_dict['val_loss'] = self.val_losses
                if arg == "val_acc":
                    metric_dict['val_acc'] = self.val_acc
                if arg == "tst_loss":
                    metric_dict['tst_loss'] = self.tst_losses
                if arg == "tst_acc":
                    metric_dict['tst_acc'] = self.tst_acc
                if arg == "trn_loss":
                    metric_dict['trn_loss'] = self.trn_losses
                if arg == "trn_acc":
                    metric_dict['trn_acc'] = self.trn_acc
                if arg == "subtrn_loss":
                    metric_dict['subtrn_loss'] = self.subtrn_losses
                if arg == "subtrn_acc":
                    metric_dict['subtrn_acc'] = self.subtrn_acc
                if arg == "time":
                    metric_dict['time'] = self.timing

            ckpt_state = {
                'epoch': self.timestep + 1,
                'loss': self.loss_function(),
                'metrics': metric_dict
            }

            # save checkpoint
            print("Saving at: %s" % self.checkpoint_path)
            self.save_ckpt(ckpt_state, self.checkpoint_path)
            print("Model checkpoint saved at epoch " + str(self.timestep + 1))

    def train(self):
        """
        #General Training Loop with Data Selection Strategies
        """
        # Loading the Dataset
        self.setup()
        # for epoch in range(start_epoch, self.configdata['train_args']['num_epochs']):
        while self.timestep < self.configdata['train_args']['num_epochs']:
            self.step()
            self.timestep += 1

        print(self.configdata['dss_strategy']['type'] + " Selection Run---------------------------------")
        print("Final SubsetTrn:", self.subtrn_loss)
        if "val_loss" in self.print_args:
            if "val_acc" in self.print_args:
                print("Validation Loss and Accuracy: ", self.val_loss, self.val_acc[-1])
            else:
                print("Validation Loss: ", self.val_loss)

        if "tst_loss" in self.print_args:
            if "tst_acc" in self.print_args:
                print("Test Data Loss and Accuracy: ", self.tst_loss, self.tst_acc[-1])
            else:
                print("Test Data Loss: ", self.tst_loss)
        print('-----------------------------------')
        print(self.configdata['dss_strategy']['type'], file=self.logfile)
        print('---------------------------------------------------------------------', file=self.logfile)

        if "val_acc" in self.print_args:
            val_str = "Validation Accuracy, "
            for val in self.val_acc:
                val_str = val_str + " , " + str(val)
            print(val_str, file=self.logfile)

        if "tst_acc" in self.print_args:
            tst_str = "Test Accuracy, "
            for tst in self.tst_acc:
                tst_str = tst_str + " , " + str(tst)
            print(tst_str, file=self.logfile)

        if "time" in self.print_args:
            time_str = "Time, "
            for t in self.timing:
                time_str = time_str + " , " + str(t)
            print(self.timing, file=self.logfile)

        omp_timing = np.array(self.timing)
        omp_cum_timing = list(self.generate_cumulative_timing(omp_timing))
        print("Total time taken by " + self.configdata['dss_strategy']['type'] + " = " + str(omp_cum_timing[-1]))
        self.logfile.close()

        return self.val_losses[-1]

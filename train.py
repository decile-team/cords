import time
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
from cords.utils.config_utils import load_config_data
import os.path as osp
from cords.selectionstrategies.supervisedlearning import OMPGradMatchStrategy, GLISTERStrategy, RandomStrategy, \
    CRAIGStrategy
from ray import tune

from cords.utils.models.simpleNN_net import FourLayerNet


class TrainClassifier:
    def __init__(self, config_file):
        self.config_file = config_file
        self.configdata = load_config_data(self.config_file)
        # if self.configdata['setting'] == 'supervisedlearning':
        #     from cords.selectionstrategies.supervisedlearning import OMPGradMatchStrategy, GLISTERStrategy, \
        #         RandomStrategy, CRAIGStrategy
        # elif self.configdata['setting'] == 'general':
        #     from cords.selectionstrategies.general import GLISTERStrategy

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
            # dataset_embedding = {"twitter": "glove_twitter_100", "corona": "glove_twitter_100", "news": "glove.6B.300d"}
            # dataset_embedding = {"twitter": "glove_twitter_200", "corona": "glove_twitter_200", "news": "glove.6B.300d"}
            # dataset_embedding = {"twitter": "glove_twitter_200", "corona": "glove_twitter_200", "news": "glove.6B.300d"}
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
            # optimizer = optim.SGD(model.parameters(), lr=self.configdata['optimizer']['lr'],
            #                       momentum=self.configdata['optimizer']['momentum'],
            #                       weight_decay=self.configdata['optimizer']['weight_decay'])
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

    def train(self):
        """
        #General Training Loop with Data Selection Strategies
        """
        # Loading the Dataset
        # breakpoint()
        if self.configdata['dataset']['feature'] == 'classimb':
            trainset, validset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                       self.configdata['dataset']['name'],
                                                                       self.configdata['dataset']['feature'],
                                                                       classimb_ratio=self.configdata['dataset'][
                                                                           'classimb_ratio'])
        else:
            trainset, validset, testset, num_cls = load_dataset_custom(self.configdata['dataset']['datadir'],
                                                                       self.configdata['dataset']['name'],
                                                                       self.configdata['dataset']['feature'])

        N = len(trainset)
        # print("--------------------------------------------------------")
        # print(N)
        # print("--------------------------------------------------------")
        # trn_batch_size = 20
        # trn_batch_size = 128
        trn_batch_size = self.configdata["dataloader"]["batch_size"]

        val_batch_size = 1000
        tst_batch_size = 1000

        train_start_time = round(time.time())
        print("train_start_time: %s" % train_start_time)

        # print("--------------------------------------------------------")
        print("trn_batch_size: %s. " % trn_batch_size)

        print("Using dss_strategy: %s" % self.configdata["dss_strategy"]["type"])
        # print("--------------------------------------------------------")

        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                  shuffle=False, pin_memory=True)

        valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                                shuffle=False, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                                 shuffle=False, pin_memory=True)

        # Budget for subset selection
        bud = int(self.configdata['dss_strategy']['fraction'] * N)
        print("Budget, fraction and N:", bud, self.configdata['dss_strategy']['fraction'], N)

        # Subset Selection and creating the subset data loader
        start_idxs = np.random.choice(N, size=bud, replace=False)
        idxs = start_idxs
        data_sub = Subset(trainset, idxs)
        subset_trnloader = torch.utils.data.DataLoader(data_sub,
                                                       batch_size=self.configdata['dataloader']['batch_size'],
                                                       shuffle=self.configdata['dataloader']['shuffle'],
                                                       pin_memory=self.configdata['dataloader']['pin_memory'])

        # Variables to store accuracies
        gammas = torch.ones(len(idxs)).to(self.configdata['train_args']['device'])
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
        print_every = self.configdata['train_args']['print_every']
        results_dir = osp.abspath(osp.expanduser(self.configdata['train_args']['results_dir']))
        # all_logs_dir = os.path.join(results_dir, self.configdata['dss_strategy']['type'],
        #                             self.configdata['dataset']['name'], str(
        #         self.configdata['dss_strategy']['fraction']), str(self.configdata['dss_strategy']['select_every']))
        all_logs_dir = os.path.join(results_dir, self.configdata['dss_strategy']['type'],
                                    self.configdata['dataset']['name'],
                                    str(self.configdata['dss_strategy']['fraction']),
                                    str(self.configdata['dss_strategy']['select_every']),
                                    str(train_start_time))

        os.makedirs(all_logs_dir, exist_ok=True)
        path_logfile = os.path.join(all_logs_dir, self.configdata['dataset']['name'] + '.txt')
        logfile = open(path_logfile, 'w')

        checkpoint_dir = osp.abspath(osp.expanduser(self.configdata['ckpt']['dir']))
        if "save_dir" in self.configdata:
            ckpt_dir = os.path.join(checkpoint_dir, self.configdata["save_dir"])
            checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        else:
            ckpt_dir = os.path.join(checkpoint_dir, self.configdata['dss_strategy']['type'],
                                    self.configdata['dataset']['name'], str(
                    self.configdata['dss_strategy']['fraction']), str(self.configdata['dss_strategy']['select_every']),
                                    str(train_start_time))
            checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        print("making dir: %s" % ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model Creation
        model = self.create_model()
        model1 = self.create_model()

        n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print("Training model: %s, number of parameters: %d" % (self.configdata['model']['architecture'], n_params))

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(model)

        if self.configdata['dss_strategy']['type'] == 'GradMatch':
            # OMPGradMatch Selection strategy
            setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'], num_cls, True,
                                              'PerClassPerGradient',
                                              valid=self.configdata['dss_strategy']['valid'],
                                              lam=self.configdata['dss_strategy']['lam'], eps=1e-100)
        elif self.configdata['dss_strategy']['type'] == 'GradMatchPB':
            setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'], num_cls, True, 'PerBatch',
                                              valid=self.configdata['dss_strategy']['valid'],
                                              lam=self.configdata['dss_strategy']['lam'], eps=1e-100)
        elif self.configdata['dss_strategy']['type'] == 'GLISTER':
            # GLISTER Selection strategy
            # setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
            #                              self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
            #                              num_cls, True, 'Supervised', 'Stochastic', r=int(bud))
            # GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
            #                 self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
            #                 num_cls, False, 'Supervised', 'Stochastic', r=int(bud))
            setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
                                         self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
                                         num_cls, True, 'Supervised', 'Stochastic', r=int(bud))
            # GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
            #                 self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
            #                 num_cls, False, 'Supervised', 'Stochastic', r=int(bud))
        # elif self.configdata['dss_strategy']['type'] == 'R-GLISTER':
        #     # GLISTER Selection strategy
        #     setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
        #                                  self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
        #                                  num_cls, True, 'RGreedy', r=int(self.configdata["dss_strategy"]["r"]))
        elif self.configdata['dss_strategy']['type'] == 'GLISTERPB':
            setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
                                         self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
                                         num_cls, True, 'Supervised', 'PerBatch', r=int(bud))
        elif self.configdata['dss_strategy']['type'] == 'GLISTERPC':
            setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
                                         self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
                                         num_cls, True, 'Supervised', 'PerClass', r=int(bud))
            # (self, trainloader, valloader, model,
            #  loss_func, eta, device, num_classes,
            #  linear_layer, selection_type, greedy, r=15, verbose=True)
            setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
                                         self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
                                         num_cls, True, 'PerBatch', 'Stochastic', r=int(bud))
        # elif self.configdata['dss_strategy']['type'] == 'GLISTERPC':
        #     setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
        #                                  self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
        #                                  num_cls, True, 'Supervised', 'PerClass', r=int(bud))
        elif self.configdata['dss_strategy']['type'] == 'CRAIG':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(trainloader, valloader, model1, criterion_nored,
                                       self.configdata['train_args']['device'], num_cls, False, False, 'PerClass')

        elif self.configdata['dss_strategy']['type'] == 'CRAIGPB':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(trainloader, valloader, model1, criterion_nored,
                                       self.configdata['train_args']['device'], num_cls, False, False, 'PerBatch')

        elif self.configdata['dss_strategy']['type'] == 'CRAIG-Warm':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(trainloader, valloader, model1, criterion_nored,
                                       self.configdata['train_args']['device'], num_cls, False, False, 'PerClass')
            # Random-Online Selection strategy
            # rand_setf_model = RandomStrategy(trainloader, online=True)
            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'CRAIGPB-Warm':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(trainloader, valloader, model1, criterion_nored,
                                       self.configdata['train_args']['device'], num_cls, False, False, 'PerBatch')
            # Random-Online Selection strategy
            # rand_setf_model = RandomStrategy(trainloader, online=True)
            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'Random':
            # Random Selection strategy
            setf_model = RandomStrategy(trainloader, online=False)

        elif self.configdata['dss_strategy']['type'] == 'Random-Online':
            # Random-Online Selection strategy
            setf_model = RandomStrategy(trainloader, online=True)

        elif self.configdata['dss_strategy']['type'] == 'GLISTER-Warm':
            # GLISTER Selection strategy
            # setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
            #                              self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
            #                              num_cls, False, 'Stochastic', r=int(bud))
            # (self, trainloader, valloader, model,
            #  loss_func, eta, device, num_classes,
            #  linear_layer, selection_type, greedy, r=15, verbose=True)
            setf_model = GLISTERStrategy(trainloader, valloader, model1, criterion_nored,
                                         self.configdata['optimizer']['lr'], self.configdata['train_args']['device'],
                                         num_cls, True, 'Supervised', 'Stochastic', r=int(bud))
            # Random-Online Selection strategy
            # rand_setf_model = RandomStrategy(trainloader, online=True)
            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'GradMatch-Warm':
            # OMPGradMatch Selection strategy
            setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'],
                                              num_cls, True, 'PerClassPerGradient',
                                              valid=self.configdata['dss_strategy']['valid'],
                                              lam=self.configdata['dss_strategy']['lam'], eps=1e-100)
            # Random-Online Selection strategy
            # rand_setf_model = RandomStrategy(trainloader, online=True)
            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'GradMatchPB-Warm':
            # OMPGradMatch Selection strategy
            setf_model = OMPGradMatchStrategy(trainloader, valloader, model1, criterion_nored,
                                              self.configdata['optimizer']['lr'],
                                              self.configdata['train_args']['device'],
                                              num_cls, True, 'PerBatch', valid=self.configdata['dss_strategy']['valid'],
                                              lam=self.configdata['dss_strategy']['lam'], eps=1e-100)
            # Random-Online Selection strategy
            # rand_setf_model = RandomStrategy(trainloader, online=True)
            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        elif self.configdata['dss_strategy']['type'] == 'Random-Warm':
            if 'kappa' in self.configdata['dss_strategy']:
                kappa_epochs = int(
                    self.configdata['dss_strategy']['kappa'] * self.configdata['train_args']['num_epochs'])
                full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])
            else:
                raise KeyError("Specify a kappa value in the config file")

        print("=======================================", file=logfile)

        if self.configdata['ckpt']['is_load'] == True:
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

        for epoch in range(start_epoch, self.configdata['train_args']['num_epochs']):
            subtrn_loss = 0
            subtrn_correct = 0
            subtrn_total = 0
            subset_selection_time = 0

            if self.configdata['dss_strategy']['type'] in ['Random-Online']:
                start_time = time.time()
                subset_idxs, gammas = setf_model.select(int(bud))
                idxs = subset_idxs
                subset_selection_time += (time.time() - start_time)
                gammas = gammas.to(self.configdata['train_args']['device'])

            elif self.configdata['dss_strategy']['type'] in ['Random']:
                pass

            elif (self.configdata['dss_strategy']['type'] in ['GLISTER', 'GLISTERPB', 'GradMatch',
                                                              'GradMatchPB', 'CRAIG', 'CRAIGPB', 'R-GLISTER']) and (
                    ((epoch + 1) % self.configdata['dss_strategy']['select_every']) == 0):
                start_time = time.time()
                cached_state_dict = copy.deepcopy(model.state_dict())
                clone_dict = copy.deepcopy(model.state_dict())
                subset_idxs, gammas = setf_model.select(int(bud), clone_dict)
                model.load_state_dict(cached_state_dict)
                idxs = subset_idxs
                if self.configdata['dss_strategy']['type'] in ['GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']:
                    gammas = torch.from_numpy(np.array(gammas)).to(self.configdata['train_args']['device']).to(
                        torch.float32)
                subset_selection_time += (time.time() - start_time)

            elif (self.configdata['dss_strategy']['type'] in ['GLISTER-Warm', 'GradMatch-Warm', 'GradMatchPB-Warm',
                                                              'CRAIG-Warm',
                                                              'CRAIGPB-Warm']):
                start_time = time.time()
                if ((epoch % self.configdata['dss_strategy']['select_every'] == 0) and (epoch >= kappa_epochs)):
                    cached_state_dict = copy.deepcopy(model.state_dict())
                    clone_dict = copy.deepcopy(model.state_dict())
                    subset_idxs, gammas = setf_model.select(int(bud), clone_dict)
                    model.load_state_dict(cached_state_dict)
                    idxs = subset_idxs
                    if self.configdata['dss_strategy']['type'] in ['GradMatch-Warm', 'GradMatchPB-Warm', 'CRAIG-Warm',
                                                                   'CRAIGPB-Warm']:
                        gammas = torch.from_numpy(np.array(gammas)).to(self.configdata['train_args']['device']).to(
                            torch.float32)
                subset_selection_time += (time.time() - start_time)

            elif self.configdata['dss_strategy']['type'] in ['Random-Warm']:
                pass

            # print("selEpoch: %d, Selection Ended at:" % (i), str(datetime.datetime.now()))
            data_sub = Subset(trainset, idxs)
            # subset_labels = torch.tensor([data_sub[i][1] for i in range(len(data_sub))])
            # for i in range(self.configdata['model']['numclasses']):
            #     print(" \n Class " + str(i) + " percentage: " + str(len(torch.where(subset_labels == i)[0])/len(subset_labels)))

            subset_trnloader = torch.utils.data.DataLoader(data_sub, batch_size=trn_batch_size, shuffle=False,
                                                           pin_memory=True)

            model.train()
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
                    optimizer.zero_grad()
                    if batch_idx % 1000 == 0:
                        print(
                            "batch_idx: %s, inputs.shape: %s, updating model parameters... " % (
                                batch_idx, inputs.shape))
                    # if batch_idx % 10 == 0:
                    #     print(
                    #         "batch_idx: %s, inputs.shape: %s, updating model parameters... " % (
                    #             batch_idx, inputs.shape))
                    # print("inputs.device: %s. " % inputs.device)
                    # print("targets.device: %s. " % targets.device)
                    outputs = model(inputs)
                    losses = criterion_nored(outputs, targets)
                    loss = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (
                        gammas[batch_wise_indices[batch_idx]].sum())
                    # loss = losses.mean()
                    # loss = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (gammas.sum())
                    ########################################################
                    # print(model)
                    # print(loss)
                    ########################################################
                    loss.backward()
                    subtrn_loss += loss.item()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
                train_time = time.time() - start_time

            elif self.configdata['dss_strategy']['type'] in ['CRAIGPB-Warm', 'CRAIG-Warm', 'GradMatch-Warm',
                                                             'GradMatchPB-Warm']:
                start_time = time.time()
                if epoch < full_epochs:
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'],
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

                elif epoch >= kappa_epochs:
                    for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'],
                            non_blocking=True)  # targets can have non_blocking=True.
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        losses = criterion_nored(outputs, targets)
                        loss = torch.dot(losses, gammas[batch_wise_indices[batch_idx]]) / (
                            gammas[batch_wise_indices[batch_idx]].sum())
                        # loss = losses.mean()
                        loss.backward()
                        subtrn_loss += loss.item()
                        optimizer.step()
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
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    # if batch_idx % 1000 == 0:
                    # if batch_idx % 3 == 0:
                    #     print(
                    #         "batch_idx: %s, inputs.shape: %s, updating model parameters... " % (
                    #         batch_idx, inputs.shape))
                    #     # print("loss: %s" % loss)
                    #     # print(model.linear1.weight)
                    loss.backward()
                    optimizer.step()
                    subtrn_loss += loss.item()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
                train_time = time.time() - start_time

            elif self.configdata['dss_strategy']['type'] in ['GLISTER-Warm', 'Random-Warm']:
                start_time = time.time()
                if epoch < full_epochs:
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'],
                            non_blocking=True)  # targets can have non_blocking=True.
                        optimizer.zero_grad()
                        # print(model.linear1.weight.dtype)
                        # print(inputs.dtype)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        subtrn_loss += loss.item()
                        optimizer.step()
                        _, predicted = outputs.max(1)
                        subtrn_total += targets.size(0)
                        subtrn_correct += predicted.eq(targets).sum().item()
                elif epoch >= kappa_epochs:
                    for batch_idx, (inputs, targets) in enumerate(subset_trnloader):
                        inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                            self.configdata['train_args']['device'],
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

            elif self.configdata['dss_strategy']['type'] in ['Full']:
                start_time = time.time()
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                        self.configdata['train_args']['device'],
                        non_blocking=True)  # targets can have non_blocking=True.
                    # print('inputs.shape: %s, targets.shape: %s' % (inputs.shape, targets.shape))
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    # ########################################################
                    # if batch_idx % 1 == 0:
                    #     print("Training full... ")
                    #     print(loss)
                    #     print("Shape of input: ")
                    #     print(inputs.shape)
                    # ########################################################
                    subtrn_loss += loss.item()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    subtrn_total += targets.size(0)
                    subtrn_correct += predicted.eq(targets).sum().item()
                train_time = time.time() - start_time

            scheduler.step()
            timing.append(train_time + subset_selection_time)
            # print("Adding timing: %5.2f, train_time: %5.2f, subset_selection_time: %5.2f" %
            #       (train_time + subset_selection_time, train_time, subset_selection_time))
            # print("adding timing, length: %s" % len(timing))
            print_args = self.configdata['train_args']['print_args']
            # print("Epoch timing is: " + str(timing[-1]))

            if ((epoch + 1) % self.configdata['train_args']['print_every'] == 0):
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
                        for batch_idx, (inputs, targets) in enumerate(trainloader):
                            # print(batch_idx)
                            inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                                self.configdata['train_args']['device'], non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            trn_loss += loss.item()
                            trn_losses.append(trn_loss)
                            if "trn_acc" in print_args:
                                _, predicted = outputs.max(1)
                                trn_total += targets.size(0)
                                trn_correct += predicted.eq(targets).sum().item()

                    if "trn_acc" in print_args:
                        trn_acc.append(trn_correct / trn_total)

                if (("val_loss" in print_args) or ("val_acc" in print_args)):
                    # print("Validating!!!")
                    with torch.no_grad():
                        total_targets = []
                        total_preds = []
                        for batch_idx, (inputs, targets) in enumerate(valloader):
                            # print(batch_idx)
                            inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                                self.configdata['train_args']['device'], non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                            if "val_acc" in print_args:
                                _, predicted = outputs.max(1)
                                val_total += targets.size(0)
                                val_correct += predicted.eq(targets).sum().item()
                            total_targets.extend(targets.cpu().tolist())
                            total_preds.extend(predicted.cpu().tolist())
                        val_losses.append(val_loss)
                        # print("val_losses: %s" % val_losses)

                    targets_array = np.array(total_targets)
                    preds_array = np.array(total_preds)
                    # for i in range(self.configdata["model"]["numclasses"]):    
                    #     cls_idxs = np.where(targets_array == i)
                    #     preds = preds_array[cls_idxs]
                    #     crct_preds = np.where(preds == i)
                    #     print("\n Class "+ str(i) + " val accuracy: ", 100 * len(crct_preds[0])/len(cls_idxs[0]))

                    if "val_acc" in print_args:
                        val_acc.append(val_correct / val_total)
                        print("adding val_acc, length: %s" % len(val_acc))

                if (("tst_loss" in print_args) or ("tst_acc" in print_args)):
                    with torch.no_grad():
                        total_targets = []
                        total_preds = []
                        for batch_idx, (inputs, targets) in enumerate(testloader):
                            # print(batch_idx)
                            inputs, targets = inputs.to(self.configdata['train_args']['device']), targets.to(
                                self.configdata['train_args']['device'], non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            tst_loss += loss.item()
                            if "tst_acc" in print_args:
                                _, predicted = outputs.max(1)
                                tst_total += targets.size(0)
                                tst_correct += predicted.eq(targets).sum().item()
                            total_targets.extend(targets.cpu().tolist())
                            total_preds.extend(predicted.cpu().tolist())
                        tst_losses.append(tst_loss)
                    if "tst_acc" in print_args:
                        tst_acc.append(tst_correct / tst_total)

                    # targets_array = np.array(total_targets)
                    # preds_array = np.array(total_preds)
                    # for i in range(self.configdata["model"]["numclasses"]):    
                    #     cls_idxs = np.where(targets_array == i)
                    #     preds = preds_array[cls_idxs]
                    #     crct_preds = np.where(preds == i)
                    #     print("\n Class "+ str(i) + " tst accuracy: ", 100 * len(crct_preds[0])/len(cls_idxs[0]))

                if "subtrn_acc" in print_args:
                    subtrn_acc.append(subtrn_correct / subtrn_total)

                if "subtrn_losses" in print_args:
                    subtrn_losses.append(subtrn_loss)

                print_str = "Epoch: " + str(epoch + 1)

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
                if 'report_tune' in self.configdata and self.configdata['report_tune']:
                    tune.report(mean_accuracy=val_acc[-1])

                # breakpoint()
                # import pdb; pdb.set_trace()
                print(print_str)

            if ((epoch + 1) % self.configdata['ckpt']['save_every'] == 0) and self.configdata['ckpt'][
                'is_save'] == True:

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

                # ckpt_state = {
                #     'epoch': epoch + 1,
                #     'state_dict': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'loss': self.loss_function(),
                #     'metrics': metric_dict
                # }

                ckpt_state = {
                    'epoch': epoch + 1,
                    # 'state_dict': model.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    'loss': self.loss_function(),
                    'metrics': metric_dict
                }

                # save checkpoint
                print("Saving at: %s" % checkpoint_path)
                self.save_ckpt(ckpt_state, checkpoint_path)
                print("Model checkpoint saved at epoch " + str(epoch + 1))

        print(self.configdata['dss_strategy']['type'] + " Selection Run---------------------------------")
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
        print(self.configdata['dss_strategy']['type'], file=logfile)
        print('---------------------------------------------------------------------', file=logfile)

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
        print("Total time taken by " + self.configdata['dss_strategy']['type'] + " = " + str(omp_cum_timing[-1]))
        logfile.close()

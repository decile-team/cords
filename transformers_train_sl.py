import logging
import os
import os.path as osp
import sys
import time
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ray import tune
from cords.utils.data.data_utils import WeightedSubset
from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, AdaptiveRandomDataLoader, StochasticGreedyDataLoader,\
    CRAIGDataLoader, GradMatchDataLoader, RandomDataLoader, AdapWeightsDataLoader, WeightedRandomDataLoader, MILODataLoader
from cords.utils.data.dataloader.SL.nonadaptive import FacLocDataLoader, MILOFixedDataLoader
from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.models import *
from cords.utils.data.data_utils.collate import *
from datetime import datetime
from transformers import AutoTokenizer, get_scheduler, BertConfig, AdamW
import wandb
import evaluate


LABEL_MAPPINGS = {'glue_sst2':'label', 
                  'hf_trec6':'coarse_label', 
                  'imdb':'label',
                  'rotten_tomatoes': 'label',
                  'tweet_eval': 'label'}

SENTENCE_MAPPINGS = {'glue_sst2': 'sentence', 
                    'hf_trec6':'text',  
                    'imdb':'text',
                    'rotten_tomatoes': 'text',
                    'tweet_eval': 'text'}

def tokenize_function(tokenizer, example, text_column):
    return tokenizer(example[text_column], padding = 'max_length', truncation=True)


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

class TrainClassifier:
    def __init__(self, config_file_data):
        # self.config_file = config_file
        # self.cfg = load_config_data(self.config_file)
        self.cfg = config_file_data
        results_dir = osp.abspath(osp.expanduser(self.cfg.train_args.results_dir))

        if self.cfg.dss_args.type in ['StochasticGreedyExploration', 'WeightedRandomExploration', 'SGE', 'WRE']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.kw)
        elif self.cfg.dss_args.type in ['MILO']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + self.cfg.dss_args.gc_ratio + "_" + str(self.cfg.dss_args.kw)
        else:
            subset_selection_name = self.cfg.dss_args.type
            
        all_logs_dir = os.path.join(results_dir, 
                                    self.cfg.setting,
                                    self.cfg.dataset.name,
                                    subset_selection_name,
                                    self.cfg.model.architecture,
                                    str(self.cfg.dss_args.fraction),
                                    str(self.cfg.dss_args.select_every),
                                    str(self.cfg.train_args.run))

        os.makedirs(all_logs_dir, exist_ok=True)
        # setup logger
        plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                            datefmt="%m/%d %H:%M:%S")
        now = datetime.now()
        current_time = now.strftime("%y/%m/%d %H:%M:%S")
        self.logger = logging.getLogger(__name__+"  " + current_time)
        self.logger.setLevel(logging.INFO)
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(plain_formatter)
        s_handler.setLevel(logging.INFO)
        self.logger.addHandler(s_handler)
        f_handler = logging.FileHandler(os.path.join(all_logs_dir, self.cfg.dataset.name + "_" +
                                                     self.cfg.dss_args.type + ".log"), mode='w')
        f_handler.setFormatter(plain_formatter)
        f_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(f_handler)
        self.logger.propagate = False


    """
    ############################## Loss Evaluation ##############################
    """

    def model_eval_loss(self, data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.cfg.train_args.device), \
                                  targets.to(self.cfg.train_args.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    """
    ############################## Model Creation ##############################
    """
    def create_model(self):
        if self.cfg.model.architecture == 'BERTMLP':
            model = BERTMLPModel(self.cfg.model.bert_config, self.cfg.model.checkpoint)
        model = model.to(self.cfg.train_args.device)
        return model

    """
    ############################## Loss Type, Optimizer and Learning Rate Scheduler ##############################
    """
    def loss_function(self):
        if self.cfg.loss.type == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction='none')
        elif self.cfg.loss.type == "MeanSquaredLoss":
            criterion = nn.MSELoss()
            criterion_nored = nn.MSELoss(reduction='none')
        return criterion, criterion_nored

    def optimizer_with_scheduler(self, model, num_training_steps):
        if self.cfg.optimizer.type == 'sgd':
            
            optimizer = optim.SGD(model.parameters(), lr=self.cfg.optimizer.lr,
                                  momentum=self.cfg.optimizer.momentum,
                                  weight_decay=self.cfg.optimizer.weight_decay,
                                  nesterov=self.cfg.optimizer.nesterov)
        elif self.cfg.optimizer.type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.type == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.type == "adamw":
            optimizer = AdamW(model.parameters(), lr=self.cfg.optimizer.lr)
        
        if self.cfg.scheduler.type == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.cfg.scheduler.T_max)
        elif self.cfg.scheduler.type == 'linear_decay':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=self.cfg.scheduler.stepsize, 
                                                        gamma=self.cfg.scheduler.gamma)
        # if self.cfg.scheduler.type == 'linear':
        #     scheduler = get_scheduler("linear",
        #                             optimizer=optimizer,
        #                             num_warmup_steps=self.cfg.scheduler.warmup_steps,
        #                             num_training_steps=num_training_steps
        #                             )
        else:
            scheduler = None
        return optimizer, scheduler

    @staticmethod
    def generate_cumulative_timing(mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing

    @staticmethod
    def save_ckpt(state, ckpt_path):
        torch.save(state, ckpt_path)

    @staticmethod
    def load_ckpt(ckpt_path, model, optimizer):
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        metrics = checkpoint['metrics']
        return start_epoch, model, optimizer, loss, metrics

    def count_pkl(self, path):
        if not osp.exists(path):
            return -1
        return_val = 0
        file = open(path, 'rb')
        while(True):
            try:
                _ = pickle.load(file)
                return_val += 1
            except EOFError:
                break
        file.close()
        return return_val

    def train(self, end_before_training = False):
        """
        ############################## General Training Loop with Data Selection Strategies ##############################
        """
        # Loading the Dataset
        logger = self.logger
        #logger.info(self.cfg)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.checkpoint)

        if self.cfg.dataset.feature == 'classimb':
            trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
                                                               self.cfg.dataset.name + '_transformer',
                                                               self.cfg.dataset.feature,
                                                               classimb_ratio=self.cfg.dataset.classimb_ratio, 
                                                               dataset=self.cfg.dataset,
                                                               tokenizer=tokenizer)
        else:
            trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
                                                               self.cfg.dataset.name + '_transformer',
                                                               self.cfg.dataset.feature, 
                                                               dataset=self.cfg.dataset,
                                                               tokenizer=tokenizer)

        trn_batch_size = self.cfg.dataloader.batch_size
        val_batch_size = self.cfg.dataloader.batch_size
        tst_batch_size = self.cfg.dataloader.batch_size

        
        
        assert (self.cfg.dataset.name in list(SENTENCE_MAPPINGS.keys())) and (self.cfg.dataset.name in list(LABEL_MAPPINGS.keys())), \
    "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in transformers_train_sl.py file."
        
        # tokenizer_mapping = lambda example: tokenize_function(tokenizer, example, SENTENCE_MAPPINGS[self.cfg.dataset.name])
        # trainset = trainset.map(tokenizer_mapping, batched=True) 
        # trainset = trainset.remove_columns([SENTENCE_MAPPINGS[self.cfg.dataset.name], "idx"])
        # trainset = trainset.rename_column(LABEL_MAPPINGS[self.cfg.dataset.name], "labels")
        # trainset.set_format("torch")
        # trainset = trainset.shuffle(seed=42)

        # validset = validset.map(tokenizer_mapping, batched=True)
        # validset = validset.remove_columns([SENTENCE_MAPPINGS[self.cfg.dataset.name], "idx"])
        # validset = validset.rename_column(LABEL_MAPPINGS[self.cfg.dataset.name], "labels")
        # validset.set_format("torch")
        # validset = validset.shuffle(seed=42)
        
        # testset = testset.map(tokenizer_mapping, batched=True)
        # testset = testset.remove_columns([SENTENCE_MAPPINGS[self.cfg.dataset.name], "idx"])
        # testset = testset.rename_column(LABEL_MAPPINGS[self.cfg.dataset.name], "labels")
        # testset.set_format("torch")
        # testset = testset.shuffle(seed=42)

        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                  shuffle=False, pin_memory=True)

        valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                                shuffle=False, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                                 shuffle=False, pin_memory=True, collate_fn = self.cfg.dataloader.collate_fn)

        train_eval_loader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size * 20,
                                                  shuffle=False, pin_memory=True)

        val_eval_loader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size * 20,
                                                shuffle=False, pin_memory=True)

        test_eval_loader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size * 20,
                                                 shuffle=False, pin_memory=True)

        substrn_losses = list()  # np.zeros(configdata['train_args']['num_epochs'])
        trn_losses = list()
        val_losses = list()  # np.zeros(configdata['train_args']['num_epochs'])
        tst_losses = list()
        subtrn_losses = list()
        timing = [0]
        trn_acc = list()
        val_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        tst_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])
        best_acc = list()
        curr_best_acc = 0
        subtrn_acc = list()  # np.zeros(configdata['train_args']['num_epochs'])

        # Checkpoint file
        checkpoint_dir = osp.abspath(osp.expanduser(self.cfg.ckpt.dir))
        
        if self.cfg.dss_args.type in ['StochasticGreedyExploration', 'WeightedRandomExploration', 'SGE', 'WRE']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.kw)
        elif self.cfg.dss_args.type in ['MILO']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + self.cfg.dss_args.gc_ratio + "_" + str(self.cfg.dss_args.kw)
        else:
            subset_selection_name = self.cfg.dss_args.type

        ckpt_dir = os.path.join(checkpoint_dir, 
                                self.cfg.setting,
                                self.cfg.dataset.name,
                                subset_selection_name,
                                self.cfg.model.architecture,
                                str(self.cfg.dss_args.fraction),
                                str(self.cfg.dss_args.select_every),
                                str(self.cfg.train_args.run))
                                
        checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model Creation
        model = self.create_model()
        if self.cfg.train_args.wandb:
            wandb.watch(model)
        # model1 = self.create_model()

        #Initial Checkpoint Directory
        init_ckpt_dir = os.path.abspath(os.path.expanduser("checkpoints"))
        os.makedirs(init_ckpt_dir, exist_ok=True)
        
        model_name = ""
        for key in self.cfg.model.keys():
            if (r"/" not in str(self.cfg.model[key])) and (key not in ['bert_config']):
                model_name += (str(self.cfg.model[key]) + "_")

        if model_name[-1] == "_":
            model_name = model_name[:-1]
            
        if not os.path.exists(os.path.join(init_ckpt_dir, model_name + ".pt")):
            ckpt_state = {'state_dict': model.state_dict()}
            # save checkpoint
            self.save_ckpt(ckpt_state, os.path.join(init_ckpt_dir, model_name + ".pt"))
        else:
            checkpoint = torch.load(os.path.join(init_ckpt_dir, model_name + ".pt"))
            model.load_state_dict(checkpoint['state_dict'])

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        num_training_steps = self.cfg.train_args.num_epochs * len(trainloader)
        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(model, num_training_steps)

        """
        ############################## Custom Dataloader Creation ##############################
        """

        if not 'collate_fn' in self.cfg.dss_args:
                self.cfg.dss_args.collate_fn = None

        if self.cfg.dss_args.type in ['GradMatch', 'GradMatchPB', 'GradMatch-Warm', 'GradMatchPB-Warm']:
            """
            ############################## GradMatch Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = GradMatchDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                             batch_size=self.cfg.dataloader.batch_size,
                                             shuffle=self.cfg.dataloader.shuffle,
                                             pin_memory=self.cfg.dataloader.pin_memory)
                                             #collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['GLISTER', 'GLISTER-Warm', 'GLISTERPB', 'GLISTERPB-Warm']:
            """
            ############################## GLISTER Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device
            
            dataloader = GLISTERDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                           batch_size=self.cfg.dataloader.batch_size,
                                           shuffle=self.cfg.dataloader.shuffle,
                                           pin_memory=self.cfg.dataloader.pin_memory,)
                                           #collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['CRAIG', 'CRAIG-Warm', 'CRAIGPB', 'CRAIGPB-Warm']:
            """
            ############################## CRAIG Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device

            dataloader = CRAIGDataLoader(trainloader, valloader, self.cfg.dss_args, logger,
                                         batch_size=self.cfg.dataloader.batch_size,
                                         shuffle=self.cfg.dataloader.shuffle,
                                         pin_memory=self.cfg.dataloader.pin_memory,)
                                         #collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['Random', 'Random-Warm']:
            """
            ############################## Random Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = RandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                          batch_size=self.cfg.dataloader.batch_size,
                                          shuffle=self.cfg.dataloader.shuffle,
                                          pin_memory=self.cfg.dataloader.pin_memory, )
                                          #collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['AdaptiveRandom', 'AdaptiveRandom-Warm']:
            """
            ############################## AdaptiveRandom Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = AdaptiveRandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,)
                                            #collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['MILOFixed', 'MILOFixed-Warm']:
            """
            ############################## MILOFixed Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = MILOFixedDataLoader(trainloader, self.cfg.dss_args, logger,
                                          batch_size=self.cfg.dataloader.batch_size,
                                          shuffle=self.cfg.dataloader.shuffle,
                                          pin_memory=self.cfg.dataloader.pin_memory,) 
                                          #collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['WeightedRandomExploration', 'WeightedRandomExploration-Warm', 'WRE', 'WRE-Warm']:
            """
            ############################## WeightedRandom Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = WeightedRandomDataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,)
                                            #collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type in ['StochasticGreedyExploration', 'StochasticGreedyExploration-Warm', 'SGE', 'SGE-Warm']:
            """
            ############################## StochasticGreedy Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = StochasticGreedyDataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,)
                                            #collate_fn = self.cfg.dss_args.collate_fn
					    

        elif self.cfg.dss_args.type in ['MILO']:
            """
            ############################## MILO Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs

            dataloader = MILODataLoader(trainloader, self.cfg.dss_args, logger,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,)
					    #collate_fn = self.cfg.dss_args.collate_fn)

        elif self.cfg.dss_args.type == 'FacLoc':
            """
            ############################## Facility Location Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.model = model
            self.cfg.dss_args.data_type = self.cfg.dataset.type
            
            dataloader = FacLocDataLoader(trainloader, valloader, self.cfg.dss_args, logger, 
                                        batch_size=self.cfg.dataloader.batch_size,
                                        shuffle=self.cfg.dataloader.shuffle,
                                        pin_memory=self.cfg.dataloader.pin_memory,) 
                                        #collate_fn = self.cfg.dss_args.collate_fn)

            if 'ss_path' in self.cfg.dataset and self.count_pkl(self.cfg.dataset.ss_path) < 1:
                #save subset indices if a ss_path is provided. Useful in HP tuning to avoid multiple facloc computations.
                #to avoid multiple parallel facloc computations, do facloc once(using end_before_training) then start HP tuning
                ss_indices = dataloader.subset_indices
                file_ss = open(self.cfg.dataset.ss_path, 'wb')
                try:
                    pickle.dump(ss_indices, file_ss)
                except EOFError:
                    pass
                file_ss.close()

        elif self.cfg.dss_args.type == 'AdapFacLoc':
            """
            ############################## Adaptive Facility Location Dataloader Additional Arguments ##############################
            """
            num_contents = self.count_pkl(self.cfg.dataset.ss_path)
            if num_contents < 1:
                self.cfg.dss_args.device = self.cfg.train_args.device
                self.cfg.dss_args.model = model
                self.cfg.dss_args.data_type = self.cfg.dataset.type
                
                facloc_time = time.time()
                dataloader = FacLocDataLoader(trainloader, valloader, self.cfg.dss_args, logger, 
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,) 
                                            #collate_fn = self.cfg.dss_args.collate_fn)
                ss_indices = list(dataloader.subset_indices)
                facloc_time = time.time() - facloc_time
                print("Type of ss_indices:", type(ss_indices))
                file_ss = open(self.cfg.dataset.ss_path, 'wb')
                try:
                    pickle.dump(ss_indices, file_ss)
                except EOFError:
                    pass
                file_ss.close()
                print("AdapFacLoc takes facloc time of:", facloc_time)
            elif num_contents == 1:
                print("We are in adapfacloc atleast once!")
                file_ss = open(self.cfg.dataset.ss_path, 'rb')
                ss_indices = pickle.load(file_ss)
                file_ss.close()

                self.cfg.dss_args.model = model
                self.cfg.dss_args.loss = criterion_nored
                self.cfg.dss_args.eta = self.cfg.optimizer.lr
                self.cfg.dss_args.num_classes = self.cfg.model.numclasses
                self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
                self.cfg.dss_args.device = self.cfg.train_args.device
                
                dataloader = AdapWeightsDataLoader(trainloader, valloader, self.cfg.dss_args, logger, ss_indices, 
                                            batch_size=self.cfg.dataloader.batch_size,
                                            shuffle=self.cfg.dataloader.shuffle,
                                            pin_memory=self.cfg.dataloader.pin_memory,) 
                                            #collate_fn = self.cfg.dss_args.collate_fn)
        elif self.cfg.dss_args.type == 'Full':
            """
            ############################## Full Dataloader Additional Arguments ##############################
            """
            wt_trainset = WeightedSubset(trainset, list(range(len(trainset))), [1] * len(trainset))

            dataloader = torch.utils.data.DataLoader(wt_trainset,
                                                     batch_size=self.cfg.dataloader.batch_size,
                                                     shuffle=self.cfg.dataloader.shuffle,
                                                     pin_memory=self.cfg.dataloader.pin_memory,)
                                                     #collate_fn=self.cfg.dss_args.collate_fn)

        """
        ################################################# Checkpoint Loading #################################################
        """

        if self.cfg.ckpt.is_load:
            start_epoch, model, optimizer, ckpt_loss, metric_dict = self.load_ckpt(checkpoint_path, model, optimizer)
            logger.info("Loading saved checkpoint model at epoch: {0:d}".format(start_epoch))
            for arg in metric_dict.keys():
                if arg == "val_loss":
                    val_losses = metric_dict['val_loss']
                if arg == "val_acc":
                    val_acc = metric_dict['val_acc']
                if arg == "tst_loss":
                    tst_losses = metric_dict['tst_loss']
                if arg == "tst_acc":
                    tst_acc = metric_dict['tst_acc']
                    best_acc = metric_dict['best_acc']
                if arg == "trn_loss":
                    trn_losses = metric_dict['trn_loss']
                if arg == "trn_acc":
                    trn_acc = metric_dict['trn_acc']
                if arg == "subtrn_loss":
                    subtrn_losses = metric_dict['subtrn_loss']
                if arg == "subtrn_acc":
                    subtrn_acc = metric_dict['subtrn_acc']
                if arg == "time":
                    timing = metric_dict['time']
        else:
            start_epoch = 0

        """
        ################################################# Training Loop #################################################
        """

        if end_before_training:
            torch.cuda.empty_cache()
            return
        
        train_time = 0
        for epoch in range(start_epoch, self.cfg.train_args.num_epochs+1):
            """
            ################################################# Evaluation Loop #################################################
            """
            print_args = self.cfg.train_args.print_args
            if ((epoch + 1) % self.cfg.train_args.print_every == 0) or (epoch == self.cfg.train_args.num_epochs - 1):
                trn_loss = 0
                val_loss = 0
                tst_loss = 0
                model.eval()
                logger_dict = {}
                if ("trn_loss" in print_args) or ("trn_acc" in print_args):
                    if "trn_acc" in print_args:
                        metric = evaluate.load('accuracy', experiment_id = self.cfg.train_args.wandb_name)

                    with torch.no_grad():
                        for batch in train_eval_loader:
                            batch = {k: v.to(self.cfg.train_args.device) for k, v in batch.items()}
                            outputs = model(**batch)
                            loss = criterion(outputs, batch['labels'].view(-1))
                            trn_loss += loss.item()
                            if "trn_acc" in print_args:
                                predictions = torch.argmax(outputs, dim=-1)
                                metric.add_batch(predictions=predictions, references=batch["labels"])
                        trn_losses.append(trn_loss)
                        logger_dict['trn_loss'] = trn_loss
                    if "trn_acc" in print_args:
                        trn_acc.append(metric.compute()['accuracy'])
                        logger_dict['trn_acc'] = trn_acc[-1]

                if ("val_loss" in print_args) or ("val_acc" in print_args):
                    if "val_acc" in print_args:
                        metric = evaluate.load('accuracy', experiment_id = self.cfg.train_args.wandb_name)

                    with torch.no_grad():
                        for batch in val_eval_loader:
                            batch = {k: v.to(self.cfg.train_args.device) for k, v in batch.items()}
                            outputs = model(**batch)
                            loss = criterion(outputs, batch['labels'].view(-1))
                            val_loss += loss.item()
                            if "val_acc" in print_args:
                                predictions = torch.argmax(outputs, dim=-1)
                                metric.add_batch(predictions=predictions, references=batch["labels"])
                        val_losses.append(val_loss)
                        logger_dict['val_loss'] = val_loss

                    if "val_acc" in print_args:
                        val_acc.append(metric.compute()['accuracy'])
                        logger_dict['val_acc'] = val_acc[-1]

                if ("tst_loss" in print_args) or ("tst_acc" in print_args):
                    if "tst_acc" in print_args:
                        metric = evaluate.load('accuracy', experiment_id = self.cfg.train_args.wandb_name)

                    with torch.no_grad():
                        for batch in test_eval_loader:
                            batch = {k: v.to(self.cfg.train_args.device) for k, v in batch.items()}
                            outputs = model(**batch)
                            loss = criterion(outputs, batch['labels'].view(-1))
                            tst_loss += loss.item()
                            if "tst_acc" in print_args:
                                predictions = torch.argmax(outputs, dim=-1)
                                metric.add_batch(predictions=predictions, references=batch["labels"])
                        tst_losses.append(tst_loss)
                        logger_dict['tst_loss'] = tst_loss
                    if "tst_acc" in print_args:
                        tst_acc.append(metric.compute()['accuracy'])
                        if tst_acc[-1] > curr_best_acc:
                            curr_best_acc = tst_acc[-1]
                        logger_dict['tst_acc'] = tst_acc[-1]
                        logger_dict['best_acc'] = curr_best_acc

                if "subtrn_acc" in print_args:
                    if epoch == 0:
                        subtrn_acc.append(0)
                        logger_dict['subtrn_acc'] = 0
                    else:    
                        subtrn_acc.append(curr_subtrn_acc)
                        logger_dict['subtrn_acc'] = curr_subtrn_acc

                if "subtrn_losses" in print_args:
                    if epoch == 0:
                        subtrn_losses.append(0) 
                        logger_dict['subtrn_loss'] = 0   
                    else:
                        subtrn_losses.append(subtrn_loss)
                        logger_dict['subtrn_loss'] = subtrn_loss

                print_str = "Epoch: " + str(epoch)
                logger_dict['Epoch'] = epoch
                logger_dict['Time'] = train_time
                
                if self.cfg.train_args.wandb:
                    wandb.log(logger_dict)

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
                        print_str += " , " + "Test Accuracy: " + str(best_acc[-1])

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
                if 'report_tune' in self.cfg and self.cfg.report_tune and len(dataloader) and epoch > 0:
                    tune.report(mean_accuracy=val_acc[-1])

                logger.info(print_str)

            subtrn_loss = 0
            model.train()
            start_time = time.time()
            metric= evaluate.load("accuracy", experiment_id = self.cfg.train_args.wandb_name)
            for batch in dataloader:
                batch = {k: v.to(self.cfg.train_args.device) for k, v in batch.items()}
                weights = batch['weights']
                batch.pop('weights')
                outputs = model(**batch, finetune=True)     
                losses = criterion_nored(outputs, batch['labels'].view(-1))
                loss = torch.dot(losses, weights.view(-1) / (weights.sum()))
                # loss = torch.dot(losses, weights / len(inputs))
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if not self.cfg.is_reg:
                    predictions = torch.argmax(outputs, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch["labels"])
            epoch_time = time.time() - start_time
            if not scheduler == None:
                scheduler.step()
            timing.append(epoch_time)
            train_time += epoch_time
            
            curr_subtrn_acc = metric.compute()['accuracy']
            #if "subtrn_acc" in print_args:
            #    subtrn_acc.append(metric.compute()['accuracy'])

            
            """
            ################################################# Checkpoint Saving #################################################
            """

            if ((epoch + 1) % self.cfg.ckpt.save_every == 0) and self.cfg.ckpt.is_save:

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
                        metric_dict['best_acc'] = best_acc
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
                logger.info("Model checkpoint saved at epoch: {0:d}".format(epoch + 1))

        """
        ################################################# Results Summary #################################################
        """
        original_idxs = set([x for x in range(len(trainset))])
        encountered_idxs = []
        if self.cfg.dss_args.type != 'Full':
            for key in dataloader.selected_idxs.keys():
                encountered_idxs.extend(dataloader.selected_idxs[key])
            encountered_idxs = set(encountered_idxs)
            rem_idxs = original_idxs.difference(encountered_idxs)
            encountered_percentage = len(encountered_idxs)/len(original_idxs)

            logger.info("Selected Indices: ") 
            logger.info(dataloader.selected_idxs)
            logger.info("Percentages of data samples encountered during training: %.2f", encountered_percentage)
            logger.info("Not Selected Indices: ")
            logger.info(rem_idxs)
            if self.cfg.train_args.wandb:
                wandb.log({
                           "Data Samples Encountered(in %)": encountered_percentage
                           })

        logger.info(self.cfg.dss_args.type + " Selection Run---------------------------------")
        logger.info("Final SubsetTrn: {0:f}".format(subtrn_loss))
        if "val_loss" in print_args:
            if "val_acc" in print_args:
                logger.info("Validation Loss: %.2f , Validation Accuracy: %.2f", val_loss, val_acc[-1])
            else:
                logger.info("Validation Loss: %.2f", val_loss)

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                logger.info("Test Loss: %.2f, Test Accuracy: %.2f, Best Accuracy: %.2f", tst_loss, tst_acc[-1], best_acc[-1])
            else:
                logger.info("Test Data Loss: %f", tst_loss)
        logger.info('---------------------------------------------------------------------')
        logger.info(self.cfg.dss_args.type)
        logger.info('---------------------------------------------------------------------')

        """
        ################################################# Final Results Logging #################################################
        """

        if "val_acc" in print_args:
            val_str = "Validation Accuracy: "
            for val in val_acc:
                if val_str == "Validation Accuracy: ":
                    val_str = val_str + str(val)
                else:
                    val_str = val_str + " , " + str(val)
            logger.info(val_str)

        if "tst_acc" in print_args:
            tst_str = "Test Accuracy: "
            for tst in tst_acc:
                if tst_str == "Test Accuracy: ":
                    tst_str = tst_str + str(tst)
                else:
                    tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

            tst_str = "Best Accuracy: "
            for tst in best_acc:
                if tst_str == "Best Accuracy: ":
                    tst_str = tst_str + str(tst)
                else:
                    tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

        if "time" in print_args:
            time_str = "Time: "
            for t in timing:
                if time_str == "Time: ":
                    time_str = time_str + str(t)
                else:
                    time_str = time_str + " , " + str(t)
            logger.info(time_str)

        omp_timing = np.array(timing)
        omp_cum_timing = list(self.generate_cumulative_timing(omp_timing))
        logger.info("Total time taken by %s = %.4f ", self.cfg.dss_args.type, omp_cum_timing[-1])
        return trn_acc, val_acc, tst_acc, best_acc

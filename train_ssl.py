import logging
import numpy, random, time, json, copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from cords.utils.data._utils import WeightedSubset
from cords.utils.models import WideResNet, ShakeNet, CNN13, CNN
from cords.utils.data.datasets.SSL import utils as dataset_utils
from cords.selectionstrategies.helpers.ssl_lib.algs.builder import gen_ssl_alg
from cords.selectionstrategies.helpers.ssl_lib.algs import utils as alg_utils
from cords.utils.models import utils as model_utils
from cords.selectionstrategies.helpers.ssl_lib.consistency.builder import gen_consistency
from cords.utils.data.datasets.SSL import gen_dataset
from cords.selectionstrategies.helpers.ssl_lib.param_scheduler import scheduler
from cords.selectionstrategies.helpers.ssl_lib.misc.meter import Meter
from cords.utils.data.dataloader.SSL.adaptive import *
from cords.utils.config_utils import load_config_data
import time
import os, sys


class TrainClassifier:
    def __init__(self, config_file):
        self.config_file = config_file
        self.cfg = load_config_data(self.config_file)
        os.makedirs(self.cfg.train_args.out_dir, exist_ok=True)
        # setup logger
        plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(plain_formatter)
        s_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(s_handler)
        f_handler = logging.FileHandler(os.path.join(self.cfg.train_args.out_dir, "console.log"))
        f_handler.setFormatter(plain_formatter)
        f_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(f_handler)
        self.logger.propagate = False
        self.logger.info(self.cfg)


    """
    ############################## Model Creation ##############################
    """

    def gen_model(self, name, num_classes, img_size):
        scale =  int(np.ceil(np.log2(img_size)))
        if name == "wrn":
            return WideResNet(num_classes, 32, scale, 4)
        elif name == "shake":
            return ShakeNet(num_classes, 32, scale, 4)
        elif name == "cnn13":
            return CNN13(num_classes, 32)
        elif name == 'cnn':
            return CNN(num_classes)
        else:
            raise NotImplementedError


    """
    ############################## Model Evaluation ##############################
    """
    def evaluation(self, raw_model, eval_model, loader, device):
        raw_model.eval()
        eval_model.eval()
        sum_raw_acc = sum_acc = sum_loss = 0
        with torch.no_grad():
            for (data, labels) in loader:
                data, labels = data.to(device), labels.to(device)
                preds = eval_model(data)
                raw_preds = raw_model(data)
                loss = F.cross_entropy(preds, labels)
                sum_loss += loss.item()
                acc = (preds.max(1)[1] == labels).float().mean()
                raw_acc = (raw_preds.max(1)[1] == labels).float().mean()
                sum_acc += acc.item()
                sum_raw_acc += raw_acc.item()
        mean_raw_acc = sum_raw_acc / len(loader)
        mean_acc = sum_acc / len(loader)
        mean_loss = sum_loss / len(loader)
        raw_model.train()
        eval_model.train()
        return mean_raw_acc, mean_acc, mean_loss


    """
    ############################## Model Parameters Update ##############################
    """
    def param_update(self, 
        cur_iteration,
        model,
        teacher_model,
        optimizer,
        ssl_alg,
        consistency,
        labeled_data,
        ul_weak_data,
        ul_strong_data,
        labels,
        average_model,
        weights=None,
        ood=False
    ):
        # if ood:
        #     model.update_batch_stats(False)
        start_time = time.time()
        all_data = torch.cat([labeled_data, ul_weak_data, ul_strong_data], 0)
        forward_func = model.forward
        stu_logits = forward_func(all_data)
        labeled_preds = stu_logits[:labeled_data.shape[0]]

        stu_unlabeled_weak_logits, stu_unlabeled_strong_logits = torch.chunk(stu_logits[labels.shape[0]:], 2, dim=0)

        if self.cfg.optimizer.tsa:
            none_reduced_loss = F.cross_entropy(labeled_preds, labels, reduction="none")
            L_supervised = alg_utils.anneal_loss(
                labeled_preds, labels, none_reduced_loss, cur_iteration+1,
                self.cfg.train_args.iteration, labeled_preds.shape[1], self.cfg.optimizer.tsa_schedule)
        else:
            L_supervised = F.cross_entropy(labeled_preds, labels)

        if self.cfg.ssl_args.coef > 0:
            # get target values
            if teacher_model is not None: # get target values from teacher model
                t_forward_func = teacher_model.forward
                tea_logits = t_forward_func(all_data)
                tea_unlabeled_weak_logits, _ = torch.chunk(tea_logits[labels.shape[0]:], 2, dim=0)
            else:
                t_forward_func = forward_func
                tea_unlabeled_weak_logits = stu_unlabeled_weak_logits

            # calc consistency loss
            model.update_batch_stats(False)
            y, targets, mask = ssl_alg(
                stu_preds=stu_unlabeled_strong_logits,
                tea_logits=tea_unlabeled_weak_logits.detach(),
                w_data=ul_strong_data,
                subset=False,
                stu_forward=forward_func,
                tea_forward=t_forward_func
            )
            model.update_batch_stats(True)
            # if not ood:
            #     model.update_batch_stats(True)
            if weights is None:
                L_consistency = consistency(y, targets, mask, weak_prediction=tea_unlabeled_weak_logits.softmax(1))
            else:
                L_consistency = consistency(y, targets, mask*weights, weak_prediction=tea_unlabeled_weak_logits.softmax(1))
        else:
            L_consistency = torch.zeros_like(L_supervised)
            mask = None

        # calc total loss
        coef = scheduler.exp_warmup(self.cfg.ssl_args.coef, int(self.cfg.scheduler.warmup_iter), cur_iteration + 1)
        loss = L_supervised + coef * L_consistency
        if self.cfg.ssl_args.em > 0:
            loss -= self.cfg.ssl_args.em * \
                (stu_unlabeled_weak_logits.softmax(1) * F.log_softmax(stu_unlabeled_weak_logits, 1)).sum(1).mean()

        # update parameters
        cur_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        loss.backward()
        if self.cfg.optimizer.weight_decay > 0:
            decay_coeff = self.cfg.optimizer.weight_decay * cur_lr
            model_utils.apply_weight_decay(model.modules(), decay_coeff)
        optimizer.step()

        # update teacher parameters by exponential moving average
        if self.cfg.ssl_args.ema_teacher:
            model_utils.ema_update(
                teacher_model, model, self.cfg.ssl_args.ema_teacher_factor,
                self.cfg.optimizer.weight_decay * cur_lr if self.cfg.ssl_args.ema_apply_wd else None,
                cur_iteration if self.cfg.ssl_args.ema_teacher_warmup else None)
        # update evaluation model's parameters by exponential moving average
        if self.cfg.ssl_eval_args.weight_average:
            model_utils.ema_update(
                average_model, model, self.cfg.ssl_eval_args.wa_ema_factor,
                self.cfg.optimizer.weight_decay * cur_lr if self.cfg.ssl_eval_args.wa_apply_wd else None)

        # calculate accuracy for labeled data
        acc = (labeled_preds.max(1)[1] == labels).float().mean()

        return {
            "acc": acc,
            "loss": loss.item(),
            "sup loss": L_supervised.item(),
            "ssl loss": L_consistency.item(),
            "mask": mask.float().mean().item() if mask is not None else 1,
            "coef": coef,
            "sec/iter": (time.time() - start_time)
        }


    """
    ############################## Calculate selected ID points percentage  ##############################
    """
    def get_ul_ood_ratio(self, ul_dataset):
        actual_lbls = ul_dataset.dataset.dataset['labels'][ul_dataset.indices]
        bincnt = numpy.bincount(actual_lbls, minlength=10)
        print("ID points selected", (bincnt[:6].sum()/bincnt.sum()).item())


    """
    ############################## Calculate selected ID points percentage  ##############################
    """
    def get_ul_classimb_ratio(self, ul_dataset):
        actual_lbls = ul_dataset.dataset.dataset['labels'][ul_dataset.indices]
        bincnt = numpy.bincount(actual_lbls, minlength=10)
        print("ClassImbalance points selected", (bincnt[:5].sum()/bincnt.sum()).item())


    """
    ############################## Main File ##############################
    """
    def train(self):
        logger = self.logger
        # set seed
        torch.manual_seed(self.cfg.train_args.seed)
        numpy.random.seed(self.cfg.train_args.seed)
        random.seed(self.cfg.train_args.seed)
        device = self.cfg.train_args.device
        # build data loader
        logger.info("load dataset")
        lt_data, ult_data, test_data, num_classes, img_size = gen_dataset(self.cfg.dataset.root, self.cfg.dataset.name, 
                                                                         False, self.cfg, logger)
        # set consistency type
        consistency = gen_consistency(self.cfg.ssl_args.consis, self.cfg)
        consistency_nored = gen_consistency(self.cfg.ssl_args.consis +'_red', self.cfg)
        # set ssl algorithm
        ssl_alg = gen_ssl_alg(self.cfg.ssl_args.alg, self.cfg)
        # build student model
        model = self.gen_model(self.cfg.model.architecture, num_classes, img_size).to(device)
        # build teacher model
        if self.cfg.ssl_args.ema_teacher:
            teacher_model = self.gen_model(self.cfg.model.architecture, num_classes, img_size).to(device)
            teacher_model.load_state_dict(model.state_dict())
        else:
            teacher_model = None
        # for evaluation
        if self.cfg.ssl_eval_args.weight_average:
            average_model = self.gen_model(self.cfg.model.architecture, num_classes, img_size).to(device)
            average_model.load_state_dict(model.state_dict())
        else:
            average_model = None

        """
        Subset selection arguments
        """
        if self.cfg.dss_args.type == 'Full':
            max_iteration = self.cfg.train_args.iteration
        else:
            if self.cfg.train_args.max_iter != -1:
                max_iteration = self.cfg.train_args.max_iter
            else:
                max_iteration = int(self.cfg.train_args.iteration * self.cfg.dss_args.fraction)

        # build optimizer
        N = len(ult_data)
        bud = int(self.cfg.dss_args.fraction * N)
        start_idxs = numpy.random.choice(N, size=bud, replace=False)

        #Create Data Loaders
        ult_seq_loader = DataLoader(ult_data, batch_size=self.cfg.dataloader.ul_batch_size,
                shuffle=False, pin_memory=True)

        lt_seq_loader = DataLoader(lt_data, batch_size=self.cfg.dataloader.l_batch_size,
                shuffle=False, pin_memory=True)

        test_loader = DataLoader(
            test_data,
            1,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.dataloader.num_workers
        )

        """
        ############################## Custom Dataloader Creation ##############################
        """

        if self.cfg.dss_args.type in ['GradMatch', 'GradMatchPB', 'GradMatch-Warm', 'GradMatchPB-Warm']:
            """
            ############################## GradMatch Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.tea_model = teacher_model
            self.cfg.dss_args.ssl_alg = ssl_alg
            self.cfg.dss_args.loss = consistency_nored
            self.cfg.dss_args.num_classes = num_classes
            self.cfg.dss_args.num_iters = self.cfg.train_args.iteration
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.device = self.cfg.train_args.device

            ult_loader = GradMatchDataLoader(ult_seq_loader, lt_seq_loader, self.cfg.dss_args, verbose=True, 
                                                batch_size=self.cfg.dataloader.ul_batch_size, 
                                                pin_memory=self.cfg.dataloader.pin_memory)

        
        elif self.cfg.dss_args.type in ['RETRIEVE', 'RETRIEVE-Warm', 'RETRIEVEPB', 'RETRIEVEPB-Warm']:
            """
            ############################## RETRIEVE Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.tea_model = teacher_model
            self.cfg.dss_args.ssl_alg = ssl_alg
            self.cfg.dss_args.loss = consistency_nored
            self.cfg.dss_args.num_classes = num_classes
            self.cfg.dss_args.num_iters = self.cfg.train_args.iteration
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.device = self.cfg.train_args.device

            ult_loader = RETRIEVEDataLoader(ult_seq_loader, lt_seq_loader, self.cfg.dss_args, verbose=True, 
                                                batch_size=self.cfg.dataloader.ul_batch_size, 
                                                pin_memory=self.cfg.dataloader.pin_memory)

        
        elif self.cfg.dss_args.type in ['CRAIG', 'CRAIG-Warm', 'CRAIGPB', 'CRAIGPB-Warm']:
            """
            ############################## CRAIG Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.tea_model = teacher_model
            self.cfg.dss_args.ssl_alg = ssl_alg
            self.cfg.dss_args.loss = consistency_nored
            self.cfg.dss_args.num_classes = num_classes
            self.cfg.dss_args.num_iters = self.cfg.train_args.iteration
            self.cfg.dss_args.device = self.cfg.train_args.device
            ult_loader = CRAIGDataLoader(ult_seq_loader, lt_seq_loader, self.cfg.dss_args, verbose=True, 
                                                batch_size=self.cfg.dataloader.ul_batch_size, 
                                                pin_memory=self.cfg.dataloader.pin_memory)


        elif self.cfg.dss_args.type in ['Random', 'Random-Warm']:
            """
            ############################## Random Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_classes = num_classes
            self.cfg.dss_args.num_iters = self.cfg.train_args.iteration
            self.cfg.dss_args.device = self.cfg.train_args.device
            ult_loader = RandomDataLoader(ult_seq_loader, self.cfg.dss_args, verbose=True,
                                            batch_size=self.cfg.dataloader.ul_batch_size, 
                                            pin_memory=self.cfg.dataloader.pin_memory)
            

        elif self.cfg.dss_args.type == ['OLRandom', 'OLRandom-Warm']:
            """
            ############################## OLRandom Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.device = self.cfg.train_args.device
            self.cfg.dss_args.num_classes = num_classes
            self.cfg.dss_args.num_iters = self.cfg.train_args.iteration
            self.cfg.dss_args.device = self.cfg.train_args.device
            ult_loader = OLRandomDataLoader(ult_seq_loader, self.cfg.dss_args, verbose=True,
                                            batch_size=self.cfg.dataloader.ul_batch_size, 
                                            pin_memory=self.cfg.dataloader.pin_memory)
        
        elif self.cfg.dss_args.type == 'Full':
            ############################## Full Dataloader Additional Arguments ##############################
            wt_trainset = WeightedSubset(ult_data, list(range(len(ult_data))), [1]*len(ult_data))

            ult_loader = torch.utils.data.DataLoader(wt_trainset,
                                            batch_size=self.cfg.dataloader.ul_batch_size, 
                                            pin_memory=self.cfg.dataloader.pin_memory)

        model.train()
        logger.info(model)

        if self.cfg.optimizer.type == "sgd":
            optimizer = optim.SGD(
                model.parameters(), self.cfg.optimizer.lr, self.cfg.optimizer.momentum, weight_decay=0, nesterov=True)
        elif self.cfg.optimizer.type == "adam":
            optimizer = optim.Adam(
                model.parameters(), self.cfg.optimizer.lr, (self.cfg.optimizer, 0.999), weight_decay=0
            )
        else:
            raise NotImplementedError

        # set lr scheduler
        if self.cfg.scheduler.lr_decay == "cos":
            if self.cfg.dss_args.type == 'Full':
                lr_scheduler = scheduler.CosineAnnealingLR(optimizer, max_iteration)
            else:
                lr_scheduler = scheduler.CosineAnnealingLR(optimizer, self.cfg.train_args.iteration * self.cfg.dss_args.fraction)
        elif self.cfg.scheduler.lr_decay == "step":
            # TODO: fixed milestones
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [400000, ], self.cfg.scheduler.lr_decay_rate)
        else:
            raise NotImplementedError

        # init meter
        metric_meter = Meter()
        test_acc_list = []
        raw_acc_list = []
        logger.info("training")

        if self.cfg.dataset.feature == 'ood':
            self.get_ul_ood_ratio(ult_loader.dataset)
        elif self.cfg.dataset.feature == 'classimb':
            self.get_ul_classimb_ratio(ult_loader.dataset)

        iter_count = 1
        subset_selection_time = 0
        training_time = 0
        
        while iter_count <= max_iteration:
            lt_loader = DataLoader(
                lt_data,
                self.cfg.dataloader.l_batch_size,
                sampler=dataset_utils.InfiniteSampler(len(lt_data), len(list(ult_loader.batch_sampler)) * self.cfg.dataloader.l_batch_size),
                num_workers=self.cfg.dataloader.num_workers
            )
            for batch_idx, (l_data, ul_data) in enumerate(zip(lt_loader, ult_loader)):
                batch_start_time = time.time()
                if iter_count > max_iteration:
                    break
                l_aug, labels = l_data
                ul_w_aug, ul_s_aug, _, weights = ul_data
                if self.cfg.dataset.feature in ['ood', 'classimb']:
                    ood = True
                else:
                    ood = False
                params = self.param_update(
                    iter_count, model, teacher_model, optimizer, ssl_alg,
                    consistency, l_aug.to(device), ul_w_aug.to(device),
                    ul_s_aug.to(device), labels.to(device),
                    average_model, weights=weights.to(device), ood=ood)
                training_time += (time.time() - batch_start_time)
                # moving average for reporting losses and accuracy
                metric_meter.add(params, ignores=["coef"])
                # display losses every cfg.disp iterations
                if ((iter_count + 1) % self.cfg.train_args.disp) == 0:
                    state = metric_meter.state(
                        header=f'[{iter_count + 1}/{max_iteration}]',
                        footer=f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                    )
                    logger.info(state)
                lr_scheduler.step()
                if ((iter_count + 1) % self.cfg.ckpt.checkpoint) == 0 or (iter_count + 1) == max_iteration:
                    with torch.no_grad():
                        if self.cfg.ssl_eval_args.weight_average:
                            eval_model = average_model
                        else:
                            eval_model = model
                        logger.info("test")
                        mean_raw_acc, mean_test_acc, mean_test_loss = self.evaluation(model, eval_model, test_loader, device)
                        logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                                    mean_raw_acc)
                        test_acc_list.append(mean_test_acc)
                        raw_acc_list.append(mean_raw_acc)
                    torch.save(model.state_dict(), os.path.join(self.cfg.train_args.out_dir, "model_checkpoint.pth"))
                    torch.save(optimizer.state_dict(), os.path.join(self.cfg.train_args.out_dir, "optimizer_checkpoint.pth"))
                iter_count += 1

            
        numpy.save(os.path.join(self.cfg.train_args.out_dir, "results"), test_acc_list)
        numpy.save(os.path.join(self.cfg.train_args.out_dir, "raw_results"), raw_acc_list)
        print("Total Time taken: ", training_time + subset_selection_time)
        print("Subset Selection Time: ", subset_selection_time)
        accuracies = {}
        for i in [1, 10, 20, 50]:
            logger.info("mean test acc. over last %d checkpoints: %f", i, numpy.median(test_acc_list[-i:]))
            logger.info("mean test acc. for raw model over last %d checkpoints: %f", i, numpy.median(raw_acc_list[-i:]))
            accuracies[f"last{i}"] = numpy.median(test_acc_list[-i:])

        with open(os.path.join(self.cfg.train_args.out_dir, "results.json"), "w") as f:
            json.dump(accuracies, f, sort_keys=True)


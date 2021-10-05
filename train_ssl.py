import logging
import numpy, random, time, json, copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from cords.utils.models import WideResNet, ShakeNet, CNN13, CNN
from cords.utils.data.datasets.SSL import utils as dataset_utils
from cords.selectionstrategies.helpers.ssl_lib.algs.builder import gen_ssl_alg
from cords.selectionstrategies.helpers.ssl_lib.algs import utils as alg_utils
from cords.utils.models import utils as model_utils
from cords.selectionstrategies.helpers.ssl_lib.consistency.builder import gen_consistency
from cords.utils.data.datasets.SSL import gen_dataset
from cords.selectionstrategies.helpers.ssl_lib.param_scheduler import scheduler
from cords.selectionstrategies.helpers.ssl_lib.misc.meter import Meter
from cords.selectionstrategies.SSL import *
import time


def gen_model(name, num_classes, img_size):
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


def evaluation(raw_model, eval_model, loader, device):
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


def param_update(
    cfg,
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

    if cfg.tsa:
        none_reduced_loss = F.cross_entropy(labeled_preds, labels, reduction="none")
        L_supervised = alg_utils.anneal_loss(
            labeled_preds, labels, none_reduced_loss, cur_iteration+1,
            cfg.iteration, labeled_preds.shape[1], cfg.tsa_schedule)
    else:
        L_supervised = F.cross_entropy(labeled_preds, labels)

    if cfg.coef > 0:
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
    coef = scheduler.exp_warmup(cfg.coef, int(cfg.warmup_iter), cur_iteration + 1)
    #coef = scheduler.exp_warmup(cfg.coef, int(cfg.warmup_iter*cfg.fraction), cur_iteration+1)
    loss = L_supervised + coef * L_consistency
    if cfg.entropy_minimization > 0:
        loss -= cfg.entropy_minimization * \
            (stu_unlabeled_weak_logits.softmax(1) * F.log_softmax(stu_unlabeled_weak_logits, 1)).sum(1).mean()

    # update parameters
    cur_lr = optimizer.param_groups[0]["lr"]
    optimizer.zero_grad()
    loss.backward()
    if cfg.weight_decay > 0:
        decay_coeff = cfg.weight_decay * cur_lr
        model_utils.apply_weight_decay(model.modules(), decay_coeff)
    optimizer.step()

    # update teacher parameters by exponential moving average
    if cfg.ema_teacher:
        model_utils.ema_update(
            teacher_model, model, cfg.ema_teacher_factor,
            cfg.weight_decay * cur_lr if cfg.ema_apply_wd else None,
            cur_iteration if cfg.ema_teacher_warmup else None)
    # update evaluation model's parameters by exponential moving average
    if cfg.weight_average:
        model_utils.ema_update(
            average_model, model, cfg.wa_ema_factor,
            cfg.weight_decay * cur_lr if cfg.wa_apply_wd else None)

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


def get_ul_ood_ratio(ul_dataset):
    actual_lbls = ul_dataset.dataset.dataset['labels'][ul_dataset.indices]
    bincnt = numpy.bincount(actual_lbls, minlength=10)
    print("ID points selected", (bincnt[:6].sum()/bincnt.sum()).item())


def get_ul_classimb_ratio(ul_dataset):
    actual_lbls = ul_dataset.dataset.dataset['labels'][ul_dataset.indices]
    bincnt = numpy.bincount(actual_lbls, minlength=10)
    print("ClassImbalance points selected", (bincnt[:5].sum()/bincnt.sum()).item())


def main(cfg, logger):
    # set seed
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # select device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("CUDA is NOT available")
        device = "cpu"
    # build data loader
    logger.info("load dataset")
    lt_data, ult_data, test_data, num_classes, img_size = gen_dataset(cfg.root, cfg.dataset, False, cfg, logger)

    # set consistency type
    consistency = gen_consistency(cfg.consistency, cfg)
    consistency_nored = gen_consistency(cfg.consistency+'_red', cfg)
    # set ssl algorithm
    ssl_alg = gen_ssl_alg(cfg.alg, cfg)
    # build student model
    model = gen_model(cfg.model, num_classes, img_size).to(device)
    model1 = gen_model(cfg.model, num_classes, img_size).to(device)
    # build teacher model
    if cfg.ema_teacher:
        teacher_model = gen_model(cfg.model, num_classes, img_size).to(device)
        teacher_model.load_state_dict(model.state_dict())
        teacher_model1 = gen_model(cfg.model, num_classes, img_size).to(device)
        teacher_model1.load_state_dict(model.state_dict())
    else:
        teacher_model = None
        teacher_model1 = None
    # for evaluation
    if cfg.weight_average:
        average_model = gen_model(cfg.model, num_classes, img_size).to(device)
        average_model.load_state_dict(model.state_dict())
    else:
        average_model = None

    """
    Subset selection arguments
    """
    if cfg.dss_strategy == 'Full':
        max_iteration = cfg.iteration
    else:
        if cfg.max_iter != -1:
            max_iteration = cfg.max_iter
        else:
            max_iteration = int(cfg.iteration * cfg.fraction)

    sel_iteration = int((cfg.select_every * len(ult_data) * cfg.fraction) // (cfg.ul_batch_size))  # build optimizer
    N = len(ult_data)
    bud = int(cfg.fraction * N)
    start_idxs = numpy.random.choice(N, size=bud, replace=False)

    #Create Data Loaders
    ult_seq_loader = DataLoader(ult_data, batch_size=cfg.ul_batch_size,
               shuffle=False, pin_memory=True)

    lt_seq_loader = DataLoader(lt_data, batch_size=cfg.l_batch_size,
               shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        test_data,
        1,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers
    )

    if cfg.dss_strategy == 'GradMatch':
        # OMPGradMatch Selection strategy
        setf_model = OMPGradMatchStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model, ssl_alg, consistency_nored,
        cfg.lr, device, num_classes, True, 'PerClassPerGradient', valid=cfg.valid, lam=0.25, eps=1e-10)
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'GradMatchPB':
        setf_model = OMPGradMatchStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model, ssl_alg, consistency_nored,
        cfg.lr, device, num_classes, True, 'PerBatch', valid=cfg.valid, lam=0.25, eps=1e-10)
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'RETRIEVE':
        # RETRIEVE Selection strategy
        setf_model = RETRIEVEStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model1, ssl_alg, consistency_nored,
                 cfg.lr, device, num_classes, False, 'Stochastic', r=int(bud), valid=True)
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'RETRIEVE_UL':
        # RETRIEVE Selection strategy
        setf_model = RETRIEVEStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model1, ssl_alg, consistency_nored,
                 cfg.lr, device, num_classes, False, 'Stochastic', r=int(bud), valid=False)
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'CRAIG':
        # CRAIG Selection strategy
        setf_model = CRAIGStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model1, ssl_alg, consistency_nored,
                 device, num_classes, False, False, 'PerClass', optimizer='lazy')
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'CRAIGPB':
        # CRAIG Selection strategy
        setf_model = CRAIGStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model1, ssl_alg, consistency_nored,
                 device, num_classes, False, False, 'PerBatch', optimizer='lazy')
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'CRAIG-Warm':
        # CRAIG Selection strategy
        setf_model = CRAIGStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model1, ssl_alg, consistency_nored,
                 device, num_classes, False, False, 'PerClass', optimizer='lazy')

        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)
        #full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])

    elif cfg.dss_strategy == 'CRAIGPB-Warm':
        # CRAIG Selection strategy
        setf_model = CRAIGStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model1, ssl_alg, consistency_nored,
                 device, num_classes, False, False, 'PerBatch', optimizer='lazy')
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)
        #full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])

    elif cfg.dss_strategy == 'Random':
        # Random Selection strategy
        setf_model = RandomStrategy(ult_seq_loader, online=False)
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'Random-Online':
        # Random-Online Selection strategy
        setf_model = RandomStrategy(ult_seq_loader, online=True)
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'RETRIEVE-Warm':
        # RETRIEVE Selection strategy
        setf_model = RETRIEVEStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model1, ssl_alg, consistency_nored,
                 cfg.lr, device, num_classes, False, 'Stochastic', r=int(bud), valid=True)

        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'RETRIEVE_UL-Warm':
        # RETRIEVE Selection strategy
        setf_model = RETRIEVEStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model1, ssl_alg, consistency_nored,
                                     cfg.lr, device, num_classes, False, 'Stochastic', r=int(bud), valid=False)
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'GradMatch-Warm':
        # OMPGradMatch Selection strategy
        setf_model = OMPGradMatchStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model, ssl_alg, consistency_nored,
        cfg.lr, device, num_classes, True, 'PerBatch', valid=args.valid, lam=0.25, eps=1e-10)
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)
        #full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])

    elif cfg.dss_strategy == 'GradMatchPB-Warm':
        # OMPGradMatch Selection strategy
        setf_model = OMPGradMatchStrategy(ult_seq_loader, lt_seq_loader, model1, teacher_model, ssl_alg, consistency_nored,
        cfg.lr, device, num_classes, True, 'PerBatch', valid=args.valid, lam=0.25, eps=1e-10)
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'Random-Warm':
        kappa_iterations = int(cfg.kappa * cfg.iteration * cfg.fraction)

    elif cfg.dss_strategy == 'Full':
        kappa_iterations = int(0.01 * max_iteration)

    model.train()
    logger.info(model)

    if cfg.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), cfg.lr, cfg.momentum, weight_decay=0, nesterov=True)
    elif cfg.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), cfg.lr, (cfg.momentum, 0.999), weight_decay=0
        )
    else:
        raise NotImplementedError

    # set lr scheduler
    if cfg.lr_decay == "cos":
        if cfg.dss_strategy == 'Full':
            lr_scheduler = scheduler.CosineAnnealingLR(optimizer, max_iteration)
        else:
            lr_scheduler = scheduler.CosineAnnealingLR(optimizer, cfg.iteration * cfg.fraction)
    elif cfg.lr_decay == "step":
        # TODO: fixed milestones
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [400000, ], cfg.lr_decay_rate)
    else:
        raise NotImplementedError

    # init meter
    metric_meter = Meter()
    test_acc_list = []
    raw_acc_list = []
    logger.info("training")
    timing = []
    #Initial Random Subset of Unlabeled Data
    ult_subset = Subset(ult_data, start_idxs)
    ult_subset_loader = DataLoader(
        ult_subset,
        cfg.ul_batch_size,
        sampler=dataset_utils.InfiniteSampler(len(ult_subset), sel_iteration * cfg.ul_batch_size),
        num_workers=cfg.num_workers)

    gammas = torch.ones(len(start_idxs), device=device)

    if cfg.ood:
        get_ul_ood_ratio(ult_subset)
    elif cfg.classimb:
        get_ul_classimb_ratio(ult_subset)

    ult_loader = DataLoader(
        ult_data,
        cfg.ul_batch_size,
        sampler=dataset_utils.InfiniteSampler(len(ult_data), kappa_iterations * cfg.ul_batch_size),
        num_workers=cfg.num_workers
    )

    iter_count = 1
    subset_selection_time = 0
    training_time = 0

    while iter_count <= max_iteration:

        """
        Adaptive Subset Selection
        """

        if (cfg.dss_strategy in ['Random-Online']) and (iter_count > 1):
            start_time = time.time()
            subset_idxs, gammas = setf_model.select(int(bud))
            idxs = subset_idxs
            subset_selection_time += (time.time() - start_time)
            gammas = gammas.to(device)
            ult_subset = Subset(ult_data, idxs)
            ult_subset_loader = DataLoader(
                ult_subset,
                cfg.ul_batch_size,
                sampler=dataset_utils.InfiniteSampler(len(ult_subset), sel_iteration * cfg.ul_batch_size),
                num_workers=cfg.num_workers
            )
            if cfg.ood:
                get_ul_ood_ratio(ult_subset)
            elif cfg.classimb:
                get_ul_classimb_ratio(ult_subset)

        elif (cfg.dss_strategy in ['Random']) and (iter_count > 1):
            pass

        elif (cfg.dss_strategy in ['RETRIEVE', 'RETRIEVE_UL', 'GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']) and (iter_count > 1):
            start_time = time.time()
            # Perform Subset Selection
            cached_state_dict = copy.deepcopy(model.state_dict())
            clone_dict = copy.deepcopy(model.state_dict())
            if teacher_model == None:
                tea_clone_dict = None
            else:
                tea_cached_state_dict = copy.deepcopy(teacher_model.state_dict())
                tea_clone_dict = copy.deepcopy(teacher_model.state_dict())
            subset_idxs, gammas = setf_model.select(int(bud), clone_dict, tea_clone_dict)
            model.load_state_dict(cached_state_dict)
            if teacher_model != None:
                teacher_model.load_state_dict(tea_cached_state_dict)
            idxs = subset_idxs

            if cfg.dss_strategy in ['GradMatch', 'GradMatchPB', 'CRAIG', 'CRAIGPB']:
                gammas = torch.from_numpy(numpy.array(gammas)).to(device).to(torch.float32)

            subset_selection_time += (time.time() - start_time)
            ult_subset = Subset(ult_data, idxs)
            ult_subset_loader = DataLoader(
                ult_subset,
                cfg.ul_batch_size,
                sampler=dataset_utils.InfiniteSampler(len(ult_subset), sel_iteration * cfg.ul_batch_size),
                num_workers=cfg.num_workers
            )
            if cfg.ood:
                get_ul_ood_ratio(ult_subset)
            elif cfg.classimb:
                get_ul_classimb_ratio(ult_subset)

        elif (cfg.dss_strategy in ['RETRIEVE-Warm', 'RETRIEVE_UL-Warm', 'GradMatch-Warm', 'GradMatchPB-Warm',
                                    'CRAIG-Warm', 'CRAIGPB-Warm']) and (iter_count > 1):
            start_time = time.time()
            if iter_count >= kappa_iterations:
                cached_state_dict = copy.deepcopy(model.state_dict())
                clone_dict = copy.deepcopy(model.state_dict())
                if teacher_model == None:
                    tea_clone_dict = None
                else:
                    tea_cached_state_dict = copy.deepcopy(teacher_model.state_dict())
                    tea_clone_dict = copy.deepcopy(teacher_model.state_dict())
                subset_idxs, gammas = setf_model.select(int(bud), clone_dict, tea_clone_dict)
                model.load_state_dict(cached_state_dict)
                if teacher_model != None:
                    teacher_model.load_state_dict(tea_cached_state_dict)
                idxs = subset_idxs
                if cfg.dss_strategy in ['GradMatch-Warm', 'GradMatchPB-Warm',
                                        'CRAIG-Warm', 'CRAIGPB-Warm']:
                    gammas = torch.from_numpy(numpy.array(gammas)).to(device).to(torch.float32)
            subset_selection_time += (time.time() - start_time)
            ult_subset = Subset(ult_data, idxs)
            ult_subset_loader = DataLoader(
                ult_subset,
                cfg.ul_batch_size,
                sampler=dataset_utils.InfiniteSampler(len(ult_subset), sel_iteration * cfg.ul_batch_size),
                num_workers=cfg.num_workers
            )
            if cfg.ood:
                get_ul_ood_ratio(ult_subset)
            elif cfg.classimb:
                get_ul_classimb_ratio(ult_subset)

        elif (cfg.dss_strategy in ['Random-Warm']) and (iter_count > 1):
            pass

        if cfg.dss_strategy == 'Full':
            lt_loader = DataLoader(
                lt_data,
                cfg.l_batch_size,
                sampler=dataset_utils.InfiniteSampler(len(lt_data), kappa_iterations * cfg.l_batch_size),
                num_workers=cfg.num_workers
            )

            for batch_idx, (l_data, ul_data) in enumerate(zip(lt_loader, ult_loader)):
                batch_start_time = time.time()
                if iter_count > max_iteration:
                    break
                l_aug, labels = l_data
                ul_w_aug, ul_s_aug, _ = ul_data
                params = param_update(
                    cfg, iter_count, model, teacher_model, optimizer, ssl_alg,
                    consistency, l_aug.to(device), ul_w_aug.to(device),
                    ul_s_aug.to(device), labels.to(device),
                    average_model, ood=cfg.ood
                )
                training_time += (time.time() - batch_start_time)
                # moving average for reporting losses and accuracy
                metric_meter.add(params, ignores=["coef"])
                # display losses every cfg.disp iterations
                if ((iter_count + 1) % cfg.disp) == 0:
                    state = metric_meter.state(
                        header=f'[{iter_count + 1}/{max_iteration}]',
                        footer=f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                    )
                    logger.info(state)
                lr_scheduler.step()
                if ((iter_count + 1) % cfg.checkpoint) == 0 or (iter_count + 1) == max_iteration:
                    with torch.no_grad():
                        if cfg.weight_average:
                            eval_model = average_model
                        else:
                            eval_model = model
                        logger.info("test")
                        mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader, device)
                        logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                                    mean_raw_acc)
                        test_acc_list.append(mean_test_acc)
                        raw_acc_list.append(mean_raw_acc)
                    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
                    torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))
                iter_count += 1

        elif cfg.dss_strategy in ['CRAIG', 'CRAIGPB', 'GradMatch', 'GradMatchPB']:

            lt_loader = DataLoader(
                lt_data,
                cfg.l_batch_size,
                sampler=dataset_utils.InfiniteSampler(len(lt_data), sel_iteration * cfg.l_batch_size),
                num_workers=cfg.num_workers
            )

            batch_wise_indices = list(ult_subset_loader.batch_sampler)
            for batch_idx, (l_data, ul_data) in enumerate(zip(lt_loader, ult_subset_loader)):
                batch_start_time = time.time()
                if iter_count > max_iteration:
                    break
                l_aug, labels = l_data
                ul_w_aug, ul_s_aug, _ = ul_data
                params = param_update(
                    cfg, iter_count, model, teacher_model, optimizer, ssl_alg,
                    consistency, l_aug.to(device), ul_w_aug.to(device),
                    ul_s_aug.to(device), labels.to(device),
                    average_model, weights=gammas[batch_wise_indices[batch_idx]], ood=cfg.ood
                )
                training_time += (time.time() - batch_start_time)
                # moving average for reporting losses and accuracy
                metric_meter.add(params, ignores=["coef"])
                # display losses every cfg.disp iterations
                if ((iter_count + 1) % cfg.disp) == 0:
                    state = metric_meter.state(
                        header=f'[{iter_count + 1}/{max_iteration}]',
                        footer=f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                    )
                    logger.info(state)
                lr_scheduler.step()
                if ((iter_count + 1) % cfg.checkpoint) == 0 or (iter_count + 1) == max_iteration:
                    with torch.no_grad():
                        if cfg.weight_average:
                            eval_model = average_model
                        else:
                            eval_model = model
                        logger.info("test")
                        mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader, device)
                        logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                                    mean_raw_acc)
                        test_acc_list.append(mean_test_acc)
                        raw_acc_list.append(mean_raw_acc)
                    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
                    torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))
                iter_count += 1

        elif cfg.dss_strategy in ['CRAIGPB-Warm', 'CRAIG-Warm', 'GradMatch-Warm', 'GradMatchPB-Warm']:
            if iter_count > kappa_iterations:
                lt_loader = DataLoader(
                    lt_data,
                    cfg.l_batch_size,
                    sampler=dataset_utils.InfiniteSampler(len(lt_data), sel_iteration * cfg.l_batch_size),
                    num_workers=cfg.num_workers
                )

                batch_wise_indices = list(ult_subset_loader.batch_sampler)
                for batch_idx, (l_data, ul_data) in enumerate(zip(lt_loader, ult_subset_loader)):
                    batch_start_time = time.time()
                    if iter_count > max_iteration:
                        break
                    l_aug, labels = l_data
                    ul_w_aug, ul_s_aug, _ = ul_data
                    params = param_update(
                        cfg, iter_count, model, teacher_model, optimizer, ssl_alg,
                        consistency, l_aug.to(device), ul_w_aug.to(device),
                        ul_s_aug.to(device), labels.to(device),
                        average_model, weights=gammas[batch_wise_indices[batch_idx]], ood=cfg.ood)
                    training_time += (time.time() - batch_start_time)
                    # moving average for reporting losses and accuracy
                    metric_meter.add(params, ignores=["coef"])
                    # display losses every cfg.disp iterations
                    if ((iter_count + 1) % cfg.disp) == 0:
                        state = metric_meter.state(
                            header=f'[{iter_count + 1}/{max_iteration}]',
                            footer=f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                        )
                        logger.info(state)
                    lr_scheduler.step()
                    if ((iter_count + 1) % cfg.checkpoint == 0) or (iter_count == max_iteration):
                        with torch.no_grad():
                            if cfg.weight_average:
                                eval_model = average_model
                            else:
                                eval_model = model
                            logger.info("test")
                            mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader, device)
                            logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                                        mean_raw_acc)
                            test_acc_list.append(mean_test_acc)
                            raw_acc_list.append(mean_raw_acc)
                        torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
                        torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))
                    iter_count += 1
            else:
                lt_loader = DataLoader(
                    lt_data,
                    cfg.l_batch_size,
                    sampler=dataset_utils.InfiniteSampler(len(lt_data), kappa_iterations * cfg.l_batch_size),
                    num_workers=cfg.num_workers
                )

                for batch_idx, (l_data, ul_data) in enumerate(zip(lt_loader, ult_loader)):
                    batch_start_time = time.time()
                    if iter_count > max_iteration:
                        break
                    l_aug, labels = l_data
                    ul_w_aug, ul_s_aug, _ = ul_data
                    params = param_update(
                        cfg, iter_count, model, teacher_model, optimizer, ssl_alg,
                        consistency, l_aug.to(device), ul_w_aug.to(device),
                        ul_s_aug.to(device), labels.to(device),
                        average_model, ood=cfg.ood
                    )
                    training_time += (time.time() - batch_start_time)
                    # moving average for reporting losses and accuracy
                    metric_meter.add(params, ignores=["coef"])
                    # display losses every cfg.disp iterations
                    if ((iter_count + 1) % cfg.disp) == 0:
                        state = metric_meter.state(
                            header=f'[{iter_count + 1}/{max_iteration}]',
                            footer=f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                        )
                        logger.info(state)
                    lr_scheduler.step()

                    if ((iter_count + 1) % cfg.checkpoint) == 0 or (iter_count + 1) == max_iteration:
                        with torch.no_grad():
                            if cfg.weight_average:
                                eval_model = average_model
                            else:
                                eval_model = model
                            logger.info("test")
                            mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader,
                                                                                     device)
                            logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                                        mean_raw_acc)
                            test_acc_list.append(mean_test_acc)
                            raw_acc_list.append(mean_raw_acc)
                        torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
                        torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))
                    iter_count += 1

        elif cfg.dss_strategy in ['RETRIEVE', 'RETRIEVE_UL', 'Random', 'Random-Online']:
            lt_loader = DataLoader(
                lt_data,
                cfg.l_batch_size,
                sampler=dataset_utils.InfiniteSampler(len(lt_data), sel_iteration * cfg.l_batch_size),
                num_workers=cfg.num_workers
            )

            for batch_idx, (l_data, ul_data) in enumerate(zip(lt_loader, ult_subset_loader)):
                batch_start_time = time.time()
                if iter_count > max_iteration:
                    break
                l_aug, labels = l_data
                ul_w_aug, ul_s_aug, _ = ul_data
                params = param_update(
                    cfg, iter_count, model, teacher_model, optimizer, ssl_alg,
                    consistency, l_aug.to(device), ul_w_aug.to(device),
                    ul_s_aug.to(device), labels.to(device),
                    average_model, ood=cfg.ood)
                training_time += (time.time() - batch_start_time)
                # moving average for reporting losses and accuracy
                metric_meter.add(params, ignores=["coef"])
                # display losses every cfg.disp iterations
                if ((iter_count + 1) % cfg.disp) == 0:
                    state = metric_meter.state(
                        header=f'[{iter_count + 1}/{max_iteration}]',
                        footer=f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                    )
                    logger.info(state)
                lr_scheduler.step()
                if ((iter_count + 1) % cfg.checkpoint) == 0 or (iter_count + 1) == max_iteration:
                    with torch.no_grad():
                        if cfg.weight_average:
                            eval_model = average_model
                        else:
                            eval_model = model
                        logger.info("test")
                        mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader, device)
                        logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                                    mean_raw_acc)
                        test_acc_list.append(mean_test_acc)
                        raw_acc_list.append(mean_raw_acc)
                    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
                    torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))
                iter_count += 1

        elif cfg.dss_strategy in ['RETRIEVE-Warm', 'RETRIEVE_UL-Warm', 'Random-Warm']:
            if iter_count > kappa_iterations:
                lt_loader = DataLoader(
                    lt_data,
                    cfg.l_batch_size,
                    sampler=dataset_utils.InfiniteSampler(len(lt_data), sel_iteration * cfg.l_batch_size),
                    num_workers=cfg.num_workers
                )

                for batch_idx, (l_data, ul_data) in enumerate(zip(lt_loader, ult_subset_loader)):
                    batch_start_time = time.time()
                    if iter_count > max_iteration:
                        break
                    l_aug, labels = l_data
                    ul_w_aug, ul_s_aug, _ = ul_data
                    params = param_update(
                        cfg, iter_count, model, teacher_model, optimizer, ssl_alg,
                        consistency, l_aug.to(device), ul_w_aug.to(device),
                        ul_s_aug.to(device), labels.to(device),
                        average_model, ood=cfg.ood)
                    training_time += (time.time() - batch_start_time)
                    # moving average for reporting losses and accuracy
                    metric_meter.add(params, ignores=["coef"])
                    # display losses every cfg.disp iterations
                    if ((iter_count + 1) % cfg.disp) == 0:
                        state = metric_meter.state(
                            header=f'[{iter_count + 1}/{max_iteration}]',
                            footer=f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                        )
                        logger.info(state)
                    lr_scheduler.step()
                    if ((iter_count + 1) % cfg.checkpoint == 0) or (iter_count == max_iteration):
                        with torch.no_grad():
                            if cfg.weight_average:
                                eval_model = average_model
                            else:
                                eval_model = model
                            logger.info("test")
                            mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader,
                                                                                     device)
                            logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                                        mean_raw_acc)
                            test_acc_list.append(mean_test_acc)
                            raw_acc_list.append(mean_raw_acc)
                        torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
                        torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))
                    iter_count += 1
            else:
                lt_loader = DataLoader(
                    lt_data,
                    cfg.l_batch_size,
                    sampler=dataset_utils.InfiniteSampler(len(lt_data), kappa_iterations * cfg.l_batch_size),
                    num_workers=cfg.num_workers
                )

                for batch_idx, (l_data, ul_data) in enumerate(zip(lt_loader, ult_loader)):
                    batch_start_time = time.time()
                    if iter_count > max_iteration:
                        break
                    l_aug, labels = l_data
                    ul_w_aug, ul_s_aug, _ = ul_data
                    params = param_update(
                        cfg, iter_count, model, teacher_model, optimizer, ssl_alg,
                        consistency, l_aug.to(device), ul_w_aug.to(device),
                        ul_s_aug.to(device), labels.to(device),
                        average_model, ood=cfg.ood
                    )
                    training_time += (time.time() - batch_start_time)
                    # moving average for reporting losses and accuracy
                    metric_meter.add(params, ignores=["coef"])
                    # display losses every cfg.disp iterations
                    if ((iter_count + 1) % cfg.disp) == 0:
                        state = metric_meter.state(
                            header=f'[{iter_count + 1}/{max_iteration}]',
                            footer=f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                        )
                        logger.info(state)
                    lr_scheduler.step()

                    if ((iter_count + 1) % cfg.checkpoint) == 0 or (iter_count + 1) == max_iteration:
                        with torch.no_grad():
                            if cfg.weight_average:
                                eval_model = average_model
                            else:
                                eval_model = model
                            logger.info("test")
                            mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader,
                                                                                     device)
                            logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                                        mean_raw_acc)
                            test_acc_list.append(mean_test_acc)
                            raw_acc_list.append(mean_raw_acc)
                        torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
                        torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))
                    iter_count += 1

    numpy.save(os.path.join(cfg.out_dir, "results"), test_acc_list)
    numpy.save(os.path.join(cfg.out_dir, "raw_results"), raw_acc_list)
    print("Total Time taken: ", training_time + subset_selection_time)
    print("Subset Selection Time: ", subset_selection_time)
    accuracies = {}
    for i in [1, 10, 20, 50]:
        logger.info("mean test acc. over last %d checkpoints: %f", i, numpy.median(test_acc_list[-i:]))
        logger.info("mean test acc. for raw model over last %d checkpoints: %f", i, numpy.median(raw_acc_list[-i:]))
        accuracies[f"last{i}"] = numpy.median(test_acc_list[-i:])

    with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
        json.dump(accuracies, f, sort_keys=True)


if __name__ == "__main__":
    import os, sys
    from configs.SSL import get_args
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # setup logger
    plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    s_handler = logging.StreamHandler(stream=sys.stdout)
    s_handler.setFormatter(plain_formatter)
    s_handler.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    f_handler = logging.FileHandler(os.path.join(args.out_dir, "console.log"))
    f_handler.setFormatter(plain_formatter)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)
    logger.propagate = False
    logger.info(args)
    main(args, logger)

Configuration files of CORDS for Training
==========================================

This page gives a tutorial on how to generate your custom training configuration files for SL, SSL and HPO. 
Configuration files can be used to select datasets, training configuration, and subset selection strategy settings. 
These configuration files can be in python dictionary or yaml format. However, for the sake of simplicity we show config files in python.

Configuration files for SL
---------------------------
    .. code-block:: python
    
        config = dict(setting="SL",
                is_reg = False,
                dataset=dict(name="cifar10",
                            datadir="../data",
                            feature="dss",
                            type="image"),

                dataloader=dict(shuffle=True,
                                batch_size=20,
                                pin_memory=True),

                model=dict(architecture='ResNet18',
                            type='pre-defined',
                            numclasses=10),
                
                ckpt=dict(is_load=False,
                            is_save=True,
                            dir='results/',
                            save_every=20),
                
                loss=dict(type='CrossEntropyLoss',
                            use_sigmoid=False),

                optimizer=dict(type="sgd",
                                momentum=0.9,
                                lr=0.01,
                                weight_decay=5e-4),

                scheduler=dict(type="cosine_annealing",
                                T_max=300),

                dss_args=dict(type="CRAIG",
                                    fraction=0.1,
                                    select_every=20,
                                    kappa=0,
                                    linear_layer=False,
                                    optimizer='lazy',
                                    selection_type='PerClass'
                                    ),

                train_args=dict(num_epochs=300,
                                device="cuda",
                                print_every=10,
                                results_dir='results/',
                                print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                                return_args=[]
                                )
                )


    The SL configuration files consists of following sections:
    #. Dataset(dataset)
    #. Data Loader(dataloader)
    #. Checkpoint Arguments (ckpt)
    #. Training Loss (loss)
    #. Training Optimizer (optimizer)
    #. Training Scheduler (scheduler)
    #. Data subset selection Arguments (dss_args)
    #. Training Arguments (train_args)

    You can refer to various configuration examples in the configs/ folders of the CORDS repository.

Configuration files for SSL
---------------------------
    .. code-block:: python
    
        # Learning setting
        config = dict(setting="SSL",
              dataset=dict(name="cifar10",
                           root="../data",
                           feature="dss",
                           type="pre-defined",
                           num_labels=4000,
                           val_ratio=0.1,
                           ood_ratio=0.5,
                           random_split=False,
                           whiten=False,
                           zca=True,
                           labeled_aug='WA',
                           unlabeled_aug='WA',
                           wa='t.t.f',
                           strong_aug=False),

              dataloader=dict(shuffle=True,
                              pin_memory=True,
                              num_workers=8,
                              l_batch_size=50,
                              ul_batch_size=50),

              model=dict(architecture='wrn',
                         type='pre-defined',
                         numclasses=10),

              ckpt=dict(is_load=False,
                        is_save=True,
                        checkpoint_model='model.ckpt',
                        checkpoint_optimizer='optimizer.ckpt',
                        start_iter=None,
                        checkpoint=10000),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.03,
                             weight_decay=0,
                             nesterov=True,
                             tsa=False,
                             tsa_schedule='linear'),

              scheduler=dict(lr_decay="cos",
                             warmup_iter=0),

              ssl_args=dict(alg='vat',
                            coef=0.3,
                            ema_teacher=False,
                            ema_teacher_warmup=False,
                            ema_teacher_factor=0.999,
                            ema_apply_wd=False,
                            em=0,
                            threshold=None,
                            sharpen=None,
                            temp_softmax=None,
                            consis='ce',
                            eps=6,
                            xi=1e-6,
                            vat_iter=1
                            ),

              ssl_eval_args=dict(weight_average=False,
                                 wa_ema_factor=0.999,
                                 wa_apply_wd=False),

              dss_args=dict(type="RETRIEVE",
                            fraction=0.1,
                            select_every=20,
                            kappa=0,
                            linear_layer=False,
                            selection_type='Supervised',
                            greedy='Stochastic',
                            valid=True),

              train_args=dict(iteration=500000,
                              max_iter=-1,
                              device="cuda",
                              results_dir='results/',
                              disp=256,
                              seed=96)
              )


Configuration files for HPO
----------------------------
    .. code-block:: python
    
        from ray import tune

        config = dict(setting= "hyperparamtuning",
            # parameter for subset selection
            # all settings for subset selection will be fetched from here
            subset_config = "configs/SL/config_gradmatchpb-warm_cifar100.py",
            # parameters for hyper-parameter tuning
            # search space for hyper-parameter tuning
            space = dict(
                    learning_rate=tune.uniform(0.001, 0.01), 
                    learning_rate1=tune.uniform(0.001, 0.01),
                    learning_rate2=tune.uniform(0.001, 0.01),
                    learning_rate3=tune.uniform(0.001, 0.01),
                    scheduler= tune.choice(['cosine_annealing', 'linear_decay']),
                    nesterov= tune.choice([True, False]),
                    gamma= tune.uniform(0.05, 0.5),    
                    ),

            # tuning algorithm 
            search_algo = "TPE",

            # number of hyper-parameter set to try
            num_evals = 27,

            # metric to be optimized, for 'mean_loss' metric mode should be 'min'
            metric = "mean_accuracy",
            mode = "max",

            # scheduler to be used (i.e ASHAScheduler)
            # scheduler terminates trials that perform poorly
            # learn more here: https://docs.ray.io/en/releases-0.7.1/tune-schedulers.html
            scheduler = 'asha',

            # where to store logs
            log_dir = "RayLogs/",

            # resume hyper-parameter tuning from previous log
            # specify 'name' (i.e main_2021-03-09_18-33-56) below
            resume = False,

            # only required if you want to resume from previous checkpoint
            # it can also be specified if you don't want to resume
            name = None,

            # specify resources to be used per trial
            # i.e {'gpu':1, 'cpu':2}
            resources = {'gpu':0.5},

            # if True, trains model on Full dataset with the best parameter selected.
            final_train = True
            )

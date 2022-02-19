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


    **Symbol (%) represents mandatory arguments**

    **model**

    #. architecture % 
        * Model architecture to be used, Presently it supports the below mentioned architectures.
            #. resnet18
            #. two_layer_net
    #. target_classes %
        * Number of output classes for prediction. 
    #. input_dim
        * Input dimension of the dataset. To be mentioned while using two layer net.
    #. hidden_units_1
        * Number of hidden units to be used in the first layer. To be mentioned while using two layer net.

    **train_parameters**

    #. lr %
        * Learning rate to be used for training.
    #. batch_size %
        * Batch size to be used for training.
    #. n_epoch %
        * Maximum number of epochs for the model to train.
    #. max_accuracy
        * Maximum training accuracy after which training should be stopped.
    #. isreset
        * Reset weight whenever the model training starts.
            #. True
            #. False
    #. islogs
        * Log training output.
            #. True
            #. False
    #. logs_location %
        * Location where logs should be saved.

    **active_learning**

    #. strategy %
        * Active learning strategy to be used.
            #. badge
            #. glister
            #. entropy_sampling
            #. margin_sampling
            #. least_confidence
            #. core_set
            #. random_sampling
            #. fass
            #. bald_dropout
            #. adversarial_bim
            #. kmeans_sampling
            #. baseline_sampling
            #. adversarial_deepfool
    #. budget %
        * Number of points to be selected by the active learning strategy.
    #. rounds %
        * Total number of rounds to run active learning for.
    #. initial_points
        * Initial number of points to start training with.
    #. strategy_args
        * Arguments to pass to the strategy. It varies from strategy to strategy. Please refer to the documentation of the strategy that is being used.

    **dataset**

    #. name
        * Name of the dataset to be used. It presently supports following datasets.
            #. cifar10
            #. mnist
            #. fmnist
            #. svhn
            #. cifar100
            #. satimage
            #. ijcnn1

    You can refer to various configuration examples in the configs/ folders of the DISTIL repository.
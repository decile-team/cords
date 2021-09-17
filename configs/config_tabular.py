# Learning setting
# file: data/airline.pickle, input_dim: 499, n_classes: 2
config = dict(setting="supervisedlearning",

              dataset=dict(name="airline",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=20,
                              pin_memory=True),

              model=dict(architecture='TwoLayerNet',
                         type='pre-defined',
                         input_dim=499,
                         numclasses=2,
                         hidden_units=200
                         ),

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

              dss_strategy=dict(type="GLISTER-Warm",
                                fraction=0.1,
                                valid=False,
                                lam=0.5,
                                select_every=20,
                                kappa=0.6),

              train_args=dict(num_epochs=200,
                              device="cuda:0",
                              # print_every=10,
                              print_every=1,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )

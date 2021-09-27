# Learning setting

config = dict(setting="supervisedlearning",

              dataset=dict(name="airline",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=128,
                              pin_memory=True),

              model=dict(architecture='LSTM',
                         type='pre-defined',
                         vocab_size=499,
                         hidden_units=256,
                         num_layers=1,
                         # embed_dim=64,
                         embed_dim=100,
                         ),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.1,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_strategy=dict(type="GLISTER-Warm",
                                fraction=0.1,
                                select_every=5,
                                kappa=0.6,
                                valid=False),

              # train_args=dict(num_epochs=200,
              train_args=dict(num_epochs=80,
                              device="cuda:0",
                              print_every=1,
                              results_dir='results/',
                              print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )

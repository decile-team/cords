# Learning setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="mnist",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=20,
                              pin_memory=True),

              model=dict(architecture='SimpleNN',
                         type='pre-defined',
                         input_dim=784,
                         hidden_units=2048,
                         numclasses=10
                         ),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='../results/',
                        save_every=20),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.01,
                             weight_decay=5e-4),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_strategy=dict(type="GLISTER",
                                fraction=0.01,
                                select_every=5),

              train_args=dict(
                              num_epochs=20,
                              device="cpu",
                              print_every=1,
                              results_dir='../results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )

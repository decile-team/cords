# Learning setting
# vocab_size = self.configdata["model"]["vocab_size"]
# hidden_units = self.configdata["model"]["hidden_units"]
# num_layers = self.configdata["model"]["num_layers"]
# embed_dim = self.configdata["model"]["embed_dim"]
# num_classes = self.configdata["model"]["numclasses"]

config = dict(setting="supervisedlearning",

              dataset=dict(name="airline",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=20,
                              pin_memory=True),

              model=dict(architecture='LSTM',
                         type='pre-defined',
                         vocab_size=499,
                         hidden_units=2,
                         num_layers=200,
                         embed_dim=200,
                         numclasses=200,
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
                                select_every=20,
                                kappa=0.6),

              train_args=dict(num_epochs=80,
                              device="cuda",
                              print_every=1,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )

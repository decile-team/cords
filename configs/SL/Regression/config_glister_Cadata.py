# Learning setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="cadata",
                           datadir="reg_data",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=200,
                              pin_memory=True),

              model=dict(architecture='RegressionNet',
                         type='pre-defined',
                         numclasses=1),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),

              loss=dict(type='MeanSquaredLoss',
                        use_sigmoid=False),

              optimizer=dict(type='sgd',#"adam",
                             momentum=0.9,
                             lr=0.001,
                             weight_decay=5e-4),

              scheduler=dict(type="None",
                             T_max=500),

              dss_args=dict(type="GLISTER",
                            fraction=0.1,
                            select_every=20,
                            kappa=0,
                            linear_layer=True,#False,
                            selection_type='Supervised',
                            greedy='Stochastic'),

              train_args=dict(num_epochs=500,
                              device="cuda",
                              print_every=10,
                              results_dir='results/',
                              print_args=["val_loss",  "tst_loss", "time"],
                              return_args=[]
                              )
              )

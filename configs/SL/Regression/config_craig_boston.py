# Learning setting
config = dict(setting="SL",
              is_reg = True,
              dataset=dict(name="boston",
                           datadir="../data",
                           feature="dss",
                           type="tabular"),

              dataloader=dict(shuffle=True,
                              batch_size=20,
                              pin_memory=True),

              model=dict(architecture='ThreeLayerNet',
                         type='pre-defined',
                         input_dim=13,
                         numclasses=1,
                         h1 = 16,
                         h2 = 32),
              
              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),
              
              loss=dict(type='MeanSquaredLoss',
                        use_sigmoid=False),

              optimizer=dict(type="adam",
                             lr=1e-2),

              scheduler=dict(type="none"),

              dss_args=dict(type="CRAIG",
                            fraction=0.1,
                            select_every=20,
                            kappa=0,
                            linear_layer=False,
                            optimizer='lazy',
                            selection_type='Supervised'),

              train_args=dict(num_epochs=300,
                              device="cuda",
                              print_every=10,
                              results_dir='results/',
                              print_args=["val_loss", "tst_loss", "trn_loss", "time"],
                              return_args=[]
                              )
              )

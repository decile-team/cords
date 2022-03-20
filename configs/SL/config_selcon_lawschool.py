# Learning setting
config = dict(setting="SL",

              dataset=dict(name="LawSchool",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=100,
                              pin_memory=False),

              model=dict(architecture='RegressionNet',
                         type='pre-defined',
                         numclasses=10), # verify
              
              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),
              
              loss=dict(type='MSELoss', # was CrossEntropyLoss for others
                        use_sigmoid=False),

              optimizer=dict(type="adam", 
                             lr=0.01),

              scheduler=dict(type="StepLR", # added this new scheduler type
                             step_size=1,
                             gamma=0.1),

              dss_args=dict(type="SELCON", # todo : modify this to SELCON
                                fraction=0.01,
                                select_every=20,
                                kappa=0,
                                delta=0.04,
                                linear_layer=False,
                                lam=1e-5,
                                batch_sampler='sequential',
                                selection_type='Supervised'),

              train_args=dict(num_epochs=100,
                              device="cuda",
                              print_every=1,
                              results_dir='results/',
                              print_args=["val_loss", "tst_loss", "trn_loss", "time"],
                              return_args=[]
                              )
              )

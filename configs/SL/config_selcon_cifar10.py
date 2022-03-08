# Learning setting
config = dict(setting="SL",

              dataset=dict(name="cifar10",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=100,
                              pin_memory=True),

              model=dict(architecture='ResNet18',
                         type='pre-defined',
                         numclasses=10),
              
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
                                fraction=0.1,
                                select_every=20,
                                kappa=0,
                                delta=0.4,
                                linear_layer=False,
                                selection_type='Supervised'),

              train_args=dict(num_epochs=10,
                              device="cuda",
                              print_every=10,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )

# Learning setting
from cords.utils.data.data_utils.collate import *

config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="glue_sst2",
                           datadir="/home/ayush/Documents/abhishek/data/SST/",
                           feature="dss",
                           type="text",
                           wordvec_dim=300,
                           weight_path='/home/ayush/Documents/abhishek/data/glove.6B/',),

              dataloader=dict(shuffle=True,
                              batch_size=16,
                              pin_memory=True,
                              collate_fn = collate_fn_pad_batch),

              model=dict(architecture='LSTM',
                         type='pre-defined',
                         numclasses=2,
                         wordvec_dim=300,
                         weight_path='/home/ayush/Documents/abhishek/data/glove.6B/',
                         hidden_size=128,
                         num_layers=1),

              ckpt=dict(is_load=False,
                        is_save=False,
                        dir='results/',
                        save_every=5),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="adam",
                             momentum=0.9,
                             lr=0.001,
                             weight_decay=5e-4),

              scheduler=dict(type=None,
                            #  type="cosine_annealing",
                             T_max=300),

              dss_args=dict(type="GradMatchPB-Warm",
                            fraction=0.3,
                            select_every=5,
                            lam=0,
                            selection_type='PerBatch',
                            v1=True,
                            valid=False,
                            eps=1e-100,
                            linear_layer=True,
                            kappa=0.5,
                            collate_fn = collate_fn_pad_batch),

              train_args=dict(num_epochs=12,
                              device="cuda",
                              print_every=1,
                              results_dir='results/',
                              print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "time"],
                              return_args=[]
                              )
              )

# Learning setting
from cords.utils.data.data_utils.collate import *

config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="sst2",
                           datadir="/home/kk/cords/data/SST/",
                           feature="dss",
                           type="text",
                           wordvec_dim=300,
                           weight_path='/home/kk/cords/data/glove.6B/',),

              dataloader=dict(shuffle=True,
                              batch_size=16,
                              pin_memory=True,
                              collate_fn = collate_fn_pad_batch),

              model=dict(architecture='LSTM',
                         type='pre-defined',
                         numclasses=2,
                         wordvec_dim=300,
                         weight_path='/home/kk/cords/data/glove.6B/',
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

              dss_args=dict(type="Random",
                            fraction=0.3,
                            select_every=5,
                            kappa=0,
                            collate_fn = collate_fn_pad_batch),

              train_args=dict(num_epochs=20,
                              device="cuda",
                              print_every=3,
                              results_dir='results/',
                              print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )

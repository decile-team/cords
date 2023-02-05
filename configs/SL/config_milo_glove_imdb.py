from cords.utils.data.data_utils.collate import *
import argparse, os


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Global ordering of a dataset using pretrained LMs.")
    parser.add_argument(
                        "--dataset",
                        type=str,
                        default="imdb",
                        help="Only supports datasets for hugging face currently."
                        )
    parser.add_argument(
                        "--model",
                        type=str,
                        default="all-distilroberta-v1",
                        help="Transformer model used for computing the embeddings."
                        )
    
    parser.add_argument(
                        "--data_dir",
                        type=str,
                        required=False,
                        help="Directory in which data downloads happen.",
                        default="../data"
                        ) 
    parser.add_argument(
                        "--submod_function",
                        type=str,
                        default="fl",
                        help="Submdular function used for finding the global order."
                        )
    parser.add_argument(
                        "--seed",
                        type=int,
                        default=42,
                        help="Seed value for reproducibility of the experiments."
                        )
    parser.add_argument(
                        "--device",
                        type=str,
                        default='cuda:0',
                        help= "Device used for computing the embeddings"
                        )
    parser.add_argument(
                        "--kw",
                        type=float,
                        default=0.1,
                        help= "Multiplier for RBF Kernel"
                        )
    parser.add_argument(
                        "--r2_coefficient",
                        type=float,
                        default=3,
                        help= "Multiplier for R2 Variant"
                        )
    parser.add_argument(
                        "--knn",
                        type=int,
                        default=25,
                        help= "No of nearest neighbors for KNN variant"
                        )
    args=parser.parse_args()
    return args

args = parse_args()

config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="imdb",
                           datadir="../data/imdb/",
                           feature="dss",
                           type="text",
                           wordvec_dim=300,
                           weight_path='../data/glove.6B/',),

              dataloader=dict(shuffle=True,
                              batch_size=16,
                              pin_memory=True,
                              collate_fn = collate_fn_pad_batch),

              model=dict(architecture='LSTM',
                         type='pre-defined',
                         numclasses=2,
                         wordvec_dim=300,
                         weight_path='../data/glove.6B/',
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

              dss_args=dict(type="MILO",
                            kw=0.1,
                            global_order_file=os.path.join(os.path.abspath(args.data_dir), args.dataset + '_' + args.model + '_' + args.submod_function + '_' + str(args.kw) + '_global_order.pkl'),
                            gc_ratio=1/6,
                            gc_stochastic_subsets_file=os.path.join(os.path.abspath(args.data_dir), args.dataset + '_' + args.model + '_' + args.submod_function + '_' + str(args.kw) + '_0.1_stochastic_subsets.pkl'),
                            fraction=0.1,
                            select_every=1,
                            submod_function = 'fl',
                            lam=0,
                            selection_type='PerBatch',
                            v1=True,
                            valid=False,
                            eps=1e-100,
                            linear_layer=True,
                            kappa=0,
                            per_class=True,
                            temperature=1,
                            collate_fn = collate_fn_pad_batch
                            ),

              train_args=dict(num_epochs=20,
                              device="cuda",
                              print_every=1,
                              run=1,
                              results_dir='results/',
                              wandb=False,
                            #   print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )

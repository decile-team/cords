import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Global ordering of a dataset using pretrained LMs.")
    parser.add_argument(
                        "--dataset",
                        type=str,
                        default="cifar100",
                        help="Only supports datasets for hugging face currently."
                        )
    parser.add_argument(
                        "--model",
                        type=str,
                        default="clip-ViT-L-14",
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

# Learning setting
config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="cifar100",
                           datadir="../data",
                           feature="dss",
                           type="image"),

              dataloader=dict(shuffle=True,
                              batch_size=128,
                              pin_memory=True),

              model=dict(architecture='ResNet18',
                         type='pre-defined',
                         numclasses=100),
              
              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),
              
              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.05,
                             weight_decay=5e-4,
                             nesterov=True),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_args=dict(type="MILO",
                            fraction=0.1,
                            kw=0.1,
                            global_order_file=os.path.join(os.path.abspath(args.data_dir), args.dataset + '_' + args.model + '_' + args.submod_function + '_' + str(args.kw) + '_global_order.pkl'),
                            submod_function = 'fl',
                            select_every=1,
                            kappa=0,
                            per_class=True,
                            temperature=1,
                            collate_fn = None),

              train_args=dict(num_epochs=300,
                              device="cuda",
                              print_every=1,
                              wandb=False,
                              run=1,
                              results_dir='results/',
                              print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )

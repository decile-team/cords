import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument("--root", "-r", default="./data", type=str, help="/path/to/dataset")
    parser.add_argument("--dataset", "-d", default="cifar10", choices=['stl10', 'svhn', 'cifar10', 'cifar100', 'cifarOOD', 'mnistOOD', 'cifarImbalance'], type=str, help="dataset name")
    parser.add_argument("--num_labels", default=4000, type=int, help="number of labeled data")
    parser.add_argument("--ood_ratio", default=0.5, type=float, help="the ratio of OOD in unlabeled data")
    parser.add_argument("--val_ratio", default=0.1, type=float, help="the ratio of evaluation data to training data.")
    parser.add_argument("--random_split", action="store_true", help="random sampling from training data for validation")
    parser.add_argument("--num_workers", default=8, type=int, help="number of thread for CPU parallel")
    parser.add_argument("--whiten", action="store_true", help="use whitening as preprocessing")
    parser.add_argument("--zca", action="store_true", help="use zca whitening as preprocessing")
    parser.add_argument("--ood", action="store_true", help="OOD indicator")
    parser.add_argument("--classimb", action="store_true", help="ClassImbalance indicator")
    # augmentation config
    parser.add_argument("--labeled_aug", default="WA", choices=['WA', 'RA'], type=str, help="type of augmentation for labeled data")
    parser.add_argument("--unlabeled_aug", default="WA", choices=['WA', 'RA'], type=str, help="type of augmentation for unlabeled data")
    parser.add_argument("--wa", default="t.t.f", type=str, help="transformations (flip, crop, noise) for weak augmentation. t and f indicate true and false.")
    parser.add_argument("--strong_aug", action="store_true", help="use strong augmentation (RandAugment) for unlabeled data")
    # optimization config
    parser.add_argument("--model", default="wrn", choices=['wrn', 'shake', 'cnn13', 'cnn'], type=str, help="model architecture")
    parser.add_argument("--ul_batch_size", "-ul_bs", default=50, type=int, help="mini-batch size of unlabeled data")
    parser.add_argument("--l_batch_size", "-l_bs", default=50, type=int, help="mini-batch size of labeled data")
    parser.add_argument("--optimizer", "-opt", default="sgd", choices=['sgd', 'adam'], type=str, help="optimizer")
    parser.add_argument("--lr", default=3e-2, type=float, help="learning rate")
    parser.add_argument("--weight_decay", "-wd", default=0.0005, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for sgd or beta_1 for adam")
    parser.add_argument("--iteration", default=500000, type=int, help="number of training iteration")
    parser.add_argument("--lr_decay", default="cos", choices=['cos', 'step'], type=str, help="way to decay learning rate")
    parser.add_argument("--lr_decay_rate", default=0.2, type=float, help="decay rate for step lr decay")
    parser.add_argument("--only_validation", action="store_true", help="only training and validation for hyperparameter tuning")
    parser.add_argument("--warmup_iter", default=0, type=int, help="number of warmup iteration for SSL loss coefficient")
    parser.add_argument("--tsa", action="store_true", help="use training signal annealing proposed by UDA")
    parser.add_argument("--tsa_schedule", default="linear", choices=['linear', 'exp', 'log'], type=str, help="tsa schedule")
    # SSL common config
    parser.add_argument("--alg", default="cr", choices=['ict', 'cr', 'pl', 'vat'], type=str, help="ssl algorithm")
    parser.add_argument("--coef", default=1, type=float, help="coefficient for consistency loss")
    parser.add_argument("--ema_teacher", action="store_true", help="use mean teacher")
    parser.add_argument("--ema_teacher_warmup", action="store_true", help="warmup for mean teacher")
    parser.add_argument("--ema_teacher_factor", default=0.999, type=float, help="exponential mean avarage factor for mean teacher")
    parser.add_argument("--ema_apply_wd", action="store_true", help="apply weight decay to ema model")
    parser.add_argument("--entropy_minimization", "-em", default=0, type=float, help="coefficient of entropy minimization")
    parser.add_argument("--threshold", default=None, type=float, help="pseudo label threshold")
    parser.add_argument("--sharpen", default=None, type=float, help="temperature parameter for sharpening")
    parser.add_argument("--temp_softmax", default=None, type=float, help="temperature for softmax")
    parser.add_argument("--consistency", "-consis", default="ce", choices=['ce', 'ms', 'kld'], type=str, help="consistency type")
    ## SSL alg parameter
    ### ICT config
    parser.add_argument("--alpha", default=0.1, type=float, help="parameter for beta distribution in ICT")
    ### VAT config
    parser.add_argument("--eps", default=6, type=float, help="norm of virtual adversarial noise")
    parser.add_argument("--xi", default=1e-6, type=float, help="perturbation for finite difference method")
    parser.add_argument("--vat_iter", default=1, type=int, help="number of iteration for power iteration")
    #evaluation config
    parser.add_argument("--weight_average", action="store_true", help="evaluation with weight-averaged model")
    parser.add_argument("--wa_ema_factor", default=0.999, type=float, help="exponential mean avarage factor for weight-averaged model")
    parser.add_argument("--wa_apply_wd", action="store_true", help="apply weight decay to weight-averaged model")
    parser.add_argument("--checkpoint", default=10000, type=int, help="checkpoint every N samples")
    #subset selection arguments
    parser.add_argument("--dss_strategy", default='GradMatchPB', type=str, help="Data Subset Selection Strategy")
    parser.add_argument("--fraction", default=0.1, type=float, help="Unlabeled dataset fraction")
    parser.add_argument("--select_every", default=20, type=int, help="subset selection every N epochs")
    parser.add_argument("--kappa", default=0.5, type=float, help="Kappa value for Warm Variants")
    parser.add_argument("--valid", action="store_true", help="Use Validation Set for Gradient Matching")
    parser.add_argument("--max_iter", default=-1, type=int, help="Use max iterations if not -1")
    # training from checkpoint
    parser.add_argument("--checkpoint_model", default=None, type=str, help="path to checkpoint model")
    parser.add_argument("--checkpoint_optimizer", default=None, type=str, help="path to checkpoint optimizer")
    parser.add_argument("--start_iter", default=None, type=int, help="start iteration")
    # misc
    parser.add_argument("--out_dir", default="log", type=str, help="output directory")
    parser.add_argument("--seed", default=96, type=int, help="random seed")
    parser.add_argument("--disp", default=256, type=int, help="display loss every N")
    return parser.parse_args()

from vision_train_sl import TrainClassifier
import argparse
from vision_paramtuning import HyperParamTuning
from cords.utils.config_utils import load_config_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=float, default=0.1, help='fraction in subset selection')
    parser.add_argument('--select_every', type=int, default=20, help='perform subset selection every _ epochs')
    parser.add_argument('--ft_type', type=str, default='full', help='final_train type after hp tuning. full/gmpb')
    parser.add_argument('--change', type=int, default=1, help='change params mentioned for train class?')
    
    parser.add_argument('--config_file', type=str, default='/home/kk/cords/configs/SL/config_gradmatchpb-warm_cifar100.py')
    parser.add_argument('--config_hp', type=str, default='/home/kk/cords/configs/SL/config_hyper_param_tuning_cifar100.py')
    parser.add_argument('--scheduler', type=str, default='asha')
    parser.add_argument('--search_algo', type=str, default='TPE')
    parser.add_argument('--num_evals', type=int, default=27)
    parser.add_argument('--is_hp', type=int, default=1, help='do we perform hyper parameter tuning?')
    parser.add_argument('--final_train', type=int, default=1, help='need final training hyper parameter tuning?')

    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--kappa', type=float, default=0.35)
    parser.add_argument('--dataset', type=str, default='svhn')
    parser.add_argument('--lr', type=float, default=0.01, help='lr')
    parser.add_argument('--lr1', type=float, default=0.01, help='lr1')
    parser.add_argument('--lr2', type=float, default=0.01, help='lr2')
    parser.add_argument('--lr3', type=float, default=0.01, help='lr3')
    parser.add_argument('--train_scheduler', type=str, default='cosine_annealing', help='learning rate scheduler')
    parser.add_argument('--nesterov', type=bool, help='use nesterov momentum')
    parser.add_argument('--gamma', type=float, default=0.1, help='step size decay rate')
    args = parser.parse_args()

    if bool(args.is_hp):
        config_hp_data = load_config_data(args.config_hp)
        config_hp_data.final_train = bool(args.final_train)
        config_hp_data.final_train_type = args.ft_type
        config_hp_data.subset_config = args.config_file
        config_hp_data.scheduler = args.scheduler
        config_hp_data.search_algo = args.search_algo
        config_hp_data.num_evals = args.num_evals
        train_config_data = load_config_data(args.config_file)
        if bool(args.change):
            train_config_data.dss_args.fraction = args.fraction
            train_config_data.dss_args.select_every = args.select_every
            train_config_data.report_tune = True
            train_config_data.model.numclasses = args.num_classes
            train_config_data.dataset.name = args.dataset
            train_config_data.dss_args.kappa = args.kappa
            train_config_data.optimizer.lr = args.lr
            train_config_data.optimizer.lr1 = args.lr1
            train_config_data.optimizer.lr2 = args.lr2
            train_config_data.optimizer.lr3 = args.lr3
            train_config_data.optimizer.nesterov = args.nesterov
            train_config_data.scheduler.type = args.train_scheduler
            train_config_data.scheduler.gamma = args.gamma   
            #train_config_data.train_args.device = 'cuda'
        hyperparamtuning = HyperParamTuning(config_hp_data, train_config_data)
        hyperparamtuning.start_eval()
    else:
        train_config_data = load_config_data(args.config_file)
        if bool(args.change):
            train_config_data.dss_args.fraction = args.fraction
            train_config_data.dss_args.select_every = args.select_every
            train_config_data.model.numclasses = args.num_classes
            train_config_data.dataset.name = args.dataset
            train_config_data.dss_args.kappa = args.kappa
            train_config_data.optimizer.lr = args.lr
            train_config_data.optimizer.lr1 = args.lr1
            train_config_data.optimizer.lr2 = args.lr2
            train_config_data.optimizer.lr3 = args.lr3
            train_config_data.optimizer.nesterov = args.nesterov
            train_config_data.scheduler.type = args.train_scheduler
            train_config_data.scheduler.gamma = args.gamma
            #config_file_data.train_args.device = 'cuda'
        classifier = TrainClassifier(train_config_data)
        classifier.train()

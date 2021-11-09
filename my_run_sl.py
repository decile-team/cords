from my_train_sl import TrainClassifier
import argparse
from my_paramtuning import HyperParamTuning
from cords.utils.config_utils import load_config_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=128)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--print_every', type=int, default=3)
    parser.add_argument('--wordvec_dim', type=int, default=300, help='Dimension of GloVe vectors')

    parser.add_argument('--fraction', type=float, default=0.1, help='fraction in subset selection')
    parser.add_argument('--select_every', type=int, default=5, help='perform subset selection every _ epochs')
    parser.add_argument('--change', type=int, default=1, help='change params mentioned for train class?')

    parser.add_argument('--config_file', type=str, default='/home/kk/cords/configs/SL/config_random_glove_sst2.py')
    parser.add_argument('--config_hp', type=str, default='/home/kk/cords/configs/SL/config_hp.py')
    parser.add_argument('--is_hp', type=int, default=0, help='do we perform hyper parameter tuning?')
    parser.add_argument('--final_train', type=int, default=1, help='need final training hyper parameter tuning?')
    args = parser.parse_args()
    weight_path = '/home/kk/cords/data/glove.6B/'

    if bool(args.is_hp):
        config_hp_data = load_config_data(args.config_hp)
        config_hp_data.final_train = bool(args.final_train)
        config_hp_data.subset_config = args.config_file
        
        train_config_data = load_config_data(args.config_file)
        if bool(args.change):
            # train_config_data.optimizer.lr = args.lr
            # train_config_data.dataloader.batchsize = args.batch_size
            # train_config_data.model.hidden_size = args.hidden_size

            # train_config_data.train_args.num_epochs = args.epochs
            # train_config_data.train_args.print_every = args.print_every
            # train_config_data.dataset.wordvec_dim = args.wordvec_dim
            # train_config_data.dataset.weight_path = weight_path
            # train_config_data.model.wordvec_dim = args.wordvec_dim
            # train_config_data.model.weight_path = weight_path

            train_config_data.dss_args.fraction = args.fraction
            train_config_data.dss_args.select_every = args.select_every
            train_config_data.train_args.device = 'cuda'
            

        hyperparamtuning = HyperParamTuning(config_hp_data, train_config_data)
        hyperparamtuning.start_eval()
    else:

        config_file_data = load_config_data(args.config_file)
        if bool(args.change):
            # config_file_data.optimizer.lr = args.lr
            # config_file_data.dataloader.batchsize = args.batch_size
            # config_file_data.model.hidden_size = args.hidden_size

            # config_file_data.train_args.num_epochs = args.epochs
            # config_file_data.train_args.print_every = args.print_every
            # config_file_data.dataset.wordvec_dim = args.wordvec_dim
            # config_file_data.dataset.weight_path = weight_path
            # config_file_data.model.wordvec_dim = args.wordvec_dim
            # config_file_data.model.weight_path = weight_path

            config_file_data.dss_args.fraction = args.fraction
            config_file_data.dss_args.select_every = args.select_every
            config_file_data.train_args.device = 'cuda'

        classifier = TrainClassifier(config_file_data)
        classifier.train()

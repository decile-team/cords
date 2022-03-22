from train_ssl import TrainClassifier
from cords.utils.config_utils import load_config_data

def main():
    config_file = "configs/SSL/config_gradmatch_vat_cifar10.py"
    config_data = load_config_data(config_file)
    classifier = TrainClassifier(config_data)
    classifier.cfg.dss_args.fraction = 0.1
    classifier.cfg.dss_args.select_every = 1
    classifier.cfg.train_args.device = 'cuda'
    classifier.cfg.train_args.print_every = 1
    classifier.train()


if __name__ == "__main__":
    main()    
from train_ssl import TrainClassifier


def main():
    config_file = "configs/SSL/config_gradmatch_vat_cifar10.py"
    classifier = TrainClassifier(config_file)
    classifier.cfg.dss_args.fraction = 0.1
    classifier.cfg.dss_args.select_every = 1
    classifier.cfg.train_args.device = 'cuda'
    classifier.cfg.train_args.print_every = 1
    classifier.train()


if __name__ == "__main__":
    main()    
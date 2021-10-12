from train_ssl import TrainClassifier
config_file = "configs/SSL/config_retrieve_vat_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.cfg.dss_args.fraction = 0.1
classifier.cfg.dss_args.select_every = 1
classifier.cfg.train_args.device = 'cuda:1'
classifier.cfg.train_args.print_every = 1
classifier.train()

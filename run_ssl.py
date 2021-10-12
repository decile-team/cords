from train_ssl import TrainClassifier
config_file = "configs/SSL/config_retrieve-warm_vat_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.cfg.dss_args.fraction = 0.1
classifier.cfg.dss_args.select_every = 20
classifier.cfg.train_args.device = 'cuda:2'
classifier.cfg.train_args.print_every = 1
classifier.train()

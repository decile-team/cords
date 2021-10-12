from train_sl import TrainClassifier
config_file = "configs/SL/config_glister_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.cfg.dss_args.fraction = 0.1
classifier.cfg.dss_args.select_every = 1
classifier.cfg.train_args.device = 'cuda:0'
classifier.cfg.train_args.print_every = 1
classifier.train()

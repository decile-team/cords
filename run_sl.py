from train_sl import TrainClassifier
config_file = "configs/SL/config_glister_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.configdata.dss_args.fraction = 0.1
classifier.configdata.dss_args.select_every = 1
classifier.configdata.train_args.device = 'cuda:1'
classifier.configdata.train_args.print_every = 1
classifier.train()

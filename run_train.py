from train import TrainClassifier
config_file = "configs/config_glister_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.configdata['dss_strategy']['select_every'] = 20
classifier.configdata.train_args.device = 'cuda:1'
classifier.configdata.train_args.print_every = 1
classifier.train()

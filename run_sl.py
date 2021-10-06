from train_sl import TrainClassifier
config_file = "configs/SL/config_craig_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.configdata.train_args.device = 'cuda:0'
classifier.configdata.train_args.print_every = 1
classifier.train()

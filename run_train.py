from train import TrainClassifier
# config_file = "configs/config_gradmatch_cifar10.py"
config_file = "configs/config_glister_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.configdata['dss_strategy']['select_every'] = 1
classifier.configdata['model']['architecture'] = 'MobileNet2'
classifier.configdata['optimizer']['weight_decay'] = 4e-5
classifier.configdata['train_args']['device'] = 'cpu'
classifier.train()


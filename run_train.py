from train import TrainClassifier
config_file = "configs/config_gradmatch_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.configdata['dss_strategy']['select_every'] = 1
classifier.train()

from train import TrainClassifier
config_file = "configs/config_gradmatch_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.configdata['dss_strategy']['select_every'] = 20
classifier.configdata['dss_strategy']['lam'] = 0
classifier.configdata['dss_strategy']['nnls'] = True
# classifier.configdata['model']['architecture'] = 'MobileNet2'
# classifier.configdata['optimizer']['weight_decay'] = 4e-5
classifier.train()

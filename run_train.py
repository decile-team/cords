from train import TrainClassifier
config_file = "configs/config_tabular.py"
classifier = TrainClassifier(config_file)
classifier.configdata['dss_strategy']['select_every'] = 5
classifier.configdata['dss_strategy']['fraction'] = 0.3
classifier.configdata['dss_strategy']['type'] = 'GLISTER'
classifier.train()

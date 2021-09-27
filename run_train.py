# from train import TrainClassifier
# config_file = "configs/config_tabular.py"
# classifier = TrainClassifier(config_file)
# classifier.configdata['dss_strategy']['select_every'] = 5
# classifier.configdata['dss_strategy']['fraction'] = 0.3
# classifier.configdata['dss_strategy']['type'] = 'GLISTER'
# classifier.train()

# from train import TrainClassifier
# config_file = "configs/config_random_cifar10.py"
# classifier = TrainClassifier(config_file)
# classifier.configdata["train_args"]["device"] = "cpu"
# classifier.configdata["dss_strategy"]["fraction"] = 0.001
# classifier.train()

# # airline
# from train import TrainClassifier
#
# config_file = "configs/config_tabular.py"
# classifier_airline = TrainClassifier(config_file)
# classifier_airline = TrainClassifier(config_file)
# classifier_airline.configdata["dataset"]["name"] = "airline"
# classifier_airline.configdata["dss_strategy"]["type"] = "GLISTER"
# classifier_airline.configdata["dss_strategy"]["fraction"] = 0.1
# # classifier_airline.configdata["dss_strategy"]["fraction"] = 0.001
# # classifier_airline.configdata["dss_strategy"]["select_every"] = 20
# classifier_airline.configdata["dss_strategy"]["select_every"] = 5
# classifier_airline.configdata["model"]["input_dim"] = 499
# classifier_airline.configdata["model"]["numclasses"] = 2
# classifier_airline.configdata["train_args"]["device"] = "cpu"
# classifier_airline.train()

# airline
from train import TrainClassifier

config_file = "configs/config_tabular.py"
classifier_airline = TrainClassifier(config_file)
classifier_airline = TrainClassifier(config_file)
classifier_airline.configdata["dataset"]["name"] = "airline"
#################################################################
classifier_airline.configdata["dss_strategy"]["type"] = "GradMatchPB"
classifier_airline.configdata['dss_strategy']['valid'] = False
classifier_airline.configdata['dss_strategy']['lam'] = 0.5
#################################################################
# classifier_airline.configdata["dss_strategy"]["fraction"] = 0.1
classifier_airline.configdata["dss_strategy"]["fraction"] = 0.2
classifier_airline.configdata["dss_strategy"]["select_every"] = 5
classifier_airline.configdata["model"]["input_dim"] = 499
classifier_airline.configdata["model"]["numclasses"] = 2
classifier_airline.configdata["train_args"]["device"] = "cpu"

classifier_airline.train()

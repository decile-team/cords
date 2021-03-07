from train import TrainClassifier

config_file = "configs/config_gradmatchpb_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.train()

from cords.utils.config_utils import load_config_data
from train_sl import TrainClassifier
config_file = "configs/SL/config_selcon_lawschool.py"
config_data = load_config_data(config_file)
classifier = TrainClassifier(config_data)
# classifier.cfg.dss_args.fraction = 0.01
# classifier.cfg.dss_args.select_every = 20
# classifier.cfg.train_args.device = 'cuda'
# classifier.cfg.train_args.print_every = 1
classifier.train()

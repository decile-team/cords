from cords.utils.config_utils import load_config_data
from train_sl import TrainClassifier

fractions = [0.01, 0.03, 0.05]
select_every = [35]
num_epochs = [50, 100, 200, 500]
delta = [0.01, 0.04, 0.1]

config_file = "configs/SL/config_selcon_lawschool.py"
config_data = load_config_data(config_file)
classifier = TrainClassifier(config_data)

classifier.cfg.train_args.print_every = 50

for f in fractions:
    for se in select_every:
        for ne in num_epochs:
            for d in delta:
                print(f"Hyperparms:\tfraction:{f}\tselect every:{se}\tnum epochs:{ne}\tdelta:{d}")
                classifier.cfg.dss_args.fraction = f
                classifier.cfg.dss_args.select_every = se
                classifier.cfg.dss_args.delta = d
                classifier.cfg.train_args.num_epochs = ne
                try:
                    classifier.train()
                except:
                    pass

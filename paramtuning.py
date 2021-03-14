import argparse
from cords.utils.config_utils import load_config_data
from configs import paramtune_search_space
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray import tune
import sys
from train import TrainClassifier

class HyperParamTuning:

    def __init__(self, config_file):
        self.config_data = load_config_data(config_file)
        # train_config_data = load_config_data(self.config_data['subset_config'])
        self.train_class = TrainClassifier(self.config_data['subset_config'])
        self.search_algo = self.get_search_algo(self.config_data['search_algo'], self.config_data['space'], self.config_data['metric'], self.config_data['mode'])

    def param_tune(self, config):
        #update parameters in config dict
        if 'learning_rate' in config:
            self.train_class.configdata['optimizer']['lr'] = config['learning_rate']
        if 'optimizer' in config:
            self.train_class.configdata['optimizer']['type'] = config['optimizer']
        if 'trn_batch_size' in config:
            self.train_class.configdata['dataloader']['batch_size'] = config['trn_batch_size']
        self.train_class.train()

    def start_eval(self):
        analysis = tune.run(
            self.param_tune,
            num_samples=self.config_data['num_evals'],
            config=self.config_data['space'],
            search_alg=self.search_algo,
            resources_per_trial={'gpu':1},
            local_dir=self.config_data['log_dir']+self.config_data['subset_method']+'/',
            log_to_file=True)
    
        print("Best Config: ", analysis.get_best_config(metric=self.config_data['metric'], mode=self.config_data['mode']))

    def get_search_algo(self, method, space, metric, mode):
        if method == "hyperopt" or method == "TPE":
            search = HyperOptSearch(space, metric="mean_accuracy", mode="max")
        return search


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_file", default="configs/paramtune_search_space.py")    
    args = argparser.parse_args()

    hyperparam_tuning = HyperParamTuning(args.config_file) 
    hyperparam_tuning.start_eval()

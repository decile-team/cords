import argparse
from cords.utils.config_utils import load_config_data

from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.zoopt import ZOOptSearch
from ray.tune.suggest.sigopt import SigOptSearch
from ray.tune.suggest.hebo import HEBOSearch

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB

from ray import tune
import sys
from train import TrainClassifier

class HyperParamTuning:
    def __init__(self, config_file):
        self.config_data = load_config_data(config_file)
        self.train_class = TrainClassifier(self.config_data['subset_config'])
        self.search_algo = self.get_search_algo(self.config_data['search_algo'], self.config_data['space'], self.config_data['metric'], self.config_data['mode'])
        self.scheduler = self.get_scheduler(self.config_data['scheduler'], self.config_data['metric'], self.config_data['mode'])
        # save subset method, to be used in log dir name
        self.subset_method = self.train_class.configdata['dss_strategy']['type']

    def param_tune(self, config):
        #update parameters in config dict
        new_config = self.update_parameters(self.train_class.configdata, config)
        self.train_class.configdata = new_config
        # turn on reporting to ray every time
        self.train_class.configdata['report_tune'] = True
        self.train_class.train()

    def start_eval(self):
        analysis = tune.run(
            self.param_tune,
            num_samples=self.config_data['num_evals'],
            config=self.config_data['space'],
            search_alg=self.search_algo,
            scheduler=self.scheduler,
            resources_per_trial={'gpu':1},
            local_dir=self.config_data['log_dir']+self.subset_method+'/',
            log_to_file=True,
            name=self.config_data['name'],
            resume=self.config_data['resume'])
    
        best_config = analysis.get_best_config(metric=self.config_data['metric'], mode=self.config_data['mode'])
        print("Best Config: ", best_config)

        if self.config_data['final_train']:
            self.final_train(best_config)

    def get_search_algo(self, method, space, metric, mode):
        
        # HyperOptSearch 
        if method == "hyperopt" or method == "TPE":
            search = HyperOptSearch(space, metric = metric, mode = mode)
        # BayesOptSearch
        elif method == "bayesopt" or method == "BO":
            search = BayesOptSearch(space, metric = metric, mode = mode)
        # SkoptSearch
        elif method == "skopt" or method == "SKBO":
            search = SkOptSearch(space, metric = metric, mode = mode)
        # DragonflySearch
        elif method == "dragonfly" or method == "SBO":
            search = DragonflySearch(space, metric = metric, mode = mode)
        # AxSearch
        elif method == "ax" or method == "BBO":
            search = AxSearch(space, metric = metric, mode = mode)
        # TuneBOHB
        elif method == "tunebohb" or method == "BOHB":
            search = TuneBOHB(space, metric = metric, mode = mode)
        # NevergradSearch
        elif method == "nevergrad" or method == "GFO":
            search = NevergradSearch(space, metric = metric, mode = mode)
        # OptunaSearch
        elif method == "optuna" or method == "OSA":
            search = OptunaSearch(space, metric = metric, mode = mode)
        # ZOOptSearch
        elif method == "zoopt" or method == "ZOO":
            search = ZOOptSearch(space, metric = metric, mode = mode)
        # SigOptSearch
        elif method == "sigopt":
            search = SigOptSearch(space, metric = metric, mode = mode)
        # HEBOSearch
        elif method == "hebo" or method == "HEBO":
            search = HEBOSearch(space, metric = metric, mode = mode)
        else:
            return None

        return search

    def get_scheduler(self, method, metric, mode):

        if method == "ASHA":
            scheduler = AsyncHyperBandScheduler(metric = metric, mode = mode)
        elif method == "hyperband" or method == "HB":
            scheduler = HyperBandScheduler(metric = metric, mode = mode)
        elif method == "BOHB":
            scheduler = HyperBandForBOHB(metric = metric, mode = mode)
        else:
            return None
        
        return scheduler
    
    def final_train(self, best_params):
        # change strategy to Full (i.e use whole dataset)
        # update (optimized) parameters
        new_config = self.update_parameters(self.train_class.configdata, best_params)
        self.train_class.configdata = new_config
        self.train_class.configdata['dss_strategy']['type'] = 'Full'

        self.train_class.train()
    
    def update_parameters(self, config, new_config):
        # a generic function to update parameters
        if 'learning_rate' in new_config:
            config['optimizer']['lr'] = new_config['learning_rate']
        if 'optimizer' in new_config:
            config['optimizer']['type'] = new_config['optimizer']
        if 'trn_batch_size' in new_config:
            config['dataloader']['batch_size'] = new_config['trn_batch_size']
        
        return config
        



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_file", default="configs/config_hyper_param_tuning.py")    
    args = argparser.parse_args()

    hyperparam_tuning = HyperParamTuning(args.config_file) 
    hyperparam_tuning.start_eval()
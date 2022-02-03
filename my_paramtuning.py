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
from ray.tune.suggest import BasicVariantGenerator
from ray import tune
import ray
import sys
from my_train_sl import TrainClassifier

# @ray.remote(num_gpus=1)
class HyperParamTuning:
    def __init__(self, config_file_data, train_config_data):
        # self.cfg = load_config_data(config_file)
        self.cfg = config_file_data
        self.train_class = TrainClassifier(train_config_data)
        # self.train_class = TrainClassifier(self.cfg['subset_config'])
        self.train_class.cfg.train_args.print_every = 1
        self.search_algo = self.get_search_algo(self.cfg.search_algo, self.cfg.space, self.cfg.metric, self.cfg.mode)
        self.scheduler = self.get_scheduler(self.cfg.scheduler, self.cfg.metric, self.cfg.mode)
        # save subset method, to be used in log dir name
        self.subset_method = self.train_class.cfg.dss_args.type

    def param_tune(self, config):
        #update parameters in config dict
        new_config = self.update_parameters(self.train_class.cfg, config)
        self.train_class.cfg = new_config
        # turn on reporting to ray every time
        self.train_class.cfg.report_tune = True
        self.train_class.train()

    def start_eval(self):
        if self.search_algo is None:
            analysis = tune.run(
                self.param_tune,
                num_samples=self.cfg.num_evals,
                config=self.cfg.space,
                search_alg=self.search_algo,
                scheduler=self.scheduler,
                resources_per_trial=self.cfg.resources,
                local_dir=self.cfg.log_dir+self.subset_method+'/',
                log_to_file=True,
                name=self.cfg.name,
                resume=self.cfg.resume)
        else:
            analysis = tune.run(
                self.param_tune,
                num_samples=self.cfg.num_evals,
                search_alg=self.search_algo,
                scheduler=self.scheduler,
                resources_per_trial=self.cfg.resources,
                local_dir=self.cfg.log_dir+self.subset_method+'/',
                log_to_file=True,
                name=self.cfg.name,
                resume=self.cfg.resume)
        best_config = analysis.get_best_config(metric=self.cfg.metric, mode=self.cfg.mode)
        print("Best Config: ", best_config)

        if self.cfg.final_train:
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
            scheduler = AsyncHyperBandScheduler(metric = metric, mode = mode,
                        max_t = self.train_class.cfg.train_args.num_epochs)
        elif method == "hyperband" or method == "HB":
            scheduler = HyperBandScheduler(metric = metric, mode = mode, 
                        max_t = self.train_class.cfg.train_args.num_epochs)
        elif method == "BOHB":
            scheduler = HyperBandForBOHB(metric = metric, mode = mode)
        else:
            return None
        
        return scheduler
    
    def final_train(self, best_params):
        # change strategy to Full (i.e use whole dataset)
        # update (optimized) parameters
        new_config = self.update_parameters(self.train_class.cfg, best_params)
        new_config.dss_args.type = 'Full'
        # new_config.dss_args.type = 'GradMatchPB'
        # new_config.dss_args.fraction = 0.3
        # new_config.dss_args.select_every = 5
        # new_config.dss_args.lam = 0
        # new_config.dss_args.selection_type = 'PerBatch'
        # new_config.dss_args.v1 = True
        # new_config.dss_args.valid = False
        # new_config.dss_args.eps = 1e-100
        # new_config.dss_args.linear_layer = True
        # new_config.dss_args.kappa = 0
        self.train_class.cfg = new_config
        self.train_class.train()
    
    def update_parameters(self, config, new_config):
        # a generic function to update parameters
        if 'trn_batch_size' in new_config:
            config['dataloader']['batch_size'] = new_config['trn_batch_size']
        if 'learning_rate' in new_config:
            config['optimizer']['lr'] = new_config['learning_rate']
        if 'learning_rate1' in new_config:
            config['optimizer']['lr1'] = new_config['learning_rate1']
        if 'learning_rate2' in new_config:
            config['optimizer']['lr2'] = new_config['learning_rate2']
        if 'learning_rate3' in new_config:
            config['optimizer']['lr3'] = new_config['learning_rate3']
        if 'optimizer' in new_config:
            config['optimizer']['type'] = new_config['optimizer']
        if 'nesterov' in new_config:
            config['optimizer']['nesterov'] = new_config['nesterov']
        if 'scheduler' in new_config:
            config['scheduler']['type'] = new_config['scheduler']
        if 'trn_batch_size' in new_config:
            config['dataloader']['batch_size'] = new_config['trn_batch_size']
        if 'gamma' in new_config:
            config['scheduler']['gamma'] = new_config['gamma']
        if 'epochs' in new_config:
            config.train_args.num_epochs = new_config['epochs']
        if 'trn_batch_size' in new_config:
            config.dataloader.batch_size = new_config['trn_batch_size']
        if 'hidden_size' in new_config:
            config.model.hidden_size = new_config['hidden_size']
        if 'num_layers' in new_config:
            config.model.num_layers = new_config['num_layers']
        return config
        



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_file", default="configs/config_hyper_param_tuning.py")    
    args = argparser.parse_args()

    hyperparam_tuning = HyperParamTuning(load_config_data(args.config_file))
    hyperparam_tuning.start_eval()
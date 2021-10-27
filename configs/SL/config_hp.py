from ray import tune

config = dict(setting= "hyperparamtuning",

# parameter for subset selection
# all settings for subset selection will be fetched from here
subset_config = "configs/SL/config_gradmatchpb_glove_sst2.py",

# parameters for hyper-parameter tuning
# search space for hyper-parameter tuning
space = dict(learning_rate=tune.uniform(0.001, 0.01), 
        # optimizer= tune.choice(['sgd', 'adam']),
        epochs = tune.choice([10, 20, 30]),
        trn_batch_size= tune.choice([16, 32, 64]),        
        ),

# tuning algorithm 
search_algo = "TPE",

# number of hyper-parameter set to try
num_evals = 25,

# metric to be optimized, for 'mean_loss' metric mode should be 'min'
metric = "mean_accuracy",
mode = "max",

# scheduler to be used (i.e ASHAScheduler)
# scheduler terminates trials that perform poorly
# learn more here: https://docs.ray.io/en/releases-0.7.1/tune-schedulers.html
scheduler = 'hyperband',

# where to store logs
log_dir = "RayLogs/",

# resume hyper-parameter tuning from previous log
# specify 'name' (i.e main_2021-03-09_18-33-56) below
resume = False,

# only required if you want to resume from previous checkpoint
# it can also be specified if you don't want to resume
name = None,

# specify resources to be used per trial
# i.e {'gpu':1, 'cpu':2}
resources = {'gpu' : 1},

# if True, trains model on Full dataset with the best parameter selected.
final_train = True

)

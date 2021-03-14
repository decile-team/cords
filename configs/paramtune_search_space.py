from hyperopt import hp
config = dict(setting= "supervisedlearning",

space = dict(learning_rate=hp.uniform('lr', 0.001, 0.01), 
        optimizer= hp.choice('optimizer', ['sgd', 'adam']),
        trn_batch_size= hp.choice('trn_batch_size', [20, 32, 64])),

# parameters
search_algo = "TPE",
num_evals = 20,

log_dir = "/content/drive/MyDrive/RayLogs/",
subset_config = "configs/config_gradmatchpb-warm_cifar10.py",
subset_method = "gradmatchpb-warm",
metric = "mean_accuracy",
mode = "max",
resume = False

)

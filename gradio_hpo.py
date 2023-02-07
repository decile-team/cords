import gradio as gr
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data
from cords.utils.data.datasets.SL import gen_dataset
import os, sys
from dotmap import DotMap
import pandas as pd
from matplotlib import pyplot as plt
from fastapi import FastAPI

# CUSTOM_PATH = "/gradio_hpo"
# app = FastAPI()

def return_search_space(dset):
    if dset == 'TREC6':
        return {
                'num_layers': [1],
                'hidden_size': [64, 128],
                'learning_rate': [0.001, 0.01, 0.1]
                }

def return_full_results():
    search_space = {
                'num_layers': [1, 2],
                'hidden_size': [64, 128, 256],
                'learning_rate': [0.001, 0.01, 0.1]
                }
    
    search_configs = []
    key1, key2, key3 = list(search_space.keys())[0], list(search_space.keys())[1], list(search_space.keys())[2]
    for i in range(len(search_space[key1])):
        for j in range(len(search_space[key2])):
            for k in range(len(search_space[key3])):
                search_configs.append({key1: search_space[key1][i], key2: search_space[key2][j], key3: search_space[key3][k]})

    full_results = [[0.866, 287.729421377182], 
                            [0.874, 287.24425220489502], [0.276, 287.74825239181519], [0.868, 287.75174808502197], [0.858, 287.92841958999634], [0.374, 287.94037413597107], [0.856, 287.2798364162445], [0.874, 287.7269721031189], [0.276, 287.57475852966309], [0.878, 287.33644080162048], [0.864, 287.51423025131226], [0.278, 287.54987382888794], [0.86, 287.80005693435669], [0.888, 287.19879388809204], [0.286, 287.0971245765686], [0.886, 287.03492903709412], [0.87, 287.60258483886719], [0.412, 287.89436173439026]]

    full_results_dict = {}
    for i in range(len(full_results)):
        full_results_dict[tuple(sorted(search_configs[i].items()))] = full_results[i]
    return full_results_dict


def update_model_text(choice):
    if choice == "MNIST":
      return gr.update(choices=['LeNet'], value="LeNet"), return_search_space(choice)
    elif choice == "TREC6":
        return gr.update(choices=['LSTM'], value="LSTM"), return_search_space(choice)
    elif choice == 'CIFAR10':
        return gr.update(choices=['ResNet18'], value="ResNet18"), return_search_space(choice)


def update_out_text(choice):
    # print(choice)
    return gr.update(label='Best Test Accuracy obtained by '+ str(choice)), gr.update(label="Time taken by " + str(choice) + " in seconds")


def hpo(dset, ml_model, strategy, budget):
    metric = 'cossim'
    kw  =  0.01
    search_space = return_search_space(dset)
    dset = dset.lower()

    if dset in ['mnist', 'cifar10']:
        feat_model = 'dino_cls'
    elif dset in ['trec6']:
        feat_model = 'all-distilroberta-v1'
    
    temperature  =  1
    strategy = strategy.lower()
    per_class = True
    if ml_model == 'LeNet':
        ml_model = 'MnistNet'
    else:
        ml_model = ml_model
    run_cnt = 0
    if strategy in ['gradmatchpb', 'craigpb', 'glister']:
        if dset in ['trec6']:
            select_every = 3
        else:
            select_every = 3
    else:
        select_every = 1

    submod_function = 'disp_min_pc'
    data_dir = '../data'
    stochastic_subsets_file = os.path.join(os.path.abspath(data_dir), dset + '_' + feat_model + '_' + metric + '_' + 'gc_pc' + '_' + str(kw) + '_' + str(budget/100) + '_stochastic_subsets.pkl')
    gc_stochastic_subsets_file = os.path.join(os.path.abspath(data_dir), dset + '_' + feat_model + '_' + metric + '_' + 'gc_pc' + '_' + str(kw) + '_' + str(budget/100) + '_stochastic_subsets.pkl')
    global_order_file = os.path.join(os.path.abspath(data_dir), dset + '_' + feat_model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl')
    
    #Pre-trained HPO full results
    full_results_dict = return_full_results()
    
    #Naive code for now! Will have an improved version soon
    search_configs = []
    key1, key2, key3 = list(search_space.keys())[0], list(search_space.keys())[1], list(search_space.keys())[2]
    for i in range(len(search_space[key1])):
        for j in range(len(search_space[key2])):
            for k in range(len(search_space[key3])):
                search_configs.append({key1: search_space[key1][i], key2: search_space[key2][j], key3: search_space[key3][k]})

    trainset, validset, testset, num_cls = gen_dataset('../data/TREC6',
                                                       'hf_trec6',
                                                        'dss', 
                                                        dataset=DotMap(dict(name="hf_trec6",
                                                                    datadir="../data/TREC6/",
                                                                    feature="dss",
                                                                    type="text",
                                                                    wordvec_dim=300,
                                                                    weight_path='../data/glove.6B/',)))
    results_dict = {}
    for strat in [strategy]:
        if dset in ['trec6']:
            config_file = "configs/SL/config_" + strat + "_glove_" + dset + ".py"
        else:
            config_file = "configs/SL/config_" + strat + "_" + dset + ".py"
        config_data = load_config_data(config_file)
        config_data.train_args.device = 'cuda'
        config_data.train_args.run = run_cnt
        config_data.train_args.wandb = False
        if dset in ['trec6']:
            config_data.train_args.num_epochs = 20
            config_data.train_args.print_every = 5
            config_data.dataloader.batch_size = 16
        else:
            config_data.train_args.num_epochs = 10
            config_data.train_args.print_every = 1
            config_data.scheduler.T_max = 10
            config_data.scheduler.type = "cosine_annealing"
            config_data.optimizer.type = 'sgd'
            config_data.optimizer.lr = 5e-2
            config_data.dataloader.batch_size = 128
        config_data.train_args.print_args=["tst_loss", "tst_acc", "time"]
        config_data.dss_args.fraction = (budget/100)    
        config_data.dss_args.global_order_file = global_order_file
        config_data.dss_args.gc_stochastic_subsets_file = gc_stochastic_subsets_file
        config_data.dss_args.stochastic_subsets_file = stochastic_subsets_file
        config_data.dss_args.gc_ratio = 0.1
        config_data.dss_args.kw = kw
        config_data.dss_args.per_class = per_class
        config_data.dss_args.temperature = temperature
        config_data.dss_args.submod_function = submod_function
        config_data.model.architecture = ml_model
        config_data.dss_args.select_every = select_every
        classifier = TrainClassifier(config_data)
        results_dict[strat] = []
        for search_config in search_configs:
            if dset in ['trec6']:
                if "num_layers" in search_config:
                    classifier.cfg.model.num_layes = search_config["num_layers"]

                if "hidden_size" in search_config:
                    classifier.cfg.model.hidden_size = search_config["hidden_size"]

                if "learning_rate" in search_config:
                    classifier.cfg.optimizer.lr = search_config["learning_rate"]    
            trn_acc, val_acc, tst_acc, best_acc, omp_cum_timing = classifier.train(trainset=trainset, validset=validset, testset=testset, num_cls=num_cls)
            results_dict[strat].append([best_acc[-1], omp_cum_timing[-1]])
    
    best_strat_acc = -1
    best_full_acc = -1
    best_strat_idx = -1
    strat_tuning_time = 0
    full_tuning_time = 0

    for i in range(len(search_configs)):
        strat_tuning_time += results_dict[strategy][i][1]
        full_tuning_time += full_results_dict[tuple(sorted(search_configs[i].items()))][1]

        if results_dict[strategy][i][0] > best_strat_acc:
            best_strat_acc = results_dict[strategy][i][0]
            best_strat_config = search_configs[i]
        
        if full_results_dict[tuple(sorted(search_configs[i].items()))][0] > best_full_acc:
            best_full_acc = full_results_dict[tuple(sorted(search_configs[i].items()))][0]
            best_full_config = search_configs[i]

    return best_strat_config, full_results_dict[tuple(sorted(best_strat_config.items()))][0], strat_tuning_time , best_full_config, best_full_acc, full_tuning_time


with gr.Blocks(title = "Hyper-parameter Optimization") as demo:
    with gr.Row():
        with gr.Column():
            dset = gr.Dropdown(choices=['TREC6'], label='Dataset Name')
            model = gr.Radio(["LSTM"], label="Model Architecture")
            strategy = gr.Dropdown(choices=['Random', 'AdaptiveRandom', 'MILO', 'WRE', 'SGE', 'MILOFixed', 'GradMatchPB', 'CraigPB', 'GLISTER'], label='Subset Selection Strategy')
            budget = gr.Slider(minimum=1, maximum=100, label='Budget (in %)')
            search_space = gr.JSON(label='Hyper-parameter Search Space')
            submit = gr.Button(value="Perform Grid Search")
        with gr.Column():
            # df1 = gr.DataFrame(label='Hyper-parameter Optimization Results')
            with gr.Row():
                strat_config = gr.JSON(label='Best Configuration obtained using selected strategy')
                strat_acc = gr.Number(label="Best Validation Accuracy using selected strategy")
                strat_timing = gr.Number(label="Time taken for tuning using selected strategy in seconds")
            with gr.Row():
                full_config = gr.JSON(label='Best Configuration obtained using Full')
                full_acc = gr.Number(label="Best Validation Accuracy using Full")
                full_timing = gr.Number(label="Time taken for tuning using Full")
    dset.change(fn=update_model_text, inputs=dset, outputs=[model, search_space])

    # strategy.change(fn=update_out_text, inputs=strategy, outputs=[strat_acc, strat_timing])
    
    submit.click(fn=hpo, inputs=[dset, model, strategy, budget], outputs=[strat_config, strat_acc, strat_timing, full_config, full_acc, full_timing])
demo.launch()
# app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)
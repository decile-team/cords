import gradio as gr
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data
import os, sys
import pandas as pd
from matplotlib import pyplot as plt

def return_search_space(dset):
    if dset == 'TREC6':
        return {
                'num_layers': [1, 2],
                'hidden_size': [64, 128, 256]
                }

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
    
    #Naive code for now! Will have an improved version soon
    search_configs = []
    key1, key2 = list(search_space.keys())[0], list(search_space.keys())[1]
    for i in range(len(search_space[key1])):
        for j in range(len(search_space[key2])):
            search_configs.append({key1: search_space[key1][i], key2: search_space[key2][j]})

    results_dict = {}
    for strat in [strategy, 'full']:
        results_dict[strat] = []
        for search_config in search_configs:
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
                config_data.train_args.print_every = 1
                config_data.dataloader.batch_size = 16
                if "num_layers" in search_config:
                    config_data.model.num_layes = search_config["num_layers"]

                if "hidden_size" in search_config:
                    config_data.model.hidden_size = search_config["hidden_size"]
            else:
                config_data.train_args.num_epochs = 10
                config_data.train_args.print_every = 1
                config_data.scheduler.T_max = 10
                config_data.scheduler.type = "cosine_annealing"
                config_data.optimizer.type = 'sgd'
                config_data.optimizer.lr = 5e-2
                config_data.dataloader.batch_size = 128
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
            trn_acc, val_acc, tst_acc, best_acc, omp_cum_timing = classifier.train()
            results_dict[strat].append([best_acc[-1], omp_cum_timing[-1]])
    
    # fig, ax = plt.subplots()
    hpo_dict = {key1:[], key2:[], "Validation Accuracy using selected strategy": [], "Full training Validation Accuracy": []}
    best_strat_acc = -1
    best_full_acc = -1
    strat_tuning_time = 0
    full_tuning_time = 0

    for i in range(len(search_configs)):
        hpo_dict[key1].append(search_configs[i][key1])
        hpo_dict[key2].append(search_configs[i][key2])
        hpo_dict["Validation Accuracy using selected strategy"].append(results_dict[strategy][i][0])
        hpo_dict["Full training Validation Accuracy"].append(results_dict['full'][i][0])
        strat_tuning_time += results_dict[strategy][i][1]
        full_tuning_time += results_dict['full'][i][1]

        if results_dict[strategy][i][0] > best_strat_acc:
            best_strat_acc = results_dict[strategy][i][0]
            best_strat_config = search_config
        
        if results_dict['full'][i][0] > best_full_acc:
            best_full_acc = results_dict['full'][i][0]
            best_full_config = search_config

    df1 = pd.DataFrame.from_dict(hpo_dict)
    return df1, best_strat_config, strat_tuning_time, best_full_config, full_tuning_time


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
            df1 = gr.DataFrame(label='Hyper-parameter Optimization Results')
            with gr.Row():
                strat_acc = gr.JSON(label='Best Configuration obtained using selected strategy')
                strat_timing = gr.Number(label="Time taken for tuning by selected strategy in seconds")
            with gr.Row():
                full_acc = gr.JSON(label='Best Configuration obtained using full')
                full_timing = gr.Number(label="Time taken for tuning by full in seconds")
    dset.change(fn=update_model_text, inputs=dset, outputs=[model, search_space])

    # strategy.change(fn=update_out_text, inputs=strategy, outputs=[strat_acc, strat_timing])
    
    submit.click(fn=hpo, inputs=[dset, model, strategy, budget], outputs=[df1, strat_acc, strat_timing, full_acc, full_timing])
demo.launch()

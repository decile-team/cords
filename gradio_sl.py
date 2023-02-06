import gradio as gr
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data
import os, sys
import matplotlib
from matplotlib import pyplot as plt

def train_model(dset, ml_model, strategy, budget):
    metric = 'cossim'
    kw  =  0.01
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
            select_every = 10
    else:
        select_every = 1
    submod_function = 'disp_min_pc'
    data_dir = '../data'
    stochastic_subsets_file = os.path.join(os.path.abspath(data_dir), dset + '_' + feat_model + '_' + metric + '_' + 'gc_pc' + '_' + str(kw) + '_' + str(budget/100) + '_stochastic_subsets.pkl')
    gc_stochastic_subsets_file = os.path.join(os.path.abspath(data_dir), dset + '_' + feat_model + '_' + metric + '_' + 'gc_pc' + '_' + str(kw) + '_' + str(budget/100) + '_stochastic_subsets.pkl')
    global_order_file = os.path.join(os.path.abspath(data_dir), dset + '_' + feat_model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl')
    results_dict = {}
    for strat in [strategy, 'full']:
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
            # if strat == 'full':
            #     config_data.train_args.num_epochs = 5
            config_data.train_args.print_every = 1
            config_data.dataloader.batch_size = 16
        else:
            config_data.train_args.num_epochs = 10
            config_data.train_args.print_every = 1
            config_data.scheduler.T_max = 10
            # if strat == 'full':
            #     config_data.train_args.num_epochs = 5
            #     config_data.train_args.print_every = 1
            #     config_data.scheduler.T_max = 5
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
        results_dict[strat] = [trn_acc, val_acc, tst_acc, best_acc, omp_cum_timing]
    
    fig, ax = plt.subplots()

    for strat in [strategy, 'full']:
        ax.plot(results_dict[strat][-1], results_dict[strat][3], marker=None, linestyle='-', linewidth=3, markersize=10, label=strat.upper())

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(prop={'size': 18},loc='best', frameon=False, handlelength=0.4, fontsize=27)
    plt.xlabel('Time taken (in Secs)', fontsize=24, labelpad=15)
    plt.ylabel('Test Accuracy', fontsize=24)
    plt.grid(axis='y',linestyle='-', linewidth=0.5)
    plt.grid(axis='x',linestyle='-', linewidth=0.5)
    plt.box(on=True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    return fig, results_dict[strategy][3][-1], results_dict[strategy][-1][-1], results_dict['full'][3][-1], results_dict['full'][-1][-1]


def update_model_text(choice):
    if choice == "MNIST":
      return gr.update(choices=['LeNet'], value="LeNet")
    elif choice == "TREC6":
        return gr.update(choices=['LSTM'], value="LSTM")
    elif choice == 'CIFAR10':
        return gr.update(choices=['ResNet18'], value="ResNet18")


def update_out_text(choice):
    # print(choice)
    return gr.update(label='Best Test Accuracy obtained by '+ str(choice)), gr.update(label="Time taken by " + str(choice) + " in seconds")


with gr.Blocks(title = "Classifier Training") as demo:
    with gr.Row():
        with gr.Column():
            dset = gr.Dropdown(choices=['MNIST', 'TREC6', 'CIFAR10'], label='Dataset Name')
            model = gr.Radio(["LeNet", "LSTM", "ResNet18"], label="Model Architecture")
            strategy = gr.Dropdown(choices=['Random', 'AdaptiveRandom', 'MILO', 'WRE', 'SGE', 'MILOFixed', 'GradMatchPB', 'CraigPB', 'GLISTER'], label='Subset Selection Strategy')
            budget = gr.Slider(minimum=1, maximum=100, label='Budget (in %)')
            submit = gr.Button(value="Train Model")
        with gr.Column():
            plot = gr.Plot(label='Convergence Curves')
            with gr.Row():
                strat_acc = gr.Number(label='Test Accuracy obtained by selected strategy')
                strat_timing = gr.Number(label="Time taken by selected strategy in seconds")
            with gr.Row():
                full_acc = gr.Number(label='Test Accuracy obtained by full')
                full_timing = gr.Number(label="Time taken by full in seconds")
    dset.change(fn=update_model_text, inputs=dset, outputs=model)

    # strategy.change(fn=update_out_text, inputs=strategy, outputs=[strat_acc, strat_timing])
    
    submit.click(fn=train_model, inputs=[dset, model, strategy, budget], outputs=[plot, strat_acc, strat_timing, full_acc, full_timing])
demo.launch()
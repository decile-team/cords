import gradio as gr
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data
import os, sys
import pandas as pd
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
    # Pre-trained Full Runs   
    if dset == 'mnist':
        results_dict['full'] = [[], [], [0.115 , 0.9789 , 0.9827 , 0.9861 , 0.9832 , 0.9849 , 0.9861 , 0.988 , 0.9874 , 0.9901 , 0.9868 , 0.9864 , 0.9882 , 0.9866 , 0.988 , 0.9892 , 0.9891 , 0.9899 , 0.9878 , 0.9898 , 0.9905 , 0.991 , 0.9887 , 0.9901 , 0.9888 , 0.9906 , 0.9908 , 0.9892 , 0.9905 , 0.9916 , 0.9906 , 0.9907 , 0.9901 , 0.9917 , 0.9925 , 0.9921 , 0.9916 , 0.9922 , 0.9919 , 0.9915 , 0.992 , 0.992 , 0.9918 , 0.992 , 0.9915 , 0.9919 , 0.9918 , 0.9917 , 0.9918 , 0.9918 , 0.9918], 
        [0.115 , 0.9789 , 0.9827 , 0.9861 , 0.9861 , 0.9861 , 0.9861 , 0.988 , 0.988 , 0.9901 , 0.9901 , 0.9901 , 0.9901 , 0.9901 , 0.9901 , 0.9901 , 0.9901 , 0.9901 , 0.9901 , 0.9901 , 0.9905 , 0.991 , 0.991 , 0.991 , 0.991 , 0.991 , 0.991 , 0.991 , 0.991 , 0.9916 , 0.9916 , 0.9916 , 0.9916 , 0.9917 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925 , 0.9925], 
        [0 , 14.389130115509033 , 28.760189294815063 , 43.04677176475525 , 57.34686803817749 , 71.64872980117798 , 86.07794213294983 , 100.39223647117615 , 114.78056263923645 , 129.3134298324585 , 143.80610918998718 , 158.21936583518982 , 172.56326031684875 , 187.00763177871704 , 201.51328587532043 , 216.04423141479492 , 230.46502709388733 , 244.85350632667542 , 259.26120710372925 , 273.64386916160583 , 288.0403118133545 , 302.349862575531 , 316.72026205062866 , 331.15454030036926 , 345.6315724849701 , 360.1029076576233 , 374.5343918800354 , 388.9018943309784 , 403.3672478199005 , 417.79295468330383 , 432.27216124534607 , 446.7308900356293 , 461.1081693172455 , 475.48193740844727 , 489.92033982276917 , 504.29099321365356 , 518.7385609149933 , 533.1931157112122 , 548.4689939022064 , 563.1488707065582 , 577.4998891353607 , 591.9615330696106 , 606.4804599285126 , 620.812980890274 , 635.2540225982666 , 649.7750537395477 , 664.1136946678162 , 678.4558100700378 , 692.8708505630493 , 707.2919359207153 , 721.8091855049133]]
    
    if dset == 'trec6':
        results_dict['full']= [[], [], [0.276 , 0.272 , 0.386 , 0.406 , 0.572 , 0.812 , 0.816 , 0.844 , 0.852 , 0.864 , 0.858 , 0.86 , 0.856 , 0.852 , 0.872 , 0.862 , 0.852 , 0.864 , 0.85 , 0.874 , 0.862], [0.276 , 0.276 , 0.386 , 0.406 , 0.572 , 0.812 , 0.816 , 0.844 , 0.852 , 0.864 , 0.864 , 0.864 , 0.864 , 0.864 , 0.872 , 0.872 , 0.872 , 0.872 , 0.872 , 0.874 , 0.874],
        [0 , 14.302120447158813 , 28.568480014801025 , 42.83177137374878 , 57.09772491455078 , 71.36303496360779 , 85.70327568054199 , 100.13131880760193 , 114.56033658981323 , 128.99154114723206 , 143.42615032196045 , 157.8575897216797 , 172.2942578792572 , 186.73123860359192 , 201.16823267936707 , 215.60326504707336 , 230.04524636268616 , 244.4774525165558 , 258.89773440361023 , 273.30184841156006 , 287.69908905029297]]
    
    if dset == 'cifar10':
        results_dict['full'] = [[], [], [0.1091 , 0.4922 , 0.6473 , 0.6424 , 0.7345 , 0.773 , 0.8007 , 0.8084 , 0.8106 , 0.8107 , 0.8413 , 0.8004 , 0.8509 , 0.8669 , 0.835 , 0.8398 , 0.8738 , 0.8732 , 0.8359 , 0.8331 , 0.8751 , 0.8556 , 0.8802 , 0.8899 , 0.8775 , 0.8866 , 0.8777 , 0.8952 , 0.8935 , 0.8957 , 0.8999 , 0.9051 , 0.899 , 0.9134 , 0.9107 , 0.9135 , 0.918 , 0.9232 , 0.9273 , 0.9287 , 0.9303 , 0.9353 , 0.9372 , 0.9362 , 0.9366 , 0.9381 , 0.9394 , 0.9387 , 0.9387 , 0.9384 , 0.9383],
         [0.1091 , 0.4922 , 0.6473 , 0.6473 , 0.7345 , 0.773 , 0.8007 , 0.8084 , 0.8106 , 0.8107 , 0.8413 , 0.8413 , 0.8509 , 0.8669 , 0.8669 , 0.8669 , 0.8738 , 0.8738 , 0.8738 , 0.8738 , 0.8751 , 0.8751 , 0.8802 , 0.8899 , 0.8899 , 0.8899 , 0.8899 , 0.8952 , 0.8952 , 0.8957 , 0.8999 , 0.9051 , 0.9051 , 0.9134 , 0.9134 , 0.9135 , 0.918 , 0.9232 , 0.9273 , 0.9287 , 0.9303 , 0.9353 , 0.9372 , 0.9372 , 0.9372 , 0.9381 , 0.9394 , 0.9394 , 0.9394 , 0.9394 , 0.9394],
         [0 , 51.98912858963013 , 103.96543478965759 , 156.05607771873474 , 208.0158998966217 , 259.933513879776 , 311.9376890659332 , 363.79946732521057 , 415.7738411426544 , 467.86292028427124 , 519.9261643886566 , 571.9207320213318 , 624.090588092804 , 676.4059774875641 , 728.5902750492096 , 780.7833154201508 , 833.0365333557129 , 885.2812588214874 , 937.6802985668182 , 989.459990978241 , 1041.473831653595 , 1093.491066455841 , 1145.7025349140167 , 1197.7461080551147 , 1249.7591750621796 , 1301.9592096805573 , 1353.8092248439789 , 1405.6296582221985 , 1457.5615322589874 , 1509.4092235565186 , 1561.2287991046906 , 1613.277204990387 , 1665.5947706699371 , 1717.441056728363 , 1769.4924371242523 , 1821.5556795597076 , 1873.6177792549133 , 1925.6751248836517 , 1977.7274763584137 , 2029.6933171749115 , 2081.691498041153 , 2133.5037961006165 , 2185.614940404892 , 2237.853115081787 , 2290.0417246818542 , 2342.3756618499756 , 2394.863676548004 , 2446.7810294628143 , 2498.856172800064 , 2551.005379676819 , 2603.0458319187164]]

    results_df = pd.DataFrame({'Strategy': ['Full']*len(results_dict['full'][-1]), 'Time': results_dict['full'][-1], 'Accuracy': results_dict['full'][3]})
    for strat in [strategy]: #, 'full']:
        if dset in ['trec6']:
            config_file = "configs/SL/config_" + strat + "_glove_" + dset + ".py"
        else:
            config_file = "configs/SL/config_" + strat + "_" + dset + ".py"
        config_data = load_config_data(config_file)
        config_data.train_args.device = 'cuda'
        config_data.train_args.run = run_cnt
        config_data.train_args.wandb = False
        config_data.train_args.print_args=["tst_loss", "tst_acc", "time"]
        if dset in ['trec6']:
            config_data.train_args.num_epochs = 20
            # if strat == 'full':
            #     config_data.train_args.num_epochs = 5
            config_data.train_args.print_every = 5
            config_data.dataloader.batch_size = 16
        elif dset in ['mnist']:
            config_data.train_args.num_epochs = 50
            config_data.train_args.print_every = 10
            config_data.scheduler.T_max = 50
            # if strat == 'full':
            #     config_data.train_args.num_epochs = 5
            #     config_data.train_args.print_every = 1
            #     config_data.scheduler.T_max = 5
            config_data.scheduler.type = "cosine_annealing"
            
            config_data.optimizer.type = 'sgd'
            config_data.optimizer.lr = 5e-2
            config_data.dataloader.batch_size = 128
        elif dset in ['cifar10']:
            config_data.train_args.num_epochs = 10
            config_data.train_args.print_every = 10
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
        results_df = results_df.append(pd.DataFrame({'Strategy': [strat.upper()]*len(results_dict[strat][-1]), 'Time': results_dict[strat][-1], 'Accuracy': results_dict[strat][3]}))
    
    fig = gr.LinePlot.update(
            results_df,
            x="Time",
            y="Accuracy",
            color="Strategy",
            title="Test Accuracy Convergence Curves",
            stroke_dash="Strategy",
            x_title='Time taken (in Secs)',
            y_title='Test Accuracy',
            # x_lim=[0, 50],
            tooltip=['Strategy', 'Accuracy'],
            stroke_dash_legend_title="Strategy",
            height=300,
            width=500
        )

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
            budget = gr.Slider(minimum=0, maximum=100, label='Budget (in %)')
            submit = gr.Button(value="Train Model")
        with gr.Column():
            #plot = gr.Plot(label='Convergence Curves')
            plot = gr.LinePlot(show_label=False).style(container=False)
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
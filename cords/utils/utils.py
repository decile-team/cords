import os
import glob
import pandas as pd
import numpy as np
from asyncore import read
import os
import os.path as osp
import glob
from re import L
from turtle import mode
import pandas as pd
import numpy as np
from .config_utils import check_dir_exist


"""
############################## Global Arguments ##############################
"""
COLUMN_NAMES_DICT = {
                'epoch': 'Epoch',
                'trn_loss': 'Training Loss',
                'trn_acc': 'Training Accuracy',
                'val_loss': 'Validation Loss',
                'val_acc': 'Validation Accuracy',
                'tst_loss': 'Test Loss',
                'tst_acc': 'Test Accuracy',
                'time': 'Timing'}

"""
############################## Functions ##############################
"""

def init_results_dict(print_args):
    results_dict = {}
    for print_arg in print_args:
        results_dict[print_arg] = [] 
    return results_dict


def sllogtodf(log_file, print_args, tmp_df_dict):
    with open(log_file, "r") as fp:
        read_lines = fp.readlines()
        if "Total time taken by " in read_lines[-1]:
            results_dict = init_results_dict(print_args)
            for i in range(1, len(read_lines)-11):
                for print_arg in print_args:
                    if 'takes' not in read_lines[i]:
                        if COLUMN_NAMES_DICT[print_arg] in read_lines[i]:
                            results_dict[print_arg].append(float(read_lines[i].split(COLUMN_NAMES_DICT[print_arg])[1].split(': ')[1].split(" ")[0]))
            for print_arg in print_args:
                if print_arg == 'time':
                    tmp_df_dict[COLUMN_NAMES_DICT[print_arg]] = generate_cumulative_timing(np.array(results_dict[print_arg]))
                else:
                    tmp_df_dict[COLUMN_NAMES_DICT[print_arg]] = results_dict[print_arg]
            tmp_df = pd.DataFrame(tmp_df_dict)
            return tmp_df
        else:
            return None


def generate_cumulative_timing(mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return (mod_cum_timing / 3600).tolist()
        

def sllogstodfs(results_dir, print_args=["epoch", "trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"]):
    """
    Convert Supervised Learning Logs to Dictionary of DataFrames
    """
    dirname = osp.abspath(osp.expanduser(results_dir))
    check_dir_exist(dirname)

    """
    ############################## Main Code ##############################
    """
    sub_dir = glob.glob(osp.join(dirname, 'SL'))
    column_names = ['Setting', 'Dataset', 'Strategy', 'Model', 'Fraction', 'Select every', 'Run']
    for print_arg in print_args:
        assert print_arg in COLUMN_NAMES_DICT, "Please add column name corresponding to the print argument in COLUMN_NAMES_DICT."
        column_names.append(COLUMN_NAMES_DICT[print_arg])
    """
    Current Folder Structure:
    all_logs_dir = os.path.join(results_dir, 
                                self.cfg.setting,
                                self.cfg.dataset.name,
                                subset_selection_name,
                                self.cfg.model.architecture,
                                str(self.cfg.dss_args.fraction),
                                str(self.cfg.dss_args.select_every),
                                str(self.cfg.train_args.run))
    """   
    incomplete_runs = []
    dfs_dict = {}
    full_dfs_dict = {}
    nonadaptive_dfs_dict = {'Random':{},
            'GlobalOrder_fl':{},
            'GlobalOrder_gc':{},
            'GlobalOrder_logdet':{},
            'GlobalOrder_supfl':{}}

    #DFs for Non-Adaptive Strategies
    for setting in sub_dir: #SL, SSL...
        dset_dir = glob.glob(osp.join(setting, '*')) 
        setting_value = osp.basename(setting)
        #dsets = [osp.basename(dset) for dset in dset_dir]
        for dset in dset_dir: #CIFAR10, SST2,....
            strategy_dir = glob.glob(osp.join(dset,  '*')) 
            dset_value = osp.basename(dset)
            for strategy in strategy_dir: #Random, Glister, GradMatch
                model_dir = glob.glob(osp.join(strategy,  '*')) 
                strategy_value = osp.basename(strategy)
                if strategy_value in ['Full', 'Random', 'GlobalOrder_fl', 'GlobalOrder_gc', 'GlobalOrder_logdet', 'GlobalOrder_supfl']:
                    for model in model_dir: #ResNet, BERTMLP,....
                        #Full doesn't depend on fraction
                        model_value = osp.basename(model)
                        if strategy_value == 'Full':
                            run_dir = glob.glob(osp.join(model,  '1', '1', '*'))
                            df_name = "_".join([setting_value, dset_value, model_value])  
                            if df_name not in full_dfs_dict.keys():
                                full_dfs_dict[df_name] = pd.DataFrame(columns=column_names)
                            for run in run_dir: #0, 1, 2, 3, ...
                                run_value = osp.basename(run)
                                log_dir = glob.glob(osp.join(run, '*.log'))
                                for log_file in log_dir: #.log files
                                    tmp_df_dict = {'Setting': setting_value, 'Dataset': dset_value, 'Strategy': strategy_value, 
                                                    'Model': model_value, 'Fraction': '1', 'Select every': '1', 'Run': run_value}
                                            
                                    tmp_df = sllogtodf(log_file, print_args, tmp_df_dict)
                                    if tmp_df is not None:        
                                        full_dfs_dict[df_name] = pd.concat([full_dfs_dict[df_name], tmp_df])  
                                    else:
                                        incomplete_runs.append('_'.join([setting_value, dset_value, strategy_value, model_value, '1', '1', run_value]))
                        else:
                            fraction_dir = glob.glob(osp.join(model,  '*')) 
                            for fraction in fraction_dir: #0.1, 0.2, 0.3,....
                                run_dir = glob.glob(osp.join(fraction, '1', '*')) 
                                fraction_value = osp.basename(fraction)
                                df_name = "_".join([setting_value, dset_value, model_value, fraction_value])
                                if df_name not in nonadaptive_dfs_dict[strategy_value].keys():
                                    nonadaptive_dfs_dict[strategy_value][df_name] = pd.DataFrame(columns=column_names)
                                for run in run_dir: #0, 1, 2, 3, ...
                                    run_value = osp.basename(run)
                                    log_dir = glob.glob(osp.join(run, '*.log'))
                                    for log_file in log_dir: #.log files
                                        tmp_df_dict = {'Setting': setting_value, 'Dataset': dset_value, 'Strategy': strategy_value, 
                                                    'Model': model_value, 'Fraction': fraction_value, 'Select every': '1', 'Run': run_value}
                                            
                                        tmp_df = sllogtodf(log_file, print_args, tmp_df_dict)
                                        if tmp_df is not None:        
                                            nonadaptive_dfs_dict[strategy_value][df_name] = pd.concat([nonadaptive_dfs_dict[strategy_value][df_name], tmp_df])  
                                        else:
                                            incomplete_runs.append('_'.join([setting_value, dset_value, strategy_value, model_value, fraction_value, '1', run_value]))
    
    #DFs for Adaptive Strategies
    for setting in sub_dir: #SL, SSL...
        dset_dir = glob.glob(osp.join(setting, '*')) 
        setting_value = osp.basename(setting)
        #dsets = [osp.basename(dset) for dset in dset_dir]
        for dset in dset_dir: #CIFAR10, SST2,....
            strategy_dir = glob.glob(osp.join(dset,  '*')) 
            dset_value = osp.basename(dset)
            for strategy in strategy_dir: #Random, Glister, GradMatch
                model_dir = glob.glob(osp.join(strategy,  '*')) 
                strategy_value = osp.basename(strategy)
                if strategy_value not in ['Full', 'Random', 'GlobalOrder_fl', 'GlobalOrder_gc', 'GlobalOrder_logdet', 'GlobalOrder_supfl']:
                    for model in model_dir: #ResNet, BERTMLP,....
                        fraction_dir = glob.glob(osp.join(model,  '*')) 
                        model_value = osp.basename(model)
                        for fraction in fraction_dir: #0.1, 0.2, 0.3,....
                            sel_dir = glob.glob(osp.join(fraction, '*')) 
                            fraction_value = osp.basename(fraction)
                            for sel in sel_dir: #1, 5, 10, 20, ...
                                run_dir = glob.glob(osp.join(sel, '*'))
                                sel_value = osp.basename(sel)
                                df_name = "_".join([setting_value, dset_value, model_value, fraction_value, sel_value])
                                
                                if df_name not in dfs_dict.keys():
                                    dfs_dict[df_name] = pd.DataFrame(columns=column_names)
                                    full_df_name = "_".join([setting_value, dset_value, model_value])  
                                    if full_df_name in full_dfs_dict.keys():
                                        dfs_dict[df_name] = pd.concat([dfs_dict[df_name], full_dfs_dict[full_df_name]])
                                    for nonadap_strategy in nonadaptive_dfs_dict.keys():
                                        nonadap_df_name = "_".join([setting_value, dset_value, model_value, fraction_value])  
                                        if nonadap_df_name in nonadaptive_dfs_dict[nonadap_strategy].keys():
                                            dfs_dict[df_name] = pd.concat([dfs_dict[df_name], nonadaptive_dfs_dict[nonadap_strategy][nonadap_df_name]])

                                for run in run_dir: #0, 1, 2, 3, ...
                                    run_value = osp.basename(run)
                                    log_dir = glob.glob(osp.join(run, '*.log'))
                                    for log_file in log_dir: #.log files
                                        tmp_df_dict = {'Setting': setting_value, 'Dataset': dset_value, 'Strategy': strategy_value, 
                                                    'Model': model_value, 'Fraction': fraction_value, 'Select every': sel_value, 'Run': run_value}
                                            
                                        tmp_df = sllogtodf(log_file, print_args, tmp_df_dict)
                                        if tmp_df is not None:        
                                            dfs_dict[df_name] = pd.concat([dfs_dict[df_name], tmp_df])
                                        else:
                                            incomplete_runs.append('_'.join([setting_value, dset_value, strategy_value, model_value, fraction_value, sel_value, run_value]))
                                    
    aggregated_columns = [COLUMN_NAMES_DICT[print_arg] for print_arg in print_args]
    aggregated_columns.remove("Epoch")
    aggregated_dfs_dict = {}
    for df_name in dfs_dict.keys():
        aggregated_df = dfs_dict[df_name].groupby(['Setting', 'Dataset', 'Strategy', 'Model', 'Fraction', 'Select every', 'Epoch'])
        mean = aggregated_df.mean()[aggregated_columns]
        std = aggregated_df.std()[aggregated_columns]
        std = std.fillna(0)
        merged_df = pd.merge(mean, std, on=['Setting', 'Dataset', 'Strategy', 'Model', 'Fraction', 'Select every', 'Epoch'], 
        suffixes=('_mean', '_std'), how="inner")
        merged_columns = list(merged_df.columns)
        merged_columns.append("Epoch")
        merged_df = merged_df.reset_index()
        merged_df = merged_df.groupby(['Setting', 'Dataset', 'Strategy', 'Model', 'Fraction', 'Select every']).agg({merged_column : lambda x: list(x) for merged_column in merged_columns})
        merged_df = merged_df.reset_index()
        aggregated_dfs_dict[df_name] = merged_df
    
    return dfs_dict, aggregated_dfs_dict


def sllogstoxl(results_dir, print_args=["epoch", "trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"], out_file='sl_output.xlsx'):
    """
    Convert Supervised Learning Logs to Excel File
    """
    dirname = osp.abspath(osp.expanduser(results_dir))
    check_dir_exist(dirname)

    out_file_path = osp.abspath(osp.expanduser(out_file))
    mean_file = osp.basename(out_file).split(".")[0] +  "_aggregated." + osp.basename(out_file).split(".")[1]
    mean_file_path = osp.abspath(osp.expanduser(mean_file))
        
    """
    ############################## Main Code ##############################
    """
    with pd.ExcelWriter(out_file_path, engine='xlsxwriter', mode='w') as writer: 
        sub_dir = glob.glob(osp.join(dirname, 'SL'))
        column_names = ['Setting', 'Dataset', 'Strategy', 'Model', 'Fraction', 'Select every', 'Run']
        for print_arg in print_args:
            assert print_arg in COLUMN_NAMES_DICT, "Please add column name corresponding to the print argument in COLUMN_NAMES_DICT."
            column_names.append(COLUMN_NAMES_DICT[print_arg])
        """
        Current Folder Structure:
        all_logs_dir = os.path.join(results_dir, 
                                    self.cfg.setting,
                                    self.cfg.dataset.name,
                                    subset_selection_name,
                                    self.cfg.model.architecture,
                                    str(self.cfg.dss_args.fraction),
                                    str(self.cfg.dss_args.select_every),
                                    str(self.cfg.train_args.run))
        """   
        incomplete_runs = []
        dfs_dict = {}
        full_dfs_dict = {}
        nonadaptive_dfs_dict = {'Random':{},
                'GlobalOrder_fl':{},
                'GlobalOrder_gc':{},
                'GlobalOrder_logdet':{},
                'GlobalOrder_supfl':{}}

        #DFs for Non-Adaptive Strategies
        for setting in sub_dir: #SL, SSL...
            dset_dir = glob.glob(osp.join(setting, '*')) 
            setting_value = osp.basename(setting)
            #dsets = [osp.basename(dset) for dset in dset_dir]
            for dset in dset_dir: #CIFAR10, SST2,....
                strategy_dir = glob.glob(osp.join(dset,  '*')) 
                dset_value = osp.basename(dset)
                if dset_value == 'rotten_tomatoes':
                    dset_value = 'rt'
                for strategy in strategy_dir: #Random, Glister, GradMatch
                    model_dir = glob.glob(osp.join(strategy,  '*')) 
                    strategy_value = osp.basename(strategy)
                    if strategy_value in ['Full', 'Random', 'GlobalOrder_fl', 'GlobalOrder_gc', 'GlobalOrder_logdet', 'GlobalOrder_supfl']:
                        for model in model_dir: #ResNet, BERTMLP,....
                            #Full doesn't depend on fraction
                            model_value = osp.basename(model)
                            if strategy_value == 'Full':
                                run_dir = glob.glob(osp.join(model,  '1', '1', '*'))
                                df_name = "_".join([setting_value, dset_value, model_value])  
                                if df_name not in full_dfs_dict.keys():
                                    full_dfs_dict[df_name] = pd.DataFrame(columns=column_names)
                                for run in run_dir: #0, 1, 2, 3, ...
                                    run_value = osp.basename(run)
                                    log_dir = glob.glob(osp.join(run, '*.log'))
                                    for log_file in log_dir: #.log files
                                        tmp_df_dict = {'Setting': setting_value, 'Dataset': dset_value, 'Strategy': strategy_value, 
                                                     'Model': model_value, 'Fraction': '1', 'Select every': '1', 'Run': run_value}
                                              
                                        tmp_df = sllogtodf(log_file, print_args, tmp_df_dict)
                                        if tmp_df is not None:        
                                            full_dfs_dict[df_name] = pd.concat([full_dfs_dict[df_name], tmp_df])  
                                        else:
                                            incomplete_runs.append('_'.join([setting_value, dset_value, strategy_value, model_value, '1', '1', run_value]))
                            else:
                                fraction_dir = glob.glob(osp.join(model,  '*')) 
                                for fraction in fraction_dir: #0.1, 0.2, 0.3,....
                                    run_dir = glob.glob(osp.join(fraction, '1', '*')) 
                                    fraction_value = osp.basename(fraction)
                                    df_name = "_".join([setting_value, dset_value, model_value, fraction_value])
                                    if df_name not in nonadaptive_dfs_dict[strategy_value].keys():
                                        nonadaptive_dfs_dict[strategy_value][df_name] = pd.DataFrame(columns=column_names)
                                    for run in run_dir: #0, 1, 2, 3, ...
                                        run_value = osp.basename(run)
                                        log_dir = glob.glob(osp.join(run, '*.log'))
                                        for log_file in log_dir: #.log files
                                            tmp_df_dict = {'Setting': setting_value, 'Dataset': dset_value, 'Strategy': strategy_value, 
                                                     'Model': model_value, 'Fraction': fraction_value, 'Select every': '1', 'Run': run_value}
                                              
                                            tmp_df = sllogtodf(log_file, print_args, tmp_df_dict)
                                            if tmp_df is not None:        
                                                nonadaptive_dfs_dict[strategy_value][df_name] = pd.concat([nonadaptive_dfs_dict[strategy_value][df_name], tmp_df])  
                                            else:
                                                incomplete_runs.append('_'.join([setting_value, dset_value, strategy_value, model_value, fraction_value, '1', run_value]))
        
        #DFs for Adaptive Strategies
        for setting in sub_dir: #SL, SSL...
            dset_dir = glob.glob(osp.join(setting, '*')) 
            setting_value = osp.basename(setting)
            #dsets = [osp.basename(dset) for dset in dset_dir]
            for dset in dset_dir: #CIFAR10, SST2,....
                strategy_dir = glob.glob(osp.join(dset,  '*')) 
                dset_value = osp.basename(dset)
                if dset_value == 'rotten_tomatoes':
                    dset_value = 'rt'
                for strategy in strategy_dir: #Random, Glister, GradMatch
                    model_dir = glob.glob(osp.join(strategy,  '*')) 
                    strategy_value = osp.basename(strategy)
                    if strategy_value not in ['Full', 'Random', 'GlobalOrder_fl', 'GlobalOrder_gc', 'GlobalOrder_logdet', 'GlobalOrder_supfl']:
                        for model in model_dir: #ResNet, BERTMLP,....
                            fraction_dir = glob.glob(osp.join(model,  '*')) 
                            model_value = osp.basename(model)
                            for fraction in fraction_dir: #0.1, 0.2, 0.3,....
                                sel_dir = glob.glob(osp.join(fraction, '*')) 
                                fraction_value = osp.basename(fraction)
                                for sel in sel_dir: #1, 5, 10, 20, ...
                                    run_dir = glob.glob(osp.join(sel, '*'))
                                    sel_value = osp.basename(sel)
                                    df_name = "_".join([setting_value, dset_value, model_value, fraction_value, sel_value])
                                    
                                    if df_name not in dfs_dict.keys():
                                        dfs_dict[df_name] = pd.DataFrame(columns=column_names)
                                        full_df_name = "_".join([setting_value, dset_value, model_value])  
                                        if full_df_name in full_dfs_dict.keys():
                                            dfs_dict[df_name] = pd.concat([dfs_dict[df_name], full_dfs_dict[full_df_name]])
                                        for nonadap_strategy in nonadaptive_dfs_dict.keys():
                                            nonadap_df_name = "_".join([setting_value, dset_value, model_value, fraction_value])  
                                            if nonadap_df_name in nonadaptive_dfs_dict[nonadap_strategy].keys():
                                                dfs_dict[df_name] = pd.concat([dfs_dict[df_name], nonadaptive_dfs_dict[nonadap_strategy][nonadap_df_name]])

                                    for run in run_dir: #0, 1, 2, 3, ...
                                        run_value = osp.basename(run)
                                        log_dir = glob.glob(osp.join(run, '*.log'))
                                        for log_file in log_dir: #.log files
                                            tmp_df_dict = {'Setting': setting_value, 'Dataset': dset_value, 'Strategy': strategy_value, 
                                                     'Model': model_value, 'Fraction': fraction_value, 'Select every': sel_value, 'Run': run_value}
                                              
                                            tmp_df = sllogtodf(log_file, print_args, tmp_df_dict)
                                            if tmp_df is not None:        
                                                dfs_dict[df_name] = pd.concat([dfs_dict[df_name], tmp_df])
                                            else:
                                                incomplete_runs.append('_'.join([setting_value, dset_value, strategy_value, model_value, fraction_value, sel_value, run_value]))

        for df_name in dfs_dict.keys():
            dfs_dict[df_name].to_excel(writer, sheet_name=df_name)
        #writer.save()

    with pd.ExcelWriter(mean_file_path, engine='xlsxwriter', mode='w') as writer: 
        aggregated_columns = [COLUMN_NAMES_DICT[print_arg] for print_arg in print_args]
        aggregated_columns.remove("Epoch")
        aggregated_dfs_dict = {}
        for df_name in dfs_dict.keys():
            aggregated_df = dfs_dict[df_name].groupby(['Setting', 'Dataset', 'Strategy', 'Model', 'Fraction', 'Select every', 'Epoch'])
            mean = aggregated_df.mean()[aggregated_columns]
            std = aggregated_df.std()[aggregated_columns]
            std = std.fillna(0)
            merged_df = pd.merge(mean, std, on=['Setting', 'Dataset', 'Strategy', 'Model', 'Fraction', 'Select every', 'Epoch'], 
            suffixes=('_mean', '_std'), how="inner")
            merged_columns = list(merged_df.columns)
            merged_columns.append("Epoch")
            merged_df = merged_df.reset_index()
            merged_df = merged_df.groupby(['Setting', 'Dataset', 'Strategy', 'Model', 'Fraction', 'Select every']).agg({merged_column : lambda x: list(x) for merged_column in merged_columns})
            merged_df = merged_df.reset_index()
            aggregated_dfs_dict[df_name] = merged_df
        
        for df_name in aggregated_dfs_dict.keys():
            aggregated_dfs_dict[df_name].to_excel(writer, sheet_name=df_name)
        #writer.save()
import os
import glob
import pandas as pd
import numpy as np

def generate_cumulative_timing(mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing / 3600
        

def logtoxl(results_dir, print_arguments=['val_acc', 'tst_acc', 'time'], out_file='output.xlsx'):
    dir = results_dir
    print_args = print_arguments
    sub_dir = glob.glob(dir + '/*')

    df = pd.DataFrame()
    with pd.ExcelWriter(out_file, mode='w') as writer:
        df.to_excel(writer)
        
    with pd.ExcelWriter('output.xlsx', mode='a') as writer: 
        column_names = ['Dataset', 'Select every', 'Strategy', 'Budget', 'Accuracy', 'Time']
        mnist_df = pd.DataFrame(columns=column_names) 
        fmnist_df = pd.DataFrame(columns=column_names) 
        cifar10_df = pd.DataFrame(columns=column_names) 
        cifar100_df = pd.DataFrame(columns=column_names) 
        svhn_df = pd.DataFrame(columns=column_names) 
                                        
        for folder in sub_dir: 
            dset_dir = glob.glob(folder + '/*') #craig, craigpb...
            strat_value = os.path.basename(folder)
            
            for fraction in dset_dir: 
                frac_dir = glob.glob(fraction + '/*') #mnist, fashion-mnist,....
                dset_value = os.path.basename(fraction)
                
                for select in frac_dir: 
                    sel_dir = glob.glob(select + '/*') #0.1,0.2,0.3,...
                    bud_value = os.path.basename(select)
                    bud_value = float(bud_value)*100
                    
                    for files_dir in sel_dir: 
                        f_dir = glob.glob(files_dir + '/*.txt') #10,20,...
                        select_value = os.path.basename(files_dir)
                        
                        for file in f_dir: #.txt files
                        
                            with open(file, "r") as fp:
                                read_lines = fp.readlines()
                                strategy_name = read_lines[1]
                                if 'val_acc' in print_args:
                                    val = read_lines[3].strip().split(',')[2:]
                                    val = [float(i) for i in val]
                                    val_acc = np.array(val).max()
                                    val_acc = val_acc*100
                                if 'tst_acc' in print_args:
                                    tst = read_lines[4].strip().split(',')[2:]
                                    tst = [float(i) for i in tst]
                                    tst_acc = np.array(tst).max()
                                    tst_acc = tst_acc*100
                                if 'time' in print_args:
                                    try:
                                        timing = read_lines[5:][0]
                                        tim = generate_cumulative_timing(np.array(timing))
                                        print(tim)
                                    except TypeError:
                                        timing = read_lines[5:]
                                        req_timing = []
                                        for lin in timing:
                                            qw = [i.replace('[', '').replace(']', '') for i in lin.strip().split()]
                                            req_timing.extend(qw)
                                        req_timing = req_timing[1:-1]
                                        req_timing = [i.replace(',', '') for i in req_timing]
                                        req_timing = [float(i) for i in req_timing]
                                        tim = generate_cumulative_timing(np.array(req_timing))[-1]
                                        
                                if dset_value=='mnist':
                                    mnist_df = mnist_df.append({'Dataset':dset_value, 'Select every':select_value, 'Strategy':strategy_name, 'Budget': bud_value, 'Accuracy':tst_acc, 'Time':tim}, ignore_index=True)
                                elif dset_value=='fashion-mnist':
                                    fmnist_df = fmnist_df.append({'Dataset':dset_value, 'Select every':select_value, 'Strategy':strategy_name, 'Budget': bud_value, 'Accuracy':tst_acc, 'Time':tim}, ignore_index=True)
                                elif dset_value=='cifar10':
                                    cifar10_df = cifar10_df.append({'Dataset':dset_value, 'Select every':select_value, 'Strategy':strategy_name, 'Budget': bud_value, 'Accuracy':tst_acc, 'Time':tim}, ignore_index=True)
                                elif dset_value=='cifar100':
                                    cifar100_df = cifar100_df.append({'Dataset':dset_value, 'Select every':select_value, 'Strategy':strategy_name, 'Budget': bud_value, 'Accuracy':tst_acc, 'Time':tim}, ignore_index=True)
                                elif dset_value=='svhn':
                                    svhn_df = svhn_df.append({'Dataset':dset_value, 'Select every':select_value, 'Strategy':strategy_name, 'Budget': bud_value, 'Accuracy':tst_acc, 'Time':tim}, ignore_index=True)
        
        mnist_df.to_excel(writer, sheet_name='mnist')
        fmnist_df.to_excel(writer, sheet_name='fashion-mnist')
        cifar10_df.to_excel(writer, sheet_name='cifar10')
        cifar100_df.to_excel(writer, sheet_name='cifar100')
        svhn_df.to_excel(writer, sheet_name='svhn')
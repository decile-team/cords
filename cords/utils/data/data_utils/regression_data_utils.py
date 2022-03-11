import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


## Utility function to load datasets from libsvm datasets
def csv_file_load(path,dim,skip=False,save_data=False):
    data = []
    target = []
    with open(path) as fp:
       if skip:
           line = fp.readline()
       line = fp.readline()
       while line:
        temp = [i for i in line.strip().split(",")]
        target.append(int(float(temp[-1]))) # Class Number. # Not assumed to be in (0, K-1)
        temp_data = [0]*dim
        count = 0
        for i in temp[:-1]:
            #ind, val = i.split(':')
            temp_data[count] = float(i)
            count += 1
        data.append(temp_data)
        line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)

def libsvm_file_load(path,dim, save_data=False):
    data = []
    target = []
    with open(path) as fp:
       line = fp.readline()
       while line:
        temp = [i for i in line.strip().split(" ")]
        target.append(int(float(temp[0]))) # Class Number. # Not assumed to be in (0, K-1)
        temp_data = [0]*dim
        
        for i in temp[1:]:
            if len(i) > 1: 
                ind,val = i.split(':')
                temp_data[int(ind)-1] = float(val)
        data.append(temp_data)
        line = fp.readline()
    X_data = np.array(data,dtype=np.float32)
    Y_label = np.array(target)
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)

def clean_lawschool_full(path):
   
    df = pd.read_csv(path)
    df = df.dropna()
    # remove y from df
    y = df['ugpa']
    y = y / 4
    df = df.drop('ugpa', 1)
    # convert gender variables to 0,1
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    # add bar1 back to the feature set
    df_bar = df['bar1']
    df = df.drop('bar1', 1)
    df['bar1'] = [int(grade == 'P') for grade in df_bar]
    #df['race'] = [int(race == 7.0) for race in df['race']]
    #a = df['race']
    return df.to_numpy(), y.to_numpy()

def majority_pop(a):
    """
    Identify the main ethnicity group of each community
    """
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    races = [B, W, A, H]
    maj = a.apply(pd.Series.idxmax, axis=1)
    return maj

def clean_communities_full(path):
    """
    Extract black and white dominant communities; 
    sub_size : number of communities for each group
    """
    df = pd.read_csv(path)
    df = df.fillna(0)
    B = "racepctblack"
    W = "racePctWhite"
    A = "racePctAsian"
    H = "racePctHisp"
    sens_features = [2, 3, 4, 5]
    df_sens = df.iloc[:, sens_features]

    # creating labels using crime rate
    Y = df['ViolentCrimesPerPop']
    df = df.drop('ViolentCrimesPerPop', 1)

    maj = majority_pop(df_sens)

    # remap the values of maj
    a = maj.map({B : 0, W : 1, A : 2, H : 3})
   
    df['race'] = a
    df = df.drop(H, 1)
    df = df.drop(B, 1)
    df = df.drop(W, 1)
    df = df.drop(A, 1)

    #print(df.head())

    return df.to_numpy(), Y.to_numpy()

def community_crime_load(path,dim, save_data=False):

    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i.strip() for i in line.strip().split(",")][5:]

            target.append(float(temp[-1]))
            
            temp_data = [0.0]*dim
            
            #print(temp)

            for i in range(len(temp[:-1])):

                if temp[i] != '?':
                    temp_data[i] = float(temp[i])
            
            data.append(temp_data)
            line = fp.readline()
    
    X_data = np.array(data, dtype=np.float32)
    Y_label = np.array(target)
    
    if save_data:
        # Save the numpy files to the folder where they come from
        data_np_path = path + '.data.npy'
        target_np_path = path + '.label.npy'
        np.save(data_np_path, X_data)
        np.save(target_np_path, Y_label)
    return (X_data, Y_label)

import numpy as np
import os
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms


## Custom PyTorch Dataset Class wrapper
class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None):       
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
            self.targets = target.long().to(device)
        else:
            self.data = data.float()
            self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return (sample_data, label) #.astype('float32')

class CustomDataset_WithId(Dataset):
    def __init__(self, data, target, transform=None):       
        self.transform = transform
        self.data = data #.astype('float32')
        self.targets = target
        self.X = self.data
        self.Y = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return sample_data, label,idx #.astype('float32')


## Utility function to load datasets from libsvm datasets
def csv_file_load(path,dim, save_data=False):
    data = []
    target = []
    with open(path) as fp:
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

def census_load(path,dim, save_data=False):
    
    enum=enumerate(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 
        'Never-worked'])
    workclass = dict((j,i) for i,j in enum)

    enum=enumerate(['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', 
        '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    education = dict((j,i) for i,j in enum)

    enum=enumerate(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    marital_status = dict((j,i) for i,j in enum)

    enum=enumerate(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    occupation = dict((j,i) for i,j in enum)

    enum=enumerate(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    relationship = dict((j,i) for i,j in enum)

    enum=enumerate(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    race = dict((j,i) for i,j in enum)

    sex ={'Female':0,'Male':1}

    enum=enumerate(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
     'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
     'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 
     'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 
     'Holand-Netherlands'])
    native_country = dict((j,i) for i,j in enum)

    data = []
    target = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i.strip() for i in line.strip().split(",")]

            if '?' in temp  or len(temp) == 1:
                line = fp.readline()
                continue

            if temp[-1].strip() == "<=50K" or temp[-1].strip() == "<=50K.":
                target.append(0) 
            else:
                target.append(1)
            
            temp_data = [0]*dim
            count = 0
            #print(temp)

            for i in temp[:-1]:

                if count == 1:
                    temp_data[count] =  workclass[i.strip()]
                elif count == 3:
                    temp_data[count] =  education[i.strip()]
                elif count == 5:
                    temp_data[count] =  marital_status[i.strip()]
                elif count == 6:
                    temp_data[count] =  occupation[i.strip()]
                elif count == 7:
                    temp_data[count] =  relationship[i.strip()]
                elif count == 8:
                    temp_data[count] =  race[i.strip()]
                elif count == 9:
                    temp_data[count] =  sex[i.strip()]
                elif count == 13:
                    temp_data[count] =  native_country[i.strip()]
                else:
                    temp_data[count] = float(i)
                temp_data[count] = float(temp_data[count])
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

def write_knndata(datadir, dset_name,feature):

    # Write out the trndata
    trn_filepath = os.path.join(datadir, feature+'_knn_' + dset_name + '.trn')
    val_filepath = os.path.join(datadir, feature+'_knn_' + dset_name + '.val')
    tst_filepath = os.path.join(datadir, feature+'_knn_' + dset_name + '.tst')

    if os.path.exists(trn_filepath) and os.path.exists(val_filepath) and os.path.exists(tst_filepath):
        return

    if dset_name in ['mnist', "fashion-mnist"] :
        fullset, valset, testset, num_cls = load_mnist_cifar(datadir, dset_name, feature=feature)

        x_trn, y_trn = fullset.data, fullset.targets
        x_tst, y_tst = testset.data, testset.targets
        x_trn = x_trn.view(x_trn.shape[0], -1)
        x_tst = x_tst.view(x_tst.shape[0], -1)
        # Get validation data: Its 10% of the entire (full) training data
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

    else:
        fullset, valset, testset, data_dims, num_cls = load_dataset_custom(datadir, dset_name, feature=feature, isnumpy=True)
    
        x_trn, y_trn = fullset
        x_val , y_val = valset
        x_tst, y_tst = testset
    ## Create VAL data
    #x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

    trndata = np.c_[x_trn, y_trn]
    valdata = np.c_[x_val, y_val]
    tstdata = np.c_[x_tst, y_tst]


    np.savetxt(trn_filepath, trndata, fmt='%.6f')
    np.savetxt(val_filepath, valdata, fmt='%.6f')
    np.savetxt(tst_filepath, tstdata, fmt='%.6f')

    return

def create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls):

    samples_per_class = np.zeros(num_cls)
    val_samples_per_class = np.zeros(num_cls)
    tst_samples_per_class = np.zeros(num_cls)
    for i in range(num_cls):
        samples_per_class[i] = len(np.where(y_trn == i)[0])
        val_samples_per_class[i] = len(np.where(y_val == i)[0])
        tst_samples_per_class[i] = len(np.where(y_tst == i)[0])
    min_samples = int(np.min(samples_per_class) * 0.1)
    selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
    for i in range(num_cls):
        if i == 0:
            if i in selected_classes:
                subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
            else:
                subset_idxs = np.where(y_trn == i)[0]
            x_trn_new = x_trn[subset_idxs]
            y_trn_new = y_trn[subset_idxs].reshape(-1, 1)
        else:
            if i in selected_classes:
                subset_idxs = np.random.choice(np.where(y_trn == i)[0], size=min_samples, replace=False)
            else:
                subset_idxs = np.where(y_trn == i)[0]
            x_trn_new = np.row_stack((x_trn_new, x_trn[subset_idxs]))
            y_trn_new = np.row_stack((y_trn_new, y_trn[subset_idxs].reshape(-1, 1)))
    max_samples = int(np.max(val_samples_per_class))
    for i in range(num_cls):
        y_class = np.where(y_val == i)[0]
        if i == 0:
            subset_ids = np.random.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_val_new = np.row_stack((x_val,x_val[subset_ids]))
            y_val_new = np.row_stack((y_val.reshape(-1, 1),y_val[subset_ids].reshape(-1, 1)))
        else:
            subset_ids = np.random.choice(y_class, size=max_samples- y_class.shape[0], replace=True)
            x_val_new = np.row_stack((x_val,x_val_new, x_val[subset_ids]))
            y_val_new = np.row_stack((y_val.reshape(-1, 1),y_val_new, y_val[subset_ids].reshape(-1, 1)))
    max_samples = int(np.max(tst_samples_per_class))
    for i in range(num_cls):
        y_class = np.where(y_tst == i)[0]
        if i == 0:
            subset_ids = np.random.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_tst_new = np.row_stack((x_tst,x_tst[subset_ids]))
            y_tst_new = np.row_stack((y_tst.reshape(-1, 1),y_tst[subset_ids].reshape(-1, 1)))
        else:
            subset_ids = np.random.choice(y_class, size=max_samples - y_class.shape[0], replace=True)
            x_tst_new = np.row_stack((x_tst,x_tst_new, x_tst[subset_ids]))
            y_tst_new = np.row_stack((y_tst.reshape(-1, 1),y_tst_new, y_tst[subset_ids].reshape(-1, 1)))

    return x_trn_new, y_trn_new.reshape(-1),x_val_new, y_val_new.reshape(-1), x_tst_new,y_tst_new.reshape(-1)

def create_noisy(y_trn,num_cls):
    
    noise_size = int(len(y_trn) * 0.8)
    noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
    y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)

    return y_trn
    
def load_dataset_custom (datadir, dset_name, feature, isnumpy=True):
    if dset_name == "dna":
        np.random.seed(42)
        trn_file = os.path.join(datadir, 'dna.scale.trn')
        val_file = os.path.join(datadir, 'dna.scale.val')
        tst_file = os.path.join(datadir, 'dna.scale.tst')
        data_dims = 180
        num_cls = 3
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        y_trn -= 1  # First Class should be zero
        y_val -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))
        return fullset, valset, testset, data_dims, num_cls

    elif dset_name == "adult":
        trn_file = os.path.join(datadir, 'a9a.trn')
        tst_file = os.path.join(datadir, 'a9a.tst')
        data_dims = 123
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        y_trn[y_trn < 0] = 0
        y_tst[y_tst < 0] = 0

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, data_dims, num_cls

    elif dset_name == "connect_4":
        trn_file = os.path.join(datadir, 'connect_4.trn')

        data_dims = 126
        num_cls = 3

        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        # The class labels are (-1,0,1). Make them to (0,1,2)
        y_trn[y_trn < 0] = 2

        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, data_dims,num_cls 

    elif dset_name == "letter":
        trn_file = os.path.join(datadir, 'letter.scale.trn')
        val_file = os.path.join(datadir, 'letter.scale.val')
        tst_file = os.path.join(datadir, 'letter.scale.tst')
        data_dims = 16
        num_cls = 26 
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, data_dims,num_cls 

    elif dset_name == "satimage":
        np.random.seed(42)
        trn_file = os.path.join(datadir, 'satimage.scale.trn')
        val_file = os.path.join(datadir, 'satimage.scale.val')
        tst_file = os.path.join(datadir, 'satimage.scale.tst')
        data_dims = 36
        num_cls = 6
        
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        y_trn -= 1  # First Class should be zero
        y_val -= 1
        y_tst -= 1  # First Class should be zero
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, data_dims,num_cls 

    elif dset_name == "svmguide1":
        np.random.seed(42)
        trn_file = os.path.join(datadir, 'svmguide1.trn_full')
        tst_file = os.path.join(datadir, 'svmguide1.tst')
        data_dims = 4
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, data_dims,num_cls 

    elif dset_name == "usps":
        trn_file = os.path.join(datadir, 'usps.trn_full')
        tst_file = os.path.join(datadir, 'usps.tst')
        data_dims = 256
        num_cls = 10
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        y_trn -= 1  # First Class should be zero
        y_tst -= 1  # First Class should be zero

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, data_dims,num_cls 

    elif dset_name == "ijcnn1":
        trn_file = os.path.join(datadir, 'ijcnn1.trn')
        val_file = os.path.join(datadir, 'ijcnn1.val')
        tst_file = os.path.join(datadir, 'ijcnn1.tst')
        data_dims = 22
        num_cls = 2
        x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
        x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
        x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
        
        # The class labels are (-1,1). Make them to (0,1)
        y_trn[y_trn < 0] = 0
        y_val[y_val < 0] = 0
        y_tst[y_tst < 0] = 0    

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn, num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, data_dims,num_cls 

    elif dset_name == "sklearn-digits":
        
        np.random.seed(42)
        data, target = datasets.load_digits(return_X_y=True)
        # Test data is 10%
        x_trn, x_tst, y_trn, y_tst = train_test_split(data, target, test_size=0.1, random_state=42)

        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        num_cls = 10
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn))
            valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst))

        return fullset, valset, testset, x_trn.shape[1],num_cls 

    elif dset_name in ['prior_shift_large_linsep_4','conv_shift_large_linsep_4','red_large_linsep_4','expand_large_linsep_4',
    'shrink_large_linsep_4','red_conv_shift_large_linsep_4',"linsep_4","large_linsep_4"]:
        
        trn_file = os.path.join(datadir, dset_name+'.trn')
        val_file = os.path.join(datadir, dset_name+'.val')
        tst_file = os.path.join(datadir, dset_name+'.tst')
        data_dims = 2
        num_cls = 4
        x_trn, y_trn = csv_file_load(trn_file, dim=data_dims)
        x_val, y_val = csv_file_load(val_file, dim=data_dims)
        x_tst, y_tst = csv_file_load(tst_file, dim=data_dims)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, data_dims, num_cls

    elif dset_name in ['prior_shift_clf_2', 'prior_shift_gauss_2','conv_shift_clf_2', 'conv_shift_gauss_2',"gauss_2", "clf_2","linsep"]:
        
        trn_file = os.path.join(datadir, dset_name+'.trn')
        val_file = os.path.join(datadir, dset_name+'.val')
        tst_file = os.path.join(datadir, dset_name+'.tst')
        data_dims = 2
        num_cls = 2
        x_trn, y_trn = csv_file_load(trn_file, dim=data_dims)
        x_val, y_val = csv_file_load(val_file, dim=data_dims)
        x_tst, y_tst = csv_file_load(tst_file, dim=data_dims)
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, data_dims,num_cls 

    elif dset_name == "covertype":
        trn_file = os.path.join(datadir, 'covtype.data')

        data_dims = 54
        num_cls = 7
        x_trn, y_trn = csv_file_load(trn_file, dim=data_dims)
       
        y_trn -= 1  # First Class should be zero
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.2, random_state=42)
        
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, data_dims,num_cls

    elif dset_name == "census":
        trn_file = os.path.join(datadir, 'adult.data')
        tst_file = os.path.join(datadir, 'adult.test')

        data_dims = 14
        num_cls = 2

        x_trn, y_trn = census_load(trn_file, dim=data_dims)
        x_tst, y_tst = census_load(tst_file, dim=data_dims)
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=42)
        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        if feature == 'classimb':
            x_trn, y_trn,x_val, y_val,x_tst, y_tst = create_imbalance(x_trn, y_trn,x_val, y_val,x_tst, y_tst,num_cls)

        elif feature == 'noise':
            y_trn = create_noisy(y_trn,num_cls)

        if isnumpy:
            fullset = (x_trn, y_trn)
            valset = (x_val, y_val)
            testset = (x_tst, y_tst)

        else:
            fullset = CustomDataset(x_trn, y_trn)
            valset = CustomDataset(x_val, y_val)
            testset = CustomDataset(x_tst, y_tst)

        return fullset, valset, testset, data_dims,num_cls


def load_mnist_cifar (datadir, dset_name,feature):
    
    if dset_name == "mnist":
        mnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist_val_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        num_cls = 10
        fullset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=mnist_transform)
        
        if feature=='classimb':
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples, replace=False))
                    else:
                        subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples, replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)
        
        valset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=mnist_val_transform)
        testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=mnist_transform)
        
        return fullset, valset, testset, num_cls

    elif dset_name == "fashion-mnist":

        mnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist_val_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        num_cls = 10
        fullset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=mnist_transform)
        
        if feature=='classimb':
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(fullset.targets == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples, replace=False))
                    else:
                        subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(np.random.choice(torch.where(fullset.targets == i)[0].cpu().numpy(), size=min_samples, replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(fullset.targets == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)
        
        valset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=mnist_val_transform)
        testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=mnist_transform)
        
        return fullset, valset, testset, num_cls

    
    elif dset_name == "cifar10":
        cifar_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        cifar_tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        num_cls = 10
        fullset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=cifar_transform)
        valset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=cifar_tst_transform)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=cifar_tst_transform)
        if feature == 'classimb':
            samples_per_class = torch.zeros(num_cls)
            for i in range(num_cls):
                samples_per_class[i] = len(torch.where(torch.Tensor(fullset.targets) == i)[0])
            min_samples = int(torch.min(samples_per_class) * 0.1)
            selected_classes = np.random.choice(np.arange(num_cls), size=int(0.3 * num_cls), replace=False)
            for i in range(num_cls):
                if i == 0:
                    if i in selected_classes:
                        subset_idxs = list(
                            np.random.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                else:
                    if i in selected_classes:
                        batch_subset_idxs = list(
                            np.random.choice(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy(), size=min_samples,
                                             replace=False))
                    else:
                        batch_subset_idxs = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
                    subset_idxs.extend(batch_subset_idxs)
            fullset = torch.utils.data.Subset(fullset, subset_idxs)
        return fullset, valset, testset, num_cls
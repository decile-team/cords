import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import preprocessing

# todo : update in better format

'''def gen_rand_prior_indices(curr_size,num_cls,x_trn,y_trn,remainList=None):

    per_sample_budget = int(curr_size/num_cls)
    if remainList is None:
        per_sample_count = [len(torch.where(y_trn == x)[0]) for x in np.arange(num_cls)]
        total_set = list(np.arange(N))
    else:
        per_sample_count = [len(torch.where(y_trn[remainList] == x)[0]) for x in np.arange(num_cls)]
        total_set = remainList
    indices = []
    count = 0
    for i in range(num_cls):
        if remainList is None:
            label_idxs = torch.where(y_trn == i)[0].cpu().numpy()
        else:
            label_idxs = torch.where(y_trn[remainList] == i)[0].cpu().numpy()
            label_idxs = np.array(remainList)[label_idxs]

        if per_sample_count[i] > per_sample_budget:
            indices.extend(list(np.random.choice(label_idxs, size=per_sample_budget, replace=False)))
        else:
            indices.extend(label_idxs)
            count += (per_sample_budget - per_sample_count[i])
    for i in indices:
        total_set.remove(i)
    indices.extend(list(np.random.choice(total_set, size= count, replace=False)))
    return indices'''

def get_slices(data_name, data,labels,device,buckets=None,clean=True):

    #data_slices = []
    #abel_slices =[]
    
    val_data_slices = []
    val_label_slices =[]

    tst_data_slices = []
    tst_label_slices =[]

    '''sc = StandardScaler()
    x_trn = sc.fit_transform(x_trn)
    x_val = sc.transform(x_val)
    x_tst = sc.transform(x_tst)'''

    if data_name == 'Community_Crime_old':
        protect_feature =[2,3,4,5]

        data_class =[]

        N = int(0.1*len(data)/(buckets*len(protect_feature)))

        total_set = set(list(np.arange(len(data))))
        
        for i in protect_feature:            

            digit = np.ones(data.shape[0],dtype=np.int8)*(i-1)
            low = np.min(data[:,i])
            high = np.max(data[:,i])
            bins = np.linspace(low, high, buckets)
            digitized = np.digitize(data[:,i], bins)
            digit =  digit*10 + digitized

            classes,times = np.unique(digit,return_counts=True) 
            times, classes = zip(*sorted(zip(times, classes)))
            data_class.append(classes)
            
            count = 0
            for cl in classes[:-1]:

                indices=[]
                indices_tst=[]
                
                idx = (digit == cl).nonzero()[0].flatten()
                idx.tolist()
                idxs = set(idx)
                idxs.intersection_update(total_set)
                idx = list(idxs)
                #print(cl,len(idx))

                curr_N = int(len(idx)/3)

                #print(curr_N,N)
                 
                indices.extend(list(np.random.choice(idx, size=min(N,curr_N), replace=False)))
                total_set.difference(indices)
                idxs.difference(indices)
                idx = list(idxs)
                indices_tst.extend(list(np.random.choice(idx, size=min(N,curr_N), replace=False)))
                total_set.difference(indices_tst)
                
                if curr_N < N:
                    count += (N - curr_N)

                val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
                val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

                tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
                tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))

            indices=[]
            indices_tst=[]
            
            idx = (digit == classes[-1]).nonzero()[0].flatten()
            idx.tolist()
            idxs = set(idx)
            idxs.intersection_update(total_set)
            idx = list(idxs)

            indices.extend(list(np.random.choice(idx, size=N+count, replace=False)))
            total_set.difference(indices)
            idxs.difference(indices)
            idx = list(idxs)
            indices_tst.extend(list(np.random.choice(idx, size=N+count, replace=False)))
            total_set.difference(indices_tst)

            val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
            val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

            tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
            tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))

        final_lables = [j for sub in data_class for j in sub]
        left = list(total_set)    
        data_left = data[left]
        label_left = labels[left]

    elif data_name == 'OnlineNewsPopularity':

        protect_feature = [11,12,13,14,15,16]

        final_lables = [ 'Lifestyle','Entertainment','Business','Social Media','Tech','World']

        total_set = set(list(np.arange(len(data))))

        N = int(0.1*len(data)/len(protect_feature))

        max_times = 0
        
        for pf in protect_feature:            
        
            classes,times = np.unique(data[:,pf],return_counts=True) 

            one_id = (classes == 1.0).nonzero()[0].flatten()[0]

            if max_times < times[one_id]:
                max_times = times[one_id]
                max_id = pf

        most = final_lables[max_id-protect_feature[0]]
        final_lables.remove(most)
        final_lables.append(most)

        count = 0
        
        for pf in protect_feature:

            if pf == max_id:
                continue
            
            idx = (data[:,pf] == 1.0).nonzero()[0].flatten()
            idx.tolist()
            idxs = set(idx)
            idxs.intersection_update(total_set)
            idx = list(idxs)
            #print(cl,len(idx))

            curr_N = int(len(idx)/3)

            #print(curr_N,N)
            
            indices = list(np.random.choice(idx, size=min(N,curr_N), replace=False))
            total_set.difference(indices)
            idxs.difference(indices)
            idx = list(idxs)
            indices_tst = list(np.random.choice(idx, size=min(N,curr_N), replace=False))
            total_set.difference(indices_tst)
            
            if curr_N < N:
                count += (N - curr_N)

            '''val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
            val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

            tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
            tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))'''

            val_data_slices.append(data[indices])
            val_label_slices.append(labels[indices])

            tst_data_slices.append(data[indices_tst])
            tst_label_slices.append(labels[indices_tst])

            
        idx = (data[:,max_id] == 1.0).nonzero()[0].flatten()
        idx.tolist()
        idxs = set(idx)
        idxs.intersection_update(total_set)
        idx = list(idxs)

        indices = list(np.random.choice(idx, size=N+count, replace=False))
        total_set.difference(indices)
        idxs.difference(indices)
        idx = list(idxs)
        indices_tst = list(np.random.choice(idx, size=N+count, replace=False)) 
        total_set.difference(indices_tst)

        '''val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
        val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

        tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
        tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))'''

        val_data_slices.append(data[indices])
        val_label_slices.append(labels[indices])

        tst_data_slices.append(data[indices_tst])
        tst_label_slices.append(labels[indices_tst])

        left = list(total_set)
        sc = MinMaxScaler() #StandardScaler()
        sc_l = MinMaxScaler()

        #print(data[left][0])
        data_left = sc.fit_transform(data[left])
        label_left = np.reshape(sc_l.fit_transform(np.reshape(labels[left],(-1,1))),(-1))
        #print(data_left[0])
        #preprocessing.normalize(data[left])

        for j in range(len(val_data_slices)):
            
            val_data_slices[j] = torch.from_numpy(sc.transform(val_data_slices[j])).float().to(device)
            tst_data_slices[j] = torch.from_numpy(sc.transform(tst_data_slices[j])).float().to(device)

            val_label_slices[j] = torch.from_numpy(np.reshape(\
                sc_l.transform(np.reshape(val_label_slices[j],(-1,1))),(-1))).float().to(device)
            tst_label_slices[j] = torch.from_numpy(np.reshape(\
                sc_l.transform(np.reshape(tst_label_slices[j],(-1,1))),(-1))).float().to(device)
    
    elif data_name in ['census','LawSchool','German_credit','Community_Crime']:
        
        if data_name == 'census':
            protect_feature = 8 #9
        elif data_name == 'LawSchool':
            protect_feature = 0
        elif data_name == 'German_credit':
            protect_feature = 8
        elif data_name == 'Community_Crime':
            protect_feature = -1

        total_set = set(list(np.arange(len(data))))
        
        classes,times = np.unique(data[:,protect_feature],return_counts=True) 
        times, classes = zip(*sorted(zip(times, classes)))

        #print(times)
        #print(classes)

        N = int(0.1*len(data)/len(classes))
        
        count = 0
        for cl in classes[:-1]:

            indices=[]
            indices_tst=[]
            
            idx = (data[:,protect_feature] == cl).nonzero()[0].flatten()
            idx.tolist()

            curr_N = int(len(idx)/3)

            #print(curr_N,N)
                
            indices.extend(list(np.random.choice(idx, size=min(N,curr_N), replace=False)))
            total_set.difference(indices)
            idxs = set(idx)
            idxs.difference(indices)
            idx = list(idxs)
            indices_tst.extend(list(np.random.choice(idx, size=min(N,curr_N), replace=False)))
            total_set.difference(indices_tst)
            
            if curr_N < N:
                count += (N - curr_N)

            #val_data_slices.append(torch.from_numpy(preprocessing.normalize(data[indices])).float().to(device))
            val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
            val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

            #tst_data_slices.append(torch.from_numpy(preprocessing.normalize(data[indices_tst])).float().to(device))
            tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
            tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))

        indices=[]
        indices_tst=[]
        
        idx = (data[:,protect_feature] == classes[-1]).nonzero()[0].flatten()
        idx.tolist()

        indices.extend(list(np.random.choice(idx, size=N+count, replace=False)))
        total_set.difference(indices)
        idxs = set(idx)
        idxs.difference(indices)
        idx = list(idxs)
        indices_tst.extend(list(np.random.choice(idx, size=N+count, replace=False)))
        total_set.difference(indices_tst)

        #val_data_slices.append(torch.from_numpy(preprocessing.normalize(data[indices])).float().to(device))
        val_data_slices.append(torch.from_numpy(data[indices]).float().to(device))
        val_label_slices.append(torch.from_numpy(labels[indices]).float().to(device))

        #tst_data_slices.append(torch.from_numpy(preprocessing.normalize(data[indices_tst])).float().to(device))
        tst_data_slices.append(torch.from_numpy(data[indices_tst]).float().to(device))
        tst_label_slices.append(torch.from_numpy(labels[indices_tst]).float().to(device))

        final_lables = classes
        left = list(total_set)
        data_left = data[left] #preprocessing.normalize(data[left]) 
        label_left = labels[left]

        if not clean:

            noise_size = int(len(label_left) * 0.5)
            noise_indices = np.random.choice(np.arange(len(label_left)), size=noise_size, replace=False)
            
            sigma = 40
            label_left[noise_indices] = label_left[noise_indices] + np.random.normal(0, sigma, noise_size)
    
        
    return data_left, label_left, val_data_slices, val_label_slices, final_lables, tst_data_slices,\
        tst_label_slices,final_lables

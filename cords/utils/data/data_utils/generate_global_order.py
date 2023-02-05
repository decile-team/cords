import datasets
import torchvision
from sentence_transformers import SentenceTransformer, util
from matplotlib import pyplot as plt 
from ..datasets.__utils import TinyImageNet
from transformers import ViTFeatureExtractor, ViTModel
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from numba import jit, config
from torch.utils.data import random_split, BatchSampler, SequentialSampler
import torch
import pickle
import math
import os
import submodlib
import argparse
import h5py
import numpy as np
import time


LABEL_MAPPINGS = {'glue_sst2':'label', 
                  'trec6':'coarse_label', 
                  'imdb':'label',
                  'rotten_tomatoes': 'label',
                  'tweet_eval': 'label'}

SENTENCE_MAPPINGS = {'glue_sst2': 'sentence', 
                    'trec6':'text',  
                    'imdb':'text',
                    'rotten_tomatoes': 'text',
                    'tweet_eval': 'text'}

IMAGE_MAPPINGS = {'cifar10': 'images'}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Global ordering of a dataset using pretrained LMs.")
    parser.add_argument(
                        "--dataset",
                        type=str,
                        default="glue_sst2",
                        help="Only supports datasets for hugging face currently."
                        )
    parser.add_argument(
                        "--model",
                        type=str,
                        default="all-distilroberta-v1",
                        help="Transformer model used for computing the embeddings."
                        )
    
    parser.add_argument(
                        "--data_dir",
                        type=str,
                        required=False,
                        help="Directory in which data downloads happen.",
                        default="/home/krishnateja/data"
                        ) 
    parser.add_argument(
                        "--submod_function",
                        type=str,
                        default="logdet",
                        help="Submdular function used for finding the global order."
                        )
    parser.add_argument(
                        "--seed",
                        type=int,
                        default=42,
                        help="Seed value for reproducibility of the experiments."
                        )
    parser.add_argument(
                        "--device",
                        type=str,
                        default='cuda:1',
                        help= "Device used for computing the embeddings"
                        )
    args=parser.parse_args()
    return args


def dict2pickle(file_name, dict_object):
    """
    Store dictionary to pickle file
    """    
    with open(file_name, "wb") as fOut:
        pickle.dump(dict_object, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def pickle2dict(file_name, key):
    """
    Load dictionary from pickle file
    """
    with open(file_name, "rb") as fIn:
        stored_data = pickle.load(fIn)
        value = stored_data[key]
    return value


def store_embeddings(pickle_name, embeddings):
    """
    Store embeddings to disc
    """
    with open(pickle_name, "wb") as fOut:
        pickle.dump({'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def load_embeddings(pickle_name):
    """
    Load embeddings from disc
    """
    with open(pickle_name, "rb") as fIn:
        stored_data = pickle.load(fIn)
        #stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']
    return stored_embeddings    


def get_cdist(V):
	ct = time.time()
	dist_mat = euclidean_distances(V)
	print("Distance Matrix construction time ", time.time()-ct)
	return get_square(dist_mat)


#@torch.jit.script
@jit(nopython=True, parallel=True)
def get_square(mat):
	return mat**2


@jit(nopython=True, parallel=True)
def get_rbf_kernel(dist_mat, kw=0.1):
	sim = np.exp(-dist_mat/(kw*dist_mat.mean()))
	return sim


# @jit(nopython=True, parallel=True)
def get_dot_product(mat):
	sim = np.matmul(mat, np.transpose(mat))
	return sim


def compute_text_embeddings(model_name, sentences, device, return_tensor=False):
    """
    Compute sentence embeddings using a transformer model and return in numpy or tensor format
    """
    model = SentenceTransformer(model_name, device=device)
    if return_tensor:
        embeddings = model.encode(sentences, device=device, convert_to_tensor=True).cpu()
    else:
        embeddings = model.encode(sentences, device=device, convert_to_numpy=True)
    return embeddings


def compute_image_embeddings(model_name, images, device, return_tensor=False):
    """
    Compute image embeddings using CLIP based model and return in numpy or tensor format
    """
    model = SentenceTransformer(model_name, device=device)
    if return_tensor:
        embeddings = model.encode(images, device=device, convert_to_tensor=True).cpu()
    else:
        embeddings = model.encode(images, device=device, convert_to_numpy=True)
    return embeddings


def compute_vit_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
    model= model.to(device)
    #inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))),
                           20, 
                           drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == 'L':
            images_batch = [images[x].convert('RGB') for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state.mean(dim=1).cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict
    
    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features
    

def compute_vit_cls_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
    model= model.to(device)
    #inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))),
                           20, 
                           drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == 'L':
            images_batch = [images[x].convert('RGB') for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state[:, 0, :].cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict
    
    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_dino_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    model = ViTModel.from_pretrained('facebook/dino-vitb16')
    model= model.to(device)
    #inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))),
                           20, 
                           drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == 'L':
            images_batch = [images[x].convert('RGB') for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state.mean(dim=1).cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict
    
    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features
    

def compute_dino_cls_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
    model = ViTModel.from_pretrained('facebook/dino-vitb16')
    model= model.to(device)
    #inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))),
                           20, 
                           drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == 'L':
            images_batch = [images[x].convert('RGB') for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state[:, 0, :].cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict
    
    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_global_ordering(embeddings, submod_function, train_labels, kw, r2_coefficient, knn, metric):
    """
    Return greedy ordering and gains with different submodular functions as the global order.
    """
    if submod_function not in ["supfl", "gc_pc", "logdet_pc", "disp_min_pc", "disp_sum_pc"]:
        data_dist = get_cdist(embeddings)
        if metric == 'rbf_kernel':
            data_sijs = get_rbf_kernel(data_dist, kw)
        elif metric == 'dot':
            data_sijs = get_dot_product(embeddings)
            if submod_function in ["disp_min", "disp_sum"]:
                data_sijs = (data_sijs - np.min(data_sijs))/(np.max(data_sijs) - np.min(data_sijs))
            else:
                if np.min(data_sijs) < 0:
                    data_sijs = data_sijs - np.min(data_sijs)
        elif metric == 'cossim':
            normalized_embeddings = embeddings/np.linalg.norm(embeddings, axis=1, keepdims=True)
            data_sijs = get_dot_product(normalized_embeddings)
            if submod_function in ["disp_min", "disp_sum"]:
                data_sijs = (data_sijs - np.min(data_sijs))/(np.max(data_sijs) - np.min(data_sijs))
            else:
                data_sijs = (data_sijs + 1)/2
        else:
            raise ValueError('Please enter a valid metric')

        data_knn = np.argsort(data_dist, axis=1)[:, :knn].tolist()
        data_r2 = np.nonzero(data_dist <= max(1e-5, data_dist.mean() - r2_coefficient*data_dist.std()))
        data_r2 = zip(data_r2[0].tolist(), data_r2[1].tolist())
        data_r2_dict = {}
        for x in data_r2:
            if x[0] in data_r2_dict.keys():
                data_r2_dict[x[0]].append(x[1])
            else:
                data_r2_dict[x[0]] = [x[1]]


    if submod_function == 'fl':
        obj = submodlib.FacilityLocationFunction(n = embeddings.shape[0],
                                                separate_rep=False,
                                                mode = 'dense',
                                                sijs = data_sijs)

    elif submod_function == 'logdet':
        obj = submodlib.LogDeterminantFunction(n = embeddings.shape[0],
                                                mode = 'dense',
                                                lambdaVal = 1,
                                                sijs = data_sijs)
    
    elif submod_function == 'gc':
        obj = submodlib.GraphCutFunction(n = embeddings.shape[0],
                                        mode = 'dense',
                                        lambdaVal = 1,
                                        separate_rep=False,
                                        ggsijs = data_sijs)

    elif submod_function == 'disp_min':
        obj = submodlib.DisparityMinFunction(n = embeddings.shape[0], 
                                            mode = 'dense', 
                                            sijs= data_sijs)

    elif submod_function == 'disp_sum':
        obj = submodlib.DisparitySumFunction(n = embeddings.shape[0], 
                                            mode = 'dense', 
                                            sijs= data_sijs)
    
    if submod_function in ['gc', 'fl', 'logdet', 'disp_min', 'disp_sum']:
        if submod_function == 'disp_min':
            greedyList = obj.maximize(budget=embeddings.shape[0]-1, optimizer='NaiveGreedy', stopIfZeroGain=False,
                            stopIfNegativeGain=False, verbose=False)
        else:
            greedyList = obj.maximize(budget=embeddings.shape[0]-1, optimizer='LazyGreedy', stopIfZeroGain=False,
                            stopIfNegativeGain=False, verbose=False)
        rem_elem = list(set(range(embeddings.shape[0])).difference(set([x[0] for x in greedyList])))[0]
        rem_gain = greedyList[-1][1]
        greedyList.append((rem_elem, rem_gain))
    else:
        clusters = set(train_labels)
        data_knn  = [[] for _ in range(len(train_labels))]
        data_r2_dict = {}
        greedyList = []
        #Label-wise Partition
        cluster_idxs = {}
        for i in range(len(train_labels)):
            if train_labels[i] in cluster_idxs.keys():
                cluster_idxs[train_labels[i]].append(i)
            else:
                cluster_idxs[train_labels[i]] = [i]
        for cluster in clusters:
            idxs = cluster_idxs[cluster]
            cluster_embeddings = embeddings[idxs, :]

            print(cluster_embeddings.shape)
            clustered_dist = get_cdist(cluster_embeddings)
            if metric == 'rbf_kernel':
                clustered_sijs = get_rbf_kernel(clustered_dist, kw)
            elif metric == 'dot':
                clustered_sijs = get_dot_product(cluster_embeddings)
                if submod_function in ["disp_min_pc", "disp_sum_pc"]:
                    clustered_sijs = (clustered_sijs - np.min(clustered_sijs))/(np.max(clustered_sijs) - np.min(clustered_sijs))
                else:
                    if np.min(clustered_sijs) < 0:
                        clustered_sijs = clustered_sijs - np.min(clustered_sijs)
            elif metric == 'cossim':
                normalized_embeddings = cluster_embeddings/np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
                clustered_sijs = get_dot_product(normalized_embeddings)
                if submod_function in ["disp_min_pc", "disp_sum_pc"]:
                    clustered_sijs = (clustered_sijs - np.min(clustered_sijs))/(np.max(clustered_sijs) - np.min(clustered_sijs))
                else:
                    clustered_sijs = (1 + clustered_sijs)/2
            else:
                raise ValueError('Please enter a valid metric')
            
            if submod_function in ['supfl']:
                obj = submodlib.FacilityLocationFunction(n = cluster_embeddings.shape[0],
                                                separate_rep=False,
                                                mode = 'dense',
                                                sijs = clustered_sijs)
            elif submod_function in ['gc_pc']:
                obj = submodlib.GraphCutFunction(n = cluster_embeddings.shape[0],
                                                mode = 'dense',
                                                lambdaVal = 0.4,
                                                separate_rep=False,
                                                ggsijs = clustered_sijs)
            elif submod_function in ['logdet_pc']:
                obj = submodlib.LogDeterminantFunction(n = cluster_embeddings.shape[0],
                                                mode = 'dense',
                                                lambdaVal = 1,
                                                sijs = clustered_sijs)
            
            elif submod_function == 'disp_min_pc':
                obj = submodlib.DisparityMinFunction(n = cluster_embeddings.shape[0], 
                                                    mode = 'dense', 
                                                    sijs= clustered_sijs)

            elif submod_function == 'disp_sum_pc':
                obj = submodlib.DisparitySumFunction(n = cluster_embeddings.shape[0], 
                                                    mode = 'dense', 
                                                    sijs= clustered_sijs)

            if submod_function == 'disp_min_pc':
                clustergreedyList = obj.maximize(budget=cluster_embeddings.shape[0]-1, optimizer='NaiveGreedy', stopIfZeroGain=False,
                            stopIfNegativeGain=False, verbose=False)
            else:
                clustergreedyList = obj.maximize(budget=cluster_embeddings.shape[0]-1, optimizer='LazyGreedy', stopIfZeroGain=False,
                            stopIfNegativeGain=False, verbose=False)
            rem_elem = list(set(range(cluster_embeddings.shape[0])).difference(set([x[0] for x in clustergreedyList])))[0]
            rem_gain = clustergreedyList[-1][1]
            clustergreedyList.append((rem_elem, rem_gain))
            clusteredgreedylist_with_orig_idxs = [(idxs[x[0]], x[1]) for x in clustergreedyList]
            greedyList.extend(clusteredgreedylist_with_orig_idxs)
            del obj
            clustered_knn = np.argsort(clustered_dist, axis=1)[:, :knn].tolist()
            for i in range(len(idxs)):
                data_knn[idxs[i]] = [idxs[j] for j in clustered_knn[i]]
            clustered_r2 = np.nonzero(clustered_dist <= max(1e-5, clustered_dist.mean() - r2_coefficient*clustered_dist.std()))
            clustered_r2 = zip(clustered_r2[0].tolist(), clustered_r2[1].tolist())
            for x in clustered_r2:
                if idxs[x[0]] in data_r2_dict.keys():
                    data_r2_dict[idxs[x[0]]].append(idxs[x[1]])
                else:
                    data_r2_dict[idxs[x[0]]] = [idxs[x[1]]]
        greedyList.sort(key=lambda x: x[1], reverse=True)

    knn_list = []
    r2_list = []
    for x in greedyList:
        knn_list.append(data_knn[x[0]])
        r2_list.append(data_r2_dict[x[0]])
    #Sorted Label-wise Partition
    cluster_idxs = {}
    greedy_idxs = [x[0] for x in greedyList]
    for i in greedy_idxs:
        if train_labels[i] in cluster_idxs.keys():
            cluster_idxs[train_labels[i]].append(i)
        else:
            cluster_idxs[train_labels[i]] = [i]
    return greedyList, knn_list, r2_list, cluster_idxs


def compute_stochastic_greedy_subsets(embeddings, submod_function, train_labels, kw, metric, fraction, n_subsets=300):
    """
    Return greedy ordering and gains with different submodular functions as the global order.
    """
    budget = int(fraction * embeddings.shape[0])
    if submod_function not in ["supfl", "gc_pc", "logdet_pc", "disp_min", "disp_sum"]:
        data_dist = get_cdist(embeddings)
        if metric == 'rbf_kernel':
            data_sijs = get_rbf_kernel(data_dist, kw)
        elif metric == 'dot':
            data_sijs = get_dot_product(embeddings)
            if submod_function in ["disp_min", "disp_sum"]:
                data_sijs = (data_sijs - np.min(data_sijs))/(np.max(data_sijs) - np.min(data_sijs))
            else:
                if np.min(data_sijs) < 0:
                    data_sijs = data_sijs - np.min(data_sijs)
        elif metric == 'cossim':
            normalized_embeddings = embeddings/np.linalg.norm(embeddings, axis=1, keepdims=True)
            data_sijs = get_dot_product(normalized_embeddings)
            if submod_function in ["disp_min", "disp_sum"]:
                data_sijs = (data_sijs - np.min(data_sijs))/(np.max(data_sijs) - np.min(data_sijs))
            else:
                data_sijs = (data_sijs + 1)/2
        else:
            raise ValueError('Please enter a valid metric')

    if submod_function == 'fl':
        obj = submodlib.FacilityLocationFunction(n = embeddings.shape[0],
                                                separate_rep=False,
                                                mode = 'dense',
                                                sijs = data_sijs)

    elif submod_function == 'logdet':
        obj = submodlib.LogDeterminantFunction(n = embeddings.shape[0],
                                                mode = 'dense',
                                                lambdaVal = 1,
                                                sijs = data_sijs)
    
    elif submod_function == 'gc':
        obj = submodlib.GraphCutFunction(n = embeddings.shape[0],
                                        mode = 'dense',
                                        lambdaVal = 1,
                                        separate_rep=False,
                                        ggsijs = data_sijs)
    
    elif submod_function == 'disp_min':
        obj = submodlib.DisparityMinFunction(n = embeddings.shape[0], 
                                            mode = 'dense', 
                                            sijs= data_sijs)

    elif submod_function == 'disp_sum':
        obj = submodlib.DisparitySumFunction(n = embeddings.shape[0], 
                                            mode = 'dense', 
                                            sijs= data_sijs)
    
    subsets = []
    total_time = 0
    if submod_function not in ['supfl', 'gc_pc', 'logdet_pc', "disp_min_pc", "disp_sum_pc"]:
        for _ in range(n_subsets):
            st_time = time.time()
            if submod_function == 'disp_min':
                subset = obj.maximize(budget=budget, optimizer='StochasticGreedy', epsilon=0.001, stopIfZeroGain=False,
                            stopIfNegativeGain=False, verbose=False)
            else:
                subset = obj.maximize(budget=budget, optimizer='LazierThanLazyGreedy', epsilon=0.001, stopIfZeroGain=False,
                            stopIfNegativeGain=False, verbose=False)
            subsets.append(subset)
            total_time += (time.time() - st_time)
    else:
        clusters = set(train_labels)
        #Label-wise Partition
        cluster_idxs = {}
        #print(train_labels)
        for i in range(len(train_labels)):
            if train_labels[i] in cluster_idxs.keys():
                cluster_idxs[train_labels[i]].append(i)
            else:
                cluster_idxs[train_labels[i]] = [i]

        per_cls_cnt = [len(cluster_idxs[key]) for key in cluster_idxs.keys()]
        min_cls_cnt = min(per_cls_cnt)
        if min_cls_cnt < math.ceil(budget/len(clusters)):
            per_cls_budget = [min_cls_cnt]*len(clusters)
            while sum(per_cls_budget) < budget:
                for cls in range(len(clusters)):
                    if per_cls_budget[cls] < per_cls_cnt[cls]:
                        per_cls_budget[cls] += 1
        else:
            per_cls_budget = [math.ceil(budget/len(clusters)) for _ in per_cls_cnt]
        
        for _ in range(n_subsets):
            st_time = time.time()
            subset = []
            cluster_idx = 0
            for cluster in cluster_idxs.keys():
                idxs = cluster_idxs[cluster]
                cluster_embeddings = embeddings[idxs, :]
                clustered_dist = get_cdist(cluster_embeddings)
                if metric == 'rbf_kernel':
                    clustered_sijs = get_rbf_kernel(clustered_dist, kw)
                elif metric == 'dot':
                    clustered_sijs = get_dot_product(cluster_embeddings)
                    if submod_function in ["disp_min_pc", "disp_sum_pc"]:
                        clustered_sijs = (clustered_sijs - np.min(clustered_sijs))/(np.max(clustered_sijs) - np.min(clustered_sijs))
                    else:
                        if np.min(clustered_sijs) < 0:
                            clustered_sijs = clustered_sijs - np.min(clustered_sijs)
                elif metric == 'cossim':
                    normalized_embeddings = cluster_embeddings/np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
                    clustered_sijs = get_dot_product(normalized_embeddings)
                    if submod_function in ["disp_min_pc", "disp_sum_pc"]:
                        clustered_sijs = (clustered_sijs - np.min(clustered_sijs))/(np.max(clustered_sijs) - np.min(clustered_sijs))
                    else:
                        clustered_sijs = (1 + clustered_sijs)/2
                else:
                    raise ValueError('Please enter a valid metric')
                
                if submod_function in ['supfl']:
                    obj = submodlib.FacilityLocationFunction(n = cluster_embeddings.shape[0],
                                                    separate_rep=False,
                                                    mode = 'dense',
                                                    sijs = clustered_sijs)
                elif submod_function in ['gc_pc']:
                    obj = submodlib.GraphCutFunction(n = cluster_embeddings.shape[0],
                                                    mode = 'dense',
                                                    lambdaVal = 0.4,
                                                    separate_rep=False,
                                                    ggsijs = clustered_sijs)
                elif submod_function in ['logdet_pc']:
                    obj = submodlib.LogDeterminantFunction(n = cluster_embeddings.shape[0],
                                                    mode = 'dense',
                                                    lambdaVal = 1,
                                                    sijs = clustered_sijs)

                elif submod_function == 'disp_min_pc':
                    obj = submodlib.DisparityMinFunction(n = cluster_embeddings.shape[0], 
                                                        mode = 'dense', 
                                                        sijs= clustered_sijs)

                elif submod_function == 'disp_sum_pc':
                    obj = submodlib.DisparitySumFunction(n = cluster_embeddings.shape[0], 
                                                        mode = 'dense', 
                                                        sijs= clustered_sijs)
                #print(budget, per_cls_budget, per_cls_cnt)
                if submod_function in ['disp_min_pc', 'gc_pc']:
                    #print(cluster_idx, per_cls_budget[cluster_idx], cluster_embeddings.shape[0])
                    if per_cls_budget[cluster_idx] == cluster_embeddings.shape[0]:
                        clustergreedyList = obj.maximize(budget=per_cls_budget[cluster_idx]-1, optimizer='StochasticGreedy',
                        stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False)
                        rem_elem = list(set(range(cluster_embeddings.shape[0])).difference(set([x[0] for x in clustergreedyList])))[0]
                        rem_gain = clustergreedyList[-1][1]
                        clustergreedyList.append((rem_elem, rem_gain))
                    else:
                        clustergreedyList = obj.maximize(budget=per_cls_budget[cluster_idx], optimizer='StochasticGreedy',
                        stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False)
                else:
                    #print(cluster_idx, per_cls_budget[cluster_idx], cluster_embeddings.shape[0])
                    if per_cls_budget[cluster_idx] == cluster_embeddings.shape[0]:
                        clustergreedyList = obj.maximize(budget=per_cls_budget[cluster_idx]-1, optimizer='LazierThanLazyGreedy',
                        stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False)
                        rem_elem = list(set(range(cluster_embeddings.shape[0])).difference(set([x[0] for x in clustergreedyList])))[0]
                        rem_gain = clustergreedyList[-1][1]
                        clustergreedyList.append((rem_elem, rem_gain))
                    else:
                        clustergreedyList = obj.maximize(budget=per_cls_budget[cluster_idx], optimizer='LazierThanLazyGreedy',
                        stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False)
                cluster_idx += 1
                clusteredgreedylist_with_orig_idxs = [(idxs[x[0]], x[1]) for x in clustergreedyList]
                subset.extend(clusteredgreedylist_with_orig_idxs)
                del obj
            subset.sort(key=lambda x: x[1], reverse=True)
            subsets.append(subset)
            total_time += (time.time() - st_time)

    print("Average Time for Stochastic Greedy Subset Selection is :", total_time)
    #Sorted Label-wise Partition
    # cluster_idxs = {}
    # greedy_idxs = [x[0] for x in subset]
    # for i in greedy_idxs:
    #     if train_labels[i] in cluster_idxs.keys():
    #         cluster_idxs[train_labels[i]].append(i)
    #     else:
    #         cluster_idxs[train_labels[i]] = [i]
    return subsets


def load_dataset(dataset_name, data_dir, seed, return_valid=False, return_test=False):
    if dataset_name == "glue_sst2":
        """
        Load GLUE SST2 dataset. We are only using train and validation splits since the test split doesn't come with gold labels. For testing purposes, we use 5% of train
        dataset as test dataset.
        """
        glue_dataset = datasets.load_dataset("glue", "sst2", cache_dir=data_dir)
        fullset = glue_dataset['train']
        valset = glue_dataset['validation']
        test_set_fraction = 0.05
        seed = 42
        num_fulltrn = len(fullset)
        num_test = int(num_fulltrn * test_set_fraction)
        num_trn = num_fulltrn - num_test
        trainset, testset = random_split(fullset, [num_trn, num_test], generator=torch.Generator().manual_seed(seed))

    elif dataset_name == 'trec6':
        trec6_dataset = datasets.load_dataset("trec", cache_dir=data_dir)
        fullset = trec6_dataset["train"]
        testset = trec6_dataset['test']
        validation_set_fraction = 0.1
        seed = 42
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

    elif dataset_name == 'imdb':
        trec6_dataset = datasets.load_dataset("imdb", cache_dir=data_dir)
        fullset = trec6_dataset["train"]
        testset = trec6_dataset['test']
        validation_set_fraction = 0.1
        seed = 42
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

    elif dataset_name == 'rotten_tomatoes':
        dataset = datasets.load_dataset("rotten_tomatoes", cache_dir=data_dir)
        trainset = dataset["train"]
        valset = dataset['validation']
        testset = dataset['test']
        
    elif dataset_name == 'tweet_eval':
        dataset = datasets.load_dataset("tweet_eval", "emoji", cache_dir=data_dir)
        trainset = dataset["train"]
        valset = dataset['validation']
        testset = dataset['test']

    elif dataset_name == 'cifar10':
        fullset = torchvision.datasets.CIFAR10(root=data_dir, train=True, 
                                              download=True, transform=None)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, 
                                              download=True, transform=None)
        
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

    elif dataset_name == 'cifar100':
        fullset = torchvision.datasets.CIFAR100(root=data_dir, train=True, 
                                              download=True, transform=None)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, 
                                              download=True, transform=None)
        
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))
    
    elif dataset_name == 'tinyimagenet':
        fullset = TinyImageNet(root=data_dir, split='train', download=True, transform=None)
        testset = TinyImageNet(root=data_dir, split='val', download=True, transform=None)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

    elif dataset_name == 'mnist':
        fullset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=None)
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))
    else:
        return None

    if not (return_valid and return_test):
        if return_valid:
            return trainset, valset
        elif return_test:
            return trainset, testset
        else:
            return trainset
    else:
        return trainset, valset, testset


def generate_text_similarity_kernel(dataset, model, stats=True, seed=42, data_dir='../data', device='cpu'):    
    #Assertion Check:
    assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (dataset in list(LABEL_MAPPINGS.keys())), \
    "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."
        
    #Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    if train_dataset.__class__.__name__ == 'Subset':        
        train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset] 
        train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
    else:
        train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
        train_labels = train_dataset[LABEL_MAPPINGS[dataset]]            

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model +  '_train_embeddings.pkl')):
        train_embeddings = compute_text_embeddings(model, train_sentences, device)
        store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_' + model +  '_train_embeddings.pkl'), train_embeddings)
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_train_embeddings.pkl'))

    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_dist_kernel.h5')):
        data_dist = get_cdist(train_embeddings)
        with h5py.File(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_dist_kernel.h5'), 'w') as hf:
            hf.create_dataset("dist_kernel",  data=data_dist)
        
        if stats:
            plt.hist(data_dist, bins = 'auto') 
            plt.savefig(dataset + '_' + model + '_dist_hist.png')


def generate_image_similarity_kernel(dataset, model, stats=True, seed=42, data_dir='../data', device='cpu'):    
    #Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    train_images = [x[0] for x in train_dataset] 
    train_labels = [x[1] for x in train_dataset]

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_train_embeddings.pkl')):
        if model[:3] == 'ViT':
            train_embeddings = compute_vit_image_embeddings(train_images, device)
        else:
            train_embeddings = compute_image_embeddings(model, train_images, device)
        store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_' + model + '_train_embeddings.pkl'), train_embeddings)
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_train_embeddings.pkl'))

    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_dist_kernel.h5')):
        data_dist = get_cdist(train_embeddings)
        with h5py.File(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_dist_kernel.h5'), 'w') as hf:
            hf.create_dataset("dist_kernel",  data=data_dist)
        
        if stats:
            plt.hist(data_dist, bins = 'auto') 
            plt.savefig(dataset + '_' + model + '_dist_hist.png')
    

def generate_image_global_order(dataset, model, submod_function, metric, kw, r2_coefficient, knn, seed=42, data_dir='../data', device='cpu'):    
    #Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    train_images = [x[0] for x in train_dataset] 
    train_labels = [x[1] for x in train_dataset]

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_train_embeddings.pkl')):
        if model == 'ViT':
            train_embeddings = compute_vit_image_embeddings(train_images, device)
        elif model == 'ViT_cls':
            train_embeddings = compute_vit_cls_image_embeddings(train_images, device)
        elif model == 'dino':
            train_embeddings = compute_dino_image_embeddings(train_images, device)
        elif model == 'dino_cls':
            train_embeddings = compute_dino_cls_image_embeddings(train_images, device)
        else:
            train_embeddings = compute_image_embeddings(model, train_images, device)
        store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_' + model + '_train_embeddings.pkl'), train_embeddings)
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_train_embeddings.pkl'))

    # Load global order from pickle file if it exists otherwise compute them and store them.
    if not (os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl')) and
            os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(r2_coefficient) + '_global_r2.pkl')) and
            os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(knn) + '_global_knn.pkl'))):
        global_order, global_knn, global_r2, cluster_idxs = compute_global_ordering(train_embeddings, submod_function=submod_function, kw=kw, r2_coefficient=r2_coefficient, knn=knn, train_labels=train_labels, metric=metric)
        dict2pickle(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl'), {'globalorder': global_order, 'cluster_idxs': cluster_idxs}) 
        dict2pickle(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(r2_coefficient) + '_global_r2.pkl'), {'globalr2': global_r2, 'cluster_idxs': cluster_idxs}) 
        dict2pickle(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(knn) + '_global_knn.pkl'), {'globalknn': global_knn, 'cluster_idxs': cluster_idxs}) 
    else:
        global_order = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl'), 'globalorder')
        cluster_idxs = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl'), 'cluster_idxs')
        global_r2 = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(r2_coefficient) + '_global_r2.pkl'), 'globalr2')
        global_knn = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(knn) + '_global_knn.pkl'), 'globalknn')
    return global_order, global_knn, global_r2, cluster_idxs

    
def generate_text_global_order(dataset, model, submod_function, metric, kw, r2_coefficient, knn, seed=42, data_dir='../data', device='cpu'):    
    #Assertion Check:
    assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (dataset in list(LABEL_MAPPINGS.keys())), \
    "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."
        
    #Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    if train_dataset.__class__.__name__ == 'Subset':        
        train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset] 
        train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
    else:
        train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
        train_labels = train_dataset[LABEL_MAPPINGS[dataset]]            

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model +  '_train_embeddings.pkl')):
        train_embeddings = compute_text_embeddings(model, train_sentences, device)
        store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_' + model +  '_train_embeddings.pkl'), train_embeddings)
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_train_embeddings.pkl'))

    # Load global order from pickle file if it exists otherwise compute them and store them.
    if not (os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl')) and
            os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(r2_coefficient) + '_global_r2.pkl')) and
            os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(knn) + '_global_knn.pkl'))):
        global_order, global_knn, global_r2, cluster_idxs = compute_global_ordering(train_embeddings, submod_function=submod_function, kw=kw, r2_coefficient=r2_coefficient, knn=knn, train_labels=train_labels, metric=metric)
        dict2pickle(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl'), {'globalorder': global_order, 'cluster_idxs': cluster_idxs}) 
        dict2pickle(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(r2_coefficient) + '_global_r2.pkl'), {'globalr2': global_r2, 'cluster_idxs': cluster_idxs}) 
        dict2pickle(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(knn) + '_global_knn.pkl'), {'globalknn': global_knn, 'cluster_idxs': cluster_idxs}) 
    else:
        global_order = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl'), 'globalorder')
        cluster_idxs = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_global_order.pkl'), 'cluster_idxs')
        global_r2 = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(r2_coefficient) + '_global_r2.pkl'), 'globalr2')
        global_knn = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(knn) + '_global_knn.pkl'), 'globalknn')
    return global_order, global_knn, global_r2, cluster_idxs


def generate_image_stochastic_subsets(dataset, model, submod_function, metric, kw, fraction, n_subsets, seed=42, data_dir='../data', device='cpu'):    
    #Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    train_images = [x[0] for x in train_dataset] 
    train_labels = [x[1] for x in train_dataset]

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_train_embeddings.pkl')):
        if model == 'ViT':
            train_embeddings = compute_vit_image_embeddings(train_images, device)
        elif model == 'ViT_cls':
            train_embeddings = compute_vit_cls_image_embeddings(train_images, device)
        elif model == 'dino':
            train_embeddings = compute_dino_image_embeddings(train_images, device)
        elif model == 'dino_cls':
            train_embeddings = compute_dino_cls_image_embeddings(train_images, device)
        else:
            train_embeddings = compute_image_embeddings(model, train_images, device)
        store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_' + model + '_train_embeddings.pkl'), train_embeddings)
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_train_embeddings.pkl'))

    # Load stochastic subsets from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_' + str(fraction) + '_stochastic_subsets.pkl')):
        stochastic_subsets = compute_stochastic_greedy_subsets(train_embeddings, submod_function, train_labels, kw, metric, fraction, n_subsets=n_subsets)
        dict2pickle(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_' + str(fraction) + '_stochastic_subsets.pkl'), {'stochastic_subsets': stochastic_subsets}) 
    else:
        stochastic_subsets = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_' + str(fraction) + '_stochastic_subsets.pkl'), 'stochastic_subsets')
    return stochastic_subsets


def generate_text_stochastic_subsets(dataset, model, submod_function, metric, kw, fraction, n_subsets, seed=42, data_dir='../data', device='cpu'):    
    #Assertion Check:
    assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (dataset in list(LABEL_MAPPINGS.keys())), \
    "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."
        
    #Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    if train_dataset.__class__.__name__ == 'Subset':        
        train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset] 
        train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
    else:
        train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
        train_labels = train_dataset[LABEL_MAPPINGS[dataset]]            

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model +  '_train_embeddings.pkl')):
        train_embeddings = compute_text_embeddings(model, train_sentences, device)
        store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_' + model +  '_train_embeddings.pkl'), train_embeddings)
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_train_embeddings.pkl'))

    # Load stochastic subsets from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_' + str(fraction) + '_stochastic_subsets.pkl')):
        stochastic_subsets = compute_stochastic_greedy_subsets(train_embeddings, submod_function, train_labels, kw, metric, fraction, n_subsets=n_subsets)
        dict2pickle(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_' + str(fraction) + '_stochastic_subsets.pkl'), {'stochastic_subsets': stochastic_subsets}) 
    else:
        stochastic_subsets = pickle2dict(os.path.join(os.path.abspath(data_dir), dataset + '_' + model + '_' + metric + '_' + submod_function + '_' + str(kw) + '_' + str(fraction) + '_stochastic_subsets.pkl'), 'stochastic_subsets')
    return stochastic_subsets


# def analyze_go_wt_diff_init(dataset, model, submod_function, data_dir='../data', device='cpu', seed=42):

#     #Load Arguments
#     #args = parse_args()

#     #Assertion Check:
#     assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (dataset in list(LABEL_MAPPINGS.keys())), \
#     "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."
        
#     #Load Dataset
#     train_dataset = load_dataset(dataset, data_dir, seed)

#     if train_dataset.__class__.__name__ == 'Subset':        
#         train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset] 
#         train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
#     else:
#         train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
#         train_labels = train_dataset[LABEL_MAPPINGS[dataset]]            

#     # Load embeddings from pickle file if it exists otherwise compute them and store them.
#     if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_train_embeddings.pkl')):
#         train_embeddings = compute_text_embeddings(model, train_sentences, device)
#         store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_train_embeddings.pkl'), train_embeddings)
#     else:
#         # Load the embeddings from disc
#         train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_train_embeddings.pkl'))

#     groundset = list(range(train_embeddings.shape[0]))
#     random_inits = random.sample(groundset, 10)
#     subsets = []

#     for random_init in random_inits:
#         remset = [x for x in groundset if x != random_init]
#         remdata = train_embeddings[remset]
#         privatedata = train_embeddings[random_init].reshape(1, -1)
#         global_order = compute_global_ordering(remdata, submod_function= submod_function, train_labels=train_labels, private_embeddings=privatedata)
#         subset = [random_init]
#         rem_subset = [remset[x[0]] for x in global_order]
#         subset.extend(rem_subset)
#         subsets.append(subset)

#     budget = int(0.3 * train_embeddings.shape[0])
    
#     common_fraction = np.zeros((len(subsets), len(subsets)))
#     for i in range(len(subsets)):
#         for j in range(len(subsets)):
#            common_fraction[i][j] = len(set(subsets[i][:budget]).intersection(set(subsets[j][:budget])))/len(set(subsets[i][:budget]))
#     return common_fraction


# def analyze_go_label_dists(dataset, model, submod_function, data_dir='../data', device='cpu', seed=42):

#     #Load Arguments
#     #args = parse_args()
#     fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
#     label_dists_dict = {}
#     #Assertion Check:
#     assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (dataset in list(LABEL_MAPPINGS.keys())), \
#     "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."
        
#     #Load Dataset
#     train_dataset = load_dataset(dataset, data_dir, seed)

#     if train_dataset.__class__.__name__ == 'Subset':        
#         train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset] 
#         train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
#     else:
#         train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
#         train_labels = train_dataset[LABEL_MAPPINGS[dataset]]            

#     # Load embeddings from pickle file if it exists otherwise compute them and store them.
#     if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_train_embeddings.pkl')):
#         train_embeddings = compute_text_embeddings(model, train_sentences, device)
#         store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_train_embeddings.pkl'), train_embeddings)
#     else:
#         # Load the embeddings from disc
#         train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_train_embeddings.pkl'))

#     if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + submod_function + '_global_order.pkl')):
#         global_order, global_knn, global_r2 = compute_global_ordering(train_embeddings, submod_function= submod_function, train_labels=train_labels)
#         store_globalorder(os.path.join(os.path.abspath(data_dir), dataset + '_' + submod_function + '_global_order.pkl'), global_order, global_knn, global_r2)
#     else:
#         global_order, global_knn, global_r2 = load_globalorder(os.path.join(os.path.abspath(data_dir), dataset + '_' + submod_function + '_global_order.pkl'))
    
#     global_order_idxs = [x[0] for x in global_order]
#     num_labels = len(set(train_labels))

#     for fraction in fractions:
#         budget = int(fraction * train_embeddings.shape[0])
#         label_dist = np.array([0]*num_labels)
#         for i in range(budget):
#             label = train_labels[global_order_idxs[i]]
#             label_dist[label] += 1 
#         label_dist = label_dist/budget
#         label_dists_dict[fraction] = label_dist
#     return label_dists_dict


if __name__ == "__main__":
    #args = parse_args()
    #submod_functions = ['fl', 'supfl', 'gc', 'logdet']
    #label_dists_dict = analyze_go_label_dists(args.dataset, args.model, args.submod_function, data_dir=args.data_dir, device=args.device, seed=args.seed)
    train_dataset = load_dataset('cifar10', '../data', 42)

    train_images = [x[0] for x in train_dataset] 
    train_labels = [x[1] for x in train_dataset]
    compute_vit_image_embeddings(train_images, 'cuda')

    print()
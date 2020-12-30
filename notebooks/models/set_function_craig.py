import numpy as np
import torch
from queue import PriorityQueue
import torch.nn.functional as F
import apricot
from torch.utils.data import random_split, SequentialSampler, BatchSampler
import math
from collections import defaultdict

class SetFunction(object):

    def __init__(self, device, train_full_loader, if_convex):
        self.train_loader = train_full_loader
        self.if_convex = if_convex
        self.device = device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def distance(self, x, y, exp=2):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.pow(x - y, exp).sum(2)
        return dist

    def compute_score(self, model):

        self.N = 0
        g_is = []
        with torch.no_grad():
            for i, data_i in enumerate(self.train_loader, 0):
                inputs_i, target_i = data_i
                inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
                # print(i,end=",")
                self.N += inputs_i.size()[0]
                if not self.if_convex:
                    scores_i = F.softmax(model(inputs_i), dim=1)
                    y_i = torch.zeros(target_i.size(0), scores_i.size(1)).to(self.device)
                    y_i[range(y_i.shape[0]), target_i] = 1

                    g_is.append(scores_i - y_i)
                else:
                    g_is.append(inputs_i)

            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                # print(i,end=",")
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j)
        dist = self.dist_mat.sum(1)
        bestId = torch.argmin(dist).item()
        self.min_dist = self.dist_mat[bestId].to(self.device)
        return bestId

    def compute_gamma(self, idxs):
        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs].to(self.device)
        rep = torch.argmin(best, axis=0)
        for i in rep:
            gamma[i] += 1
        return gamma

    def naive_greedy_max(self, budget, model):
        id_first = self.compute_score(model)
        numSelected = 1
        greedyList = [id_first]
        not_selected = [i for i in range(self.N)]
        not_selected.remove(id_first)
        while (numSelected < budget):
            bestGain = -np.inf
            bestId = -1
            for i in not_selected:
                gain = (self.min_dist - torch.min(self.min_dist, self.dist_mat[
                    i])).sum()  # L([0]+greedyList+[i]) - L([0]+greedyList)  #s_0 = self.x_trn[0]
                if bestGain < gain.item():
                    bestGain = gain.item()
                    bestId = i
            if bestId == -1:  # Make sure you understand this logic
                bestId = int(np.random.choice(np.array(not_selected), size=1, replace=False)[0])
            greedyList.append(bestId)
            not_selected.remove(bestId)
            numSelected += 1
            self.min_dist = torch.min(self.min_dist, self.dist_mat[bestId])  # .to(self.device))
        gamma = self.compute_gamma(greedyList)
        return greedyList, gamma

    def lazy_greedy_max(self, budget, model):
        id_first = self.compute_score(model)
        self.gains = PriorityQueue()
        for i in range(self.N):
            if i == id_first:
                continue
            curr_gain = (self.min_dist - torch.min(self.min_dist, self.dist_mat[i])).sum()
            self.gains.put((-curr_gain.item(), i))
        numSelected = 2
        second = self.gains.get()
        greedyList = [id_first, second[1]]
        self.min_dist = torch.min(self.min_dist, self.dist_mat[second[1]])
        while (numSelected < budget):
            if self.gains.empty():
                break
            elif self.gains.qsize() == 1:
                bestId = self.gains.get()[1]
            else:
                bestGain = -np.inf
                bestId = None
                while True:
                    first = self.gains.get()
                    if bestId == first[1]:
                        break
                    curr_gain = (self.min_dist - torch.min(self.min_dist, self.dist_mat[first[1]])).sum()
                    self.gains.put((-curr_gain.item(), first[1]))
                    if curr_gain.item() >= bestGain:
                        bestGain = curr_gain.item()
                        bestId = first[1]
            greedyList.append(bestId)
            numSelected += 1
            self.min_dist = torch.min(self.min_dist, self.dist_mat[bestId])
        gamma = self.compute_gamma(greedyList)
        return greedyList, gamma


class SetFunction2(object):

    def __init__(self, device, train_full_loader, if_convex):
        self.train_loader = train_full_loader
        self.if_convex = if_convex
        self.device = device

    def distance(self, x, y, exp=2):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def compute_score(self, model):
        self.N = 0
        g_is = []
        with torch.no_grad():
            for i, data_i in enumerate(self.train_loader, 0):
                inputs_i, target_i = data_i
                inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
                self.N += inputs_i.size()[0]

                if not self.if_convex:
                    scores_i = F.softmax(model(inputs_i), dim=1)
                    y_i = torch.zeros(target_i.size(0), scores_i.size(1)).to(self.device)
                    y_i[range(y_i.shape[0]), target_i] = 1
                    g_is.append(scores_i - y_i)
                else:
                    g_is.append(inputs_i)

            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                # print(i,end=",")
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False

                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j)
        self.dist_mat = self.dist_mat.cpu().numpy()

    def compute_gamma(self, idxs):
        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs]  # .to(self.device)
        rep = np.argmax(best, axis=0)
        for i in rep:
            gamma[i] += 1
        return gamma

    def get_similarity_kernel(self):
        for i, data_i in enumerate(self.train_loader, 0):
            if i == 0:
                _, targets = data_i
            else:
                _, target_i = data_i
                targets = torch.cat((targets, target_i), dim=0)
        kernel = np.zeros((targets.shape[0], targets.shape[0]))
        targets = targets.cpu().numpy()
        for target in np.unique(targets):
            x = np.where(targets == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel

    def lazy_greedy_max(self, budget, model):
        self.compute_score(model)
        kernel = self.get_similarity_kernel()
        fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                          n_samples=budget)
        self.dist_mat = self.dist_mat * kernel
        sim_sub = fl.fit_transform(self.dist_mat)
        greedyList = list(np.argmax(sim_sub, axis=1))
        gamma = self.compute_gamma(greedyList)
        return greedyList, gamma


class DeepSetFunction(object):
    def __init__(self, device, model, trainset, N_trn, batch_size, if_convex):
        self.trainset = trainset
        self.model = model
        self.if_convex = if_convex
        self.device = device
        self.N_trn = N_trn
        self.batch_size = batch_size

    def distance(self, x, y, exp=2):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def compute_score(self, model_params):
        self.model.load_state_dict(model_params)
        self.N = 0
        g_is = []
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(self.N_trn)), self.batch_size, drop_last=False))][0])
        with torch.no_grad():
            for batch_idx in batch_wise_indices:
                inputs_i = torch.cat(
                    [self.trainset[x][0].view(-1, self.trainset[x][0].shape[0], self.trainset[x][0].shape[1],
                                              self.trainset[x][0].shape[2]) for x
                     in batch_idx], dim=0).type(torch.float)
                target_i = torch.tensor([self.trainset[x][1] for x in batch_idx])
                inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
                self.N += inputs_i.size()[0]
                if not self.if_convex:
                    scores_i = F.softmax(self.model(inputs_i), dim=1)
                    y_i = torch.zeros(target_i.size(0), scores_i.size(1)).to(self.device)
                    y_i[range(y_i.shape[0]), target_i] = 1
                    g_is.append(scores_i - y_i)
                else:
                    g_is.append(inputs_i)

            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j).cpu()
        self.dist_mat = self.dist_mat.cpu().numpy()

    def compute_gamma(self, idxs):
        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs]  # .to(self.device)
        rep = np.argmax(best, axis=0)
        for i in rep:
            gamma[i] += 1
        return gamma

    def get_similarity_kernel(self):
        targets = np.array([self.trainset[x][1] for x in range(self.N_trn)])
        kernel = np.zeros((targets.shape[0], targets.shape[0]))
        for target in np.unique(targets):
            x = np.where(targets == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel

    def lazy_greedy_max(self, budget, model_params):

        self.compute_score(model_params)
        kernel = self.get_similarity_kernel()
        fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                          n_samples=budget)
        self.dist_mat = self.dist_mat * kernel
        sim_sub = fl.fit_transform(self.dist_mat)
        greedyList = list(np.argmax(sim_sub, axis=1))
        gamma = self.compute_gamma(greedyList)
        return greedyList, gamma


class PerClassDeepSetFunction(object):

    def __init__(self, device, model, trainset, N_trn, batch_size, if_convex):
        self.trainset = trainset
        self.model = model
        self.if_convex = if_convex
        self.device = device
        self.N_trn = N_trn
        self.batch_size = batch_size

    def distance(self, x, y, exp=2):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def compute_score(self, model_params, idxs):
        self.model.load_state_dict(model_params)
        self.N = 0
        g_is = []

        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(idxs), self.batch_size, drop_last=False))][0])
        with torch.no_grad():
            for batch_idx in batch_wise_indices:
                inputs_i = torch.cat(
                    [self.trainset[x][0].view(-1, self.trainset[x][0].shape[0], self.trainset[x][0].shape[1],
                                              self.trainset[x][0].shape[2]) for x in idxs[batch_idx]], dim=0).type(torch.float)
                target_i = torch.tensor([self.trainset[x][1] for x in idxs[batch_idx]])
                inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
                self.N += inputs_i.size()[0]
                if not self.if_convex:
                    scores_i = F.softmax(self.model(inputs_i), dim=1)
                    y_i = torch.zeros(target_i.size(0), scores_i.size(1)).to(self.device)
                    y_i[range(y_i.shape[0]), target_i] = 1
                    g_is.append(scores_i - y_i)
                else:
                    g_is.append(inputs_i)

            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j).cpu()
        self.dist_mat = self.dist_mat.cpu().numpy()

    def compute_gamma(self, idxs):
        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs]  # .to(self.device)
        rep = np.argmax(best, axis=0)
        for i in rep:
            gamma[i] += 1
        return gamma

    def get_similarity_kernel(self):
        targets = np.array([self.trainset[x][1] for x in range(self.N_trn)])
        kernel = np.zeros((targets.shape[0], targets.shape[0]))
        for target in np.unique(targets):
            x = np.where(targets == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel

    def lazy_greedy_max(self, budget, model_params):
        labels = self.trainset.dataset.targets[self.trainset.indices]
        classes = len(labels.unique())
        per_class_bud = int(budget / classes)
        total_greedy_list = []
        gammas = []
        for i in range(classes):
            idxs = torch.where(labels == i)[0]
            self.compute_score(model_params, idxs)
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=per_class_bud)
            sim_sub = fl.fit_transform(self.dist_mat)
            greedyList = list(np.argmax(sim_sub, axis=1))
            gamma = self.compute_gamma(greedyList)
            total_greedy_list.extend(idxs[greedyList])
            gammas.extend(gamma)
        return total_greedy_list, gammas


class PerClassNonDeepSetFunction(object):
    def __init__(self, device, x_trn, y_trn, model, N_trn, batch_size, if_convex):
        self.x_trn = x_trn
        self.y_trn = y_trn
        self.model = model
        self.if_convex = if_convex
        self.device = device
        self.N_trn = N_trn
        self.batch_size = batch_size

    def distance(self, x, y, exp=2):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def compute_score(self, model_params, idxs):
        self.model.load_state_dict(model_params)
        self.N = 0
        g_is = []
        x_temp = self.x_trn[idxs]
        y_temp = self.y_trn[idxs]
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(len(y_temp))), self.batch_size, drop_last=False))][0])
        with torch.no_grad():
            for batch_idx in batch_wise_indices:
                inputs_i = x_temp[batch_idx].type(torch.float)
                target_i = y_temp[batch_idx]
                inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
                self.N += inputs_i.size()[0]
                if not self.if_convex:
                    scores_i = F.softmax(self.model(inputs_i), dim=1)
                    y_i = torch.zeros(target_i.size(0), scores_i.size(1)).to(self.device)
                    y_i[range(y_i.shape[0]), target_i] = 1
                    g_is.append(scores_i - y_i)
                else:
                    g_is.append(inputs_i)
            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j).cpu()
        self.dist_mat = self.dist_mat.cpu().numpy()

    def compute_gamma(self, idxs):
        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs]  # .to(self.device)
        rep = np.argmax(best, axis=0)
        for i in rep:
            gamma[i] += 1
        return gamma

    def lazy_greedy_max(self, budget, model_params):
        classes = torch.unique(self.y_trn)
        per_class_bud = int(budget / len(classes))
        total_greedy_list = []
        gammas = []
        for i in classes:
            idxs = torch.where(self.y_trn == i)[0]
            self.compute_score(model_params, idxs)
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=per_class_bud)
            sim_sub = fl.fit_transform(self.dist_mat)
            greedyList = list(np.argmax(sim_sub, axis=1))
            gamma = self.compute_gamma(greedyList)
            total_greedy_list.extend(idxs[greedyList])
            gammas.extend(gamma)
        return total_greedy_list, gammas


class SetFunctionCRAIG_Super(object):

    def __init__(self, device, X_trn, Y_trn, if_convex):  # , valid_loader):

        self.x_trn = X_trn
        self.y_trn = Y_trn
        self.if_convex = if_convex
        self.device = device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def distance(self, x, y, exp=2):

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        dist = torch.pow(x - y, exp).sum(2)
        return dist  # torch.sqrt(dist)

        # dist = torch.abs(x-y).sum(2)
        # return dist

    def class_wise(self, bud, model):

        torch.cuda.empty_cache()

        classes = torch.unique(self.y_trn)

        self.N = self.y_trn.shape[0]
        greedyList = []
        full_gamma = []

        for i in classes:

            idx = (self.y_trn == i).nonzero().flatten()
            idx.tolist()

            self.curr_x_trn = self.x_trn[idx]
            self.curr_y_trn = self.y_trn[idx]
            self.curr_N = self.curr_y_trn.shape[0]
            # self.curr_bud = math.ceil(bud*self.curr_N / self.N)

            id_first = self.compute_score(model)
            subset, gamma = self.lazy_greedy_max(math.ceil(bud * self.curr_N / self.N), id_first)
            # print("Class ",i,"fnished")

            for j in range(len(subset)):
                greedyList.append(idx[subset[j]])
                full_gamma.append(gamma[j])

        return greedyList, full_gamma

    def compute_score(self, model):

        with torch.no_grad():

            self.dist_mat = torch.zeros([self.curr_N, self.curr_N], dtype=torch.float32).to(self.device)

            train_batch_size = 1200
            train_loader = []
            for item in range(math.ceil(self.curr_N / train_batch_size)):
                inputs = self.curr_x_trn[item * train_batch_size:(item + 1) * train_batch_size]
                target = self.curr_y_trn[item * train_batch_size:(item + 1) * train_batch_size]
                train_loader.append((inputs, target))

            first_i = True
            g_is = []

            for i, data_i in enumerate(train_loader, 0):  # iter(train_loader).next()

                inputs_i, target_i = data_i
                inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)

                if not self.if_convex:
                    scores_i = model(inputs_i)
                    y_i = torch.zeros(target_i.size(0), scores_i.size(1)).to(self.device)
                    y_i[range(y_i.shape[0]), target_i] = 1

                    g_is.append(F.softmax(scores_i, dim=1), - y_i)
                else:
                    g_is.append(inputs_i)

            first_i = True
            for i, g_i in enumerate(g_is, 0):

                # print(i,end=",")
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False

                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j)

        dist = self.dist_mat.sum(1)
        bestId = torch.argmin(dist).item()

        # self.dist_mat = self.dist_mat.to(self.device)

        self.min_dist = self.dist_mat[bestId]  # .to(self.device)

        return bestId

    def compute_gamma(self, idxs):

        gamma = [0 for i in range(len(idxs))]

        # self.dist_mat = self.dist_mat.cpu()
        best = self.dist_mat[idxs]
        # print(best[0])
        rep = torch.argmin(best, axis=0)

        for i in rep:
            gamma[i] += 1

        return gamma

    def lazy_greedy_max(self, budget, id_first):

        self.gains = PriorityQueue()
        for i in range(self.curr_N):

            if i == id_first:
                continue
            curr_gain = (self.min_dist - torch.min(self.min_dist, self.dist_mat[i])).sum()
            self.gains.put((-curr_gain.item(), i))

        numSelected = 2
        second = self.gains.get()
        greedyList = [id_first, second[1]]
        self.min_dist = torch.min(self.min_dist, self.dist_mat[second[1]])

        while (numSelected < budget):

            # print(len(greedyList))
            if self.gains.empty():
                break

            elif self.gains.qsize() == 1:
                bestId = self.gains.get()[1]

            else:

                bestGain = -np.inf
                bestId = None

                while True:

                    first = self.gains.get()

                    if bestId == first[1]:
                        break

                    curr_gain = (self.min_dist - torch.min(self.min_dist, self.dist_mat[first[1]])).sum()
                    self.gains.put((-curr_gain.item(), first[1]))

                    if curr_gain.item() >= bestGain:
                        bestGain = curr_gain.item()
                        bestId = first[1]

            greedyList.append(bestId)
            numSelected += 1

            self.min_dist = torch.min(self.min_dist, self.dist_mat[bestId])

        gamma = self.compute_gamma(greedyList)
        return greedyList, gamma

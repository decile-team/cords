import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import apricot
from selectionstrategies.supervisedlearning.dataselectionstrategy import DataSelectionStrategy


class CRAIGStrategy(DataSelectionStrategy):

    def __init__(self, trainloader, valloader, model, loss_type,
                 device, num_classes, linear_layer, if_convex):
        super().__init__(trainloader, valloader, model, linear_layer)

        self.loss_type = loss_type  # Make sure it has reduction='none' instead of default
        self.device = device
        self.num_classes = num_classes
        self.if_convex = if_convex

    def distance(self, x, y, exp=2):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.pow(x - y, exp).sum(2)
        #dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def compute_score(self, model_params, idxs):
        trainset = self.trainloader.sampler.data_source
        subset_loader = torch.utils.data.DataLoader(trainset, batch_size=self.trainloader.batch_size, shuffle=False,
                                                   sampler=SubsetRandomSampler(idxs),
                                                   pin_memory=True)
        self.model.load_state_dict(model_params)
        self.N = 0
        g_is = []

        with torch.no_grad():
            if self.if_convex:
                for batch_idx, (inputs, targets) in enumerate(subset_loader):
                    inputs, targets = inputs, targets
                    self.N += inputs.size()[0]
                    g_is.append(inputs.view(inputs.size()[0], -1))
            else:
                embDim = self.model.get_embedding_dim()
                for batch_idx, (inputs, targets) in enumerate(subset_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    self.N += inputs.size()[0]
                    with torch.no_grad():
                        out, l1 = self.model(inputs, last=True)
                        data = F.softmax(out, dim=1)
                    outputs = torch.zeros(len(inputs), self.num_classes).to(self.device)
                    outputs.scatter_(1, targets.view(-1, 1), 1)
                    l0_grads = data - outputs
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        g_is.append(torch.cat((l0_grads, l1_grads), dim=1))
                    else:
                        g_is.append(l0_grads)

            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j)
        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()

    def compute_gamma(self, idxs):
        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs]  # .to(self.device)
        rep = np.argmax(best, axis=0)
        for i in rep:
            gamma[i] += 1
        return gamma

    def get_similarity_kernel(self):
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                tmp_target = targets
            else:
                tmp_target_i = targets
                targets = torch.cat((tmp_target, tmp_target_i), dim=0)
        kernel = np.zeros((targets.shape[0], targets.shape[0]))
        for target in np.unique(targets):
            x = np.where(targets == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel

    def select(self, budget, model_params, optimizer):
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        per_class_bud = int(budget / self.num_classes)
        total_greedy_list = []
        gammas = []

        for i in range(self.num_classes):
            idxs = torch.where(labels == i)[0]
            self.compute_score(model_params, idxs)
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=per_class_bud,
                                                                              optimizer=optimizer)
            sim_sub = fl.fit_transform(self.dist_mat)
            greedyList = list(np.argmax(sim_sub, axis=1))
            gamma = self.compute_gamma(greedyList)
            total_greedy_list.extend(idxs[greedyList])
            gammas.extend(gamma)
        return total_greedy_list, gammas

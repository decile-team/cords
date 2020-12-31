import numpy as np
import torch
import torch.nn.functional as F


class DataSelectionStrategy(object):

    def __init__(self, trainloader, valloader, model, linear_layer):
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader
        self.model = model
        self.N_trn = len(trainloader.sampler.data_source)
        self.grads_per_elem = None
        self.numSelected = 0
        self.linear_layer = linear_layer


    def select(self, budget, model_dict):
        pass

    def compute_gradients(self):
        embDim = self.model.get_embedding_dim()
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
            if batch_idx == 0:
                with torch.no_grad():
                    out, l1 = self.model(inputs, last=True)
                    data = F.softmax(out, dim=1)
                outputs = torch.zeros(len(inputs), self.num_classes).to(self.device)
                outputs.scatter_(1, targets.view(-1, 1), 1)
                l0_grads = data - outputs
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.num_classes)
            else:
                with torch.no_grad():
                    out, l1 = self.model(inputs, last=True)
                    data = F.softmax(out, dim=1)
                outputs = torch.zeros(len(inputs), self.num_classes).to(self.device)
                outputs.scatter_(1, targets.view(-1, 1), 1)
                batch_l0_grads = data - outputs
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                if self.linear_layer:
                    batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                    batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                    l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
        torch.cuda.empty_cache()
        print("Per Element Gradient Computation is Completed")
        if self.linear_layer:
            self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        else:
            self.grads_per_elem = l0_grads

    def update_model(self, model_dict):
        self.model.load_state_dict(model_dict)

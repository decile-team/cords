import time
import copy
import datetime
import os
import subprocess
import sys
import math
import numpy as np
import apricot
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler
from torch.autograd import Variable
import torch.nn.functional as F


class BaseStrategy(object):

    def __init__():
        pass

    def select(budget, args):
        pass
             
        return indices, gammas 
        

    def compute_gradients()
        pass

    def update_model()
        pass


import numpy as np

np.seterr(all='raise')
from numpy.linalg import cond
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import sparse as sp
from scipy.linalg import lstsq
from scipy.linalg import solve
from scipy.optimize import nnls

import torch



def OptimalWeights(A, b, tol=1E-4, nnz=None, device="cpu"):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      nnz = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    '''

    sum_sel_grad = torch.zeros_like(b,device= device)
    w = 1.0

    final_indices = []
    remainList = [i for i in range(A.shape[0])]

    b_norm = b.norm()

    for i in range(nnz):

        projection = (A + sum_sel_grad - w*b).norm(dim=1)
        index = torch.argmin(projection).item()

        sum_sel_grad += A[index]
        w = torch.dot(A[index],sum_sel_grad)/b_norm

        actual_idx = remainList[index]
        final_indices.append(actual_idx)

        remainList.remove(actual_idx)        
        A = torch.cat((A[:index], A[index + 1:]), dim=0)

    return final_indices, [w for _ in range(nnz)]


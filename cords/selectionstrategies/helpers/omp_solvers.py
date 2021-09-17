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


# NOTE: Textbook Primal-Dual IPM: Boyd & Vandenberghe, ``Chapter 11: Interior-point Methods," Convex Optimization, 2004.
# NOTE: Works on toy problems but struggles in word embedding recovery setting (n>10000).
def NonnegativeBP(A, b, x0=None, tol=1E-4, niter=100, biter=32):
    '''solves min |x|_1 s.t. Ax=b,x>=0 using a Primal-Dual Interior Point Method
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      x0: starting point; if None sets x0=A^T(AA^T)^(-1)b
      tol: solver tolerance
      niter: maximum length of central path
      biter: maximum number of steps in backtracking line search
    Returns:
      vector of length n
    '''

    AT = A.T
    d, n = A.shape
    alpha = 0.01
    beta = 0.5
    mu = 20
    e = np.ones(n)

    if x0 is None:
        x = tol / np.sqrt(n) * e
        x += nnls(A, b - A.dot(x))[0]
    else:
        x = np.copy(x0)
    lam = 1.0 / x
    v = -A.dot(lam)
    t = mu * d
    rp = A.dot(x) - b
    rd = 1.0 - lam + AT.dot(v)

    for i in range(niter):

        oot = 1.0 / t
        rc = lam * x - oot
        resnorm = np.sqrt(norm(rd) ** 2 + norm(rc) ** 2 + norm(rp) ** 2)

        try:
            dv = solve(A.dot(AT * (x / lam)[:, np.newaxis]), rp - A.dot((rc + x * rd) / lam), assume_a='pos')
        except np.linalg.linalg.LinAlgError:
            return x
        dlam = AT.dot(dv) + rd
        dx = -(rc + x * dlam) / lam

        ind = np.less(dlam, 0.0)
        s = 0.99 * min(1.0, min(-lam[ind] / dlam[ind])) if np.any(ind) else 0.99
        for j in range(biter):
            xp = x + s * dx
            lamp = lam + s * dlam
            vp = v + s * dv
            s *= beta
            if all(xp > 0.0) and np.sqrt(
                    norm(1.0 - lamp + AT.dot(vp)) ** 2 + norm(lamp * xp - oot) ** 2 + norm(A.dot(xp) - b) ** 2) <= (
                    1.0 - alpha * s) * resnorm:
                break
        else:
            break

        eta = np.inner(lam, x)
        rp = A.dot(xp) - b
        rd = 1.0 - lamp + AT.dot(vp)
        if max(eta, norm(rp), norm(rd)) < tol:
            return x
        x = xp
        lam = lamp
        v = vp
        t = mu * d / eta

    return x


# NOTE: Ported to Python from l1-MAGIC: Cand\'es & Romberg, ``l_1-MAGIC: Recovery of Sparse Signals via Convex Programming," Technical Report, 2005.
def BasisPursuit(A, b, x0=None, ATinvAAT=None, positive=False, tol=1E-4, niter=100, biter=32):
    '''solves min |x|_1 s.t. Ax=b using a Primal-Dual Interior Point Method
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      x0: starting point; if None sets x0=A^T(AA^T)^(-1)b
      ATinvAAT: precomputed matrix A^T(AA^T)^(-1); computed if None; ignored if not x0 is None
      positive: only allow positive nonzero coefficients
      tol: solver tolerance
      niter: maximum length of central path
      biter: maximum number of steps in backtracking line search
    Returns:
      vector of length n
    '''

    if positive:
        return NonnegativeBP(A, b, x0=x0, tol=tol, niter=niter, biter=biter)

    AT = A.T
    d, n = A.shape
    alpha = 0.01
    beta = 0.5
    mu = 10
    e = np.ones(n)
    gradf0 = np.hstack([np.zeros(n), e])

    if x0 is None:
        if ATinvAAT is None:
            ATinvAAT = AT.dot(inv(A.dot(AT)))
        x = ATinvAAT.dot(b)
    else:
        x = np.copy(x0)
    absx = np.abs(x)
    u = 0.95 * absx + 0.1 * max(absx)

    fu1 = x - u
    fu2 = -x - u
    lamu1 = -1.0 / fu1
    lamu2 = -1.0 / fu2
    v = A.dot(lamu2 - lamu1)
    ATv = AT.dot(v)
    sdg = -(np.inner(fu1, lamu1) + np.inner(fu2, lamu2))
    tau = 2.0 * n * mu / sdg
    ootau = 1.0 / tau

    rcent = np.hstack([-lamu1 * fu1, -lamu2 * fu2]) - ootau
    rdual = gradf0 + np.hstack([lamu1 - lamu2 + ATv, -lamu1 - lamu2])
    rpri = A.dot(x) - b
    resnorm = np.sqrt(norm(rdual) ** 2 + norm(rcent) ** 2 + norm(rpri) ** 2)
    rdp = np.empty(2 * n)
    rcp = np.empty(2 * n)

    for i in range(niter):

        oofu1 = 1.0 / fu1
        oofu2 = 1.0 / fu2
        w1 = -ootau * (oofu2 - oofu1) - ATv
        w2 = -1.0 - ootau * (oofu1 + oofu2)
        w3 = -rpri

        lamu1xoofu1 = lamu1 * oofu1
        lamu2xoofu2 = lamu2 * oofu2
        sig1 = -lamu1xoofu1 - lamu2xoofu2
        sig2 = lamu1xoofu1 - lamu2xoofu2
        sigx = sig1 - sig2 ** 2 / sig1
        if min(np.abs(sigx)) == 0.0:
            break

        w1p = -(w3 - A.dot(w1 / sigx - w2 * sig2 / (sigx * sig1)))
        H11p = A.dot(AT * (e / sigx)[:, np.newaxis])
        if min(sigx) > 0.0:
            dv = solve(H11p, w1p, assume_a='pos')
        else:
            dv = solve(H11p, w1p, assume_a='sym')
        dx = (w1 - w2 * sig2 / sig1 - AT.dot(dv)) / sigx
        Adx = A.dot(dx)
        ATdv = AT.dot(dv)

        du = (w2 - sig2 * dx) / sig1
        dlamu1 = lamu1xoofu1 * (du - dx) - lamu1 - ootau * oofu1
        dlamu2 = lamu2xoofu2 * (dx + du) - lamu2 - ootau * oofu2

        s = 1.0
        indp = np.less(dlamu1, 0.0)
        indn = np.less(dlamu2, 0.0)
        if np.any(indp):
            s = min(s, min(-lamu1[indp] / dlamu1[indp]))
        if np.any(indn):
            s = min(s, min(-lamu2[indn] / dlamu2[indn]))
        indp = np.greater(dx - du, 0.0)
        indn = np.greater(-dx - du, 0.0)
        if np.any(indp):
            s = min(s, min(-fu1[indp] / (dx[indp] - du[indp])))
        if np.any(indn):
            s = min(s, min(-fu2[indn] / (-dx[indn] - du[indn])))
        s = 0.99 * s

        for j in range(biter):
            xp = x + s * dx
            up = u + s * du
            vp = v + s * dv
            ATvp = ATv + s * ATdv
            lamu1p = lamu1 + s * dlamu1
            lamu2p = lamu2 + s * dlamu2
            fu1p = xp - up
            fu2p = -xp - up
            rdp[:n] = lamu1p - lamu2p + ATvp
            rdp[n:] = -lamu1p - lamu2p
            rdp += gradf0
            rcp[:n] = -lamu1p * fu1p
            rcp[n:] = lamu2p * fu2p
            rcp -= ootau
            rpp = rpri + s * Adx
            s *= beta
            if np.sqrt(norm(rdp) ** 2 + norm(rcp) ** 2 + norm(rpp) ** 2) <= (1 - alpha * s) * resnorm:
                break
        else:
            break

        x = xp
        lamu1 = lamu1p
        lamu2 = lamu2p
        fu1 = fu1p
        fu2 = fu2p
        sdg = -(np.inner(fu1, lamu1) + np.inner(fu2, lamu2))
        if sdg < tol:
            return x

        u = up
        v = vp
        ATv = ATvp
        tau = 2.0 * n * mu / sdg
        rpri = rpp
        rcent[:n] = lamu1 * fu1
        rcent[n:] = lamu2 * fu2
        ootau = 1.0 / tau
        rcent -= ootau
        rdual[:n] = lamu1 - lamu2 + ATv
        rdual[n:] = -lamu1 + lamu2
        rdual += gradf0
        resnorm = np.sqrt(norm(rdual) ** 2 + norm(rcent) ** 2 + norm(rpri) ** 2)

    return x


BP = BasisPursuit


# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP(A, b, tol=1E-4, nnz=None, positive=False):
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

    AT = A.T
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = np.zeros(n)
    resid = np.copy(b)
    normb = norm(b)
    indices = []

    for i in range(nnz):
        if norm(resid) / normb < tol:
            break
        projections = AT.dot(resid)
        if positive:
            index = np.argmax(projections)
        else:
            index = np.argmax(abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / A_i.T.dot(A_i)
        else:
            A_i = np.vstack([A_i, A[:, index]])
            x_i = solve(A_i.dot(A_i.T), A_i.dot(b), assume_a='sym')
            if positive:
                while min(x_i) < 0.0:
                    argmin = np.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = np.vstack([A_i[:argmin], A_i[argmin + 1:]])
                    x_i = solve(A_i.dot(A_i.T), A_i.dot(b), assume_a='sym')
        resid = b - A_i.T.dot(x_i)

    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    return x


# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG_NNLS(A, b, tol=1E-4, nnz=None, positive=False, lam=1):
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

    AT = A.T
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = np.zeros(n)
    resid = np.copy(b)
    normb = norm(b)
    indices = []

    for i in range(nnz):
        if norm(resid) / normb < tol:
            break
        projections = AT.dot(resid)
        if positive:
            index = np.argmax(projections)
        else:
            index = np.argmax(abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / A_i.T.dot(A_i)
        else:
            A_i = np.vstack([A_i, A[:, index]])
            if positive:
                x_i = nnls(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
            else:
                x_i = lstsq(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
                # print(x_i)
        # print(b.shape,A_i.T.shape,x_i.shape)
        resid = b - A_i.T.dot(x_i)

    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    return x


# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG(A, b, tol=1E-4, nnz=None, positive=False, lam=1):
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
    AT = A.T
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = np.zeros(n)
    resid = np.copy(b)
    normb = norm(b)
    indices = []

    for i in range(nnz):
        if norm(resid) / normb < tol:
            break
        projections = AT.dot(resid)
        if positive:
            index = np.argmax(projections)
        else:
            index = np.argmax(abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / A_i.T.dot(A_i)
        else:
            A_i = np.vstack([A_i, A[:, index]])
            x_i = lstsq(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
            # print(x_i.shape)
            if positive:
                while min(x_i) < 0.0:
                    # print("Negative",b.shape,A_i.T.shape,x_i.shape)
                    argmin = np.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = np.vstack([A_i[:argmin], A_i[argmin + 1:]])
                    x_i = lstsq(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
        resid = b - A_i.T.dot(x_i)
    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    return x


# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG_Parallel1(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
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
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []

    argmin = torch.tensor([-1])
    for i in range(nnz):
        if resid.norm().item() / normb < tol:
            break
        projections = torch.matmul(AT, resid)  # AT.dot(resid)
        # print("Projections",projections.shape)

        if positive:
            index = torch.argmax(projections)
        else:
            index = torch.argmax(torch.abs(projections))

        if index not in indices:
            indices.append(index)

        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / torch.dot(A_i, A_i).view(-1)  # A_i.T.dot(A_i)
            A_i = A[:, index].view(1, -1)
        else:
            # print(indices)
            A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)  # np.vstack([A_i, A[:,index]])
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)
            # print(x_i.shape)
            if positive:
                while min(x_i) < 0.0:
                    # print("Negative",b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape)
                    argmin = torch.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),
                                    dim=0)  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                    if argmin.item() == A_i.shape[0]:
                        break
                    # print(argmin.item(),A_i.shape[0],index.item())
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)
        if argmin.item() == A_i.shape[0]:
            break
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)  # A_i.T.dot(x_i)
    x_i = x_i.view(-1)
    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    return x


# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG_Parallel(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
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
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []
    argmin = torch.tensor([-1])
    for i in range(nnz):
        if resid.norm().item() / normb < tol:
            break
        projections = torch.matmul(AT, resid)  # AT.dot(resid)
        # print("Projections",projections.shape)
        if positive:
            index = torch.argmax(projections)
        else:
            index = torch.argmax(torch.abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / torch.dot(A_i, A_i).view(-1)  # A_i.T.dot(A_i)
            A_i = A[:, index].view(1, -1)
        else:
            # print(indices)
            A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)  # np.vstack([A_i, A[:,index]])
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
            # print(x_i.shape)
            if positive:

                while min(x_i) < 0.0:
                    # print("Negative",b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape)
                    argmin = torch.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),
                                    dim=0)  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                    if argmin.item() == A_i.shape[0]:
                        break
                    # print(argmin.item(),A_i.shape[0],index.item())
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
        if argmin.item() == A_i.shape[0]:
            break
        # print(b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape,\
        #  torch.matmul(torch.transpose(A_i, 0, 1), x_i).shape)
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)  # A_i.T.dot(x_i)
        # print("REsID",resid.shape)

    x_i = x_i.view(-1)
    # print(x_i.shape)
    # print(len(indices))
    for i, index in enumerate(indices):
        # print(i,index,end="\t")
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    # print(x[indices])
    return x


# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG_NNLS_Parallel(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
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
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []
    argmin = torch.tensor([-1])
    for i in range(nnz):
        if resid.norm().item() / normb < tol:
            break
        projections = torch.matmul(AT, resid)  # AT.dot(resid)
        # print("Projections",projections.shape)
        if positive:
            index = torch.argmax(projections)
        else:
            index = torch.argmax(torch.abs(projections))
        if index not in indices:
            indices.append(index)
            #break
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / torch.dot(A_i, A_i).view(-1)  # A_i.T.dot(A_i)
            A_i = A[:, index].view(1, -1)
        else:
            # print(indices)
            A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)  # np.vstack([A_i, A[:,index]])
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            if positive:
                x_i, _ = nnls(temp.cpu().numpy(), torch.matmul(A_i, b).view(-1).cpu().numpy())
                x_i = torch.from_numpy(x_i).float().to(device=device)
            else:
                x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)  # A_i.T.dot(x_i)
    x_i = x_i.view(-1)
    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    # print(x[indices])
    return x


# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def MatchingPursuit(A, b, tol=1E-4, nnz=None, positive=False, orthogonal=False):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      nnz = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
      orthogonal: use Orthogonal Matching Pursuit (OMP)
    Returns:
       vector of length n
    '''

    if orthogonal:
        return OrthogonalMP(A, b, tol=tol, nnz=nnz, positive=positive)

    AT = A.T
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = np.zeros(n)
    resid = np.copy(b)
    normb = norm(b)
    selected = np.zeros(n, dtype=np.bool)

    for i in range(nnz):
        if norm(resid) / normb < tol:
            break
        projections = AT.dot(resid)
        projections[selected] = 0.0
        if positive:
            index = np.argmax(projections)
        else:
            index = np.argmax(abs(projections))
        atom = AT[index]
        coef = projections[index] / norm(A[:, index])
        if positive and coef <= 0.0:
            break
        resid -= coef * atom
        x[index] = coef
        selected[index] = True
    return x


MP = MatchingPursuit


def outer_product_cache(X, limit=float('inf')):
    '''cache method for computing and storing outer products
    Args:
      X: matrix of row vectors
      limit: stops storing outer products after cache contains this many elements
    Returns:
      function that computes outer product of row with itself given its index
    '''

    cache = {}

    def outer_product(i):
        output = cache.get(i)
        if output is None:
            output = np.outer(X[i], X[i])
            if len(cache) < limit:
                cache[i] = output
        return output

    return outer_product


def binary_line_search(x, dx, f, nsplit=16):
    '''computes update coefficient using binary line search
    Args:
      x: current position
      dx: full step
      f: objective function
      nsplit: how many binary splits to perform when doing line search
    Returns:
      (coefficient, whether any coefficient was found to improve objective)
    '''

    obj = f(x)
    alpha = 0.0
    failed = True
    increment = True
    while increment:
        alpha += 0.5
        for i in range(2, nsplit + 1):
            step = x + alpha * dx
            objstep = f(step)
            if objstep < obj:
                alpha += 2.0 ** -i
                obj = objstep
                failed = False
            else:
                alpha -= 2.0 ** -i
                increment = False
    return alpha, failed


# NOTE: Hybrid (1st & 2nd Order) Method Based on Boyd & Vandenberghe, ``Chapter 10: Equality-Contrained Minimization," Convex Optimization, 2004.
def SupportingHyperplaneProperty(x, A, niter=None, eps=1.0, nsplit=16):
    '''checks SHP property by solving min_h sum(max{Ch+eps,0}^2) s.t. Sh=0, where C=(A_{supp(x)^C}^T 1) and S=(A_supp(x)^T 1)
    Args:
      x: nonnegative vector of length n
      A: matrix of size (d, n)
      niter: give up after this many iterations; if None sets niter=n
      eps: separation of non-support vertices from support supporting hyperplane
      nsplit: how many binary splits to perform when doing line search
    Returns:
      hyperplane (d+1-dimensional vector, last dimension the negative intercept) supporting columns of A in support of x, if one exists; otherwise False
    '''

    assert not (x < 0).sum(), "signal (x) must be nonnegative"
    assert eps > 0.0, "separation (eps) must be positive"
    d, n = A.shape
    A = np.append(A, np.zeros((d, 1)), axis=1)
    n += 1
    if niter is None:
        niter = d
    if type(x) != np.ndarray:
        x = x.toarray()[0]
    x = np.append(x, 0)
    nz = np.where(x > 0)[0]
    z = np.where(x == 0)[0]
    nnz = nz.shape[0]
    C = np.hstack([A[:, z].T, np.ones((n - nnz, 1))])
    AST = A[:, nz].T
    h, ssr, _, _ = lstsq(AST, np.ones(nnz))
    if ssr:
        return False
    h = np.append(h, -1.0)
    Ch = C.dot(h)
    if all(Ch < 0.0):
        return h
    S = np.hstack([AST, np.ones((nnz, 1))])
    ST = S.T
    correction = (ST.dot(inv(S.dot(ST)).dot(S)) - np.eye(d + 1))
    objective = lambda Chpeps: sum(Chpeps[Chpeps > 0.0] ** 3)
    b = np.zeros(d + 1 + nnz)
    outer_product = outer_product_cache(C, 1E10 / d ** 2)

    for i in range(niter):

        Chpeps = Ch + eps
        v = np.where(Ch > - eps)
        gradient = (3.0 * Chpeps[v] ** 2).dot(C[v])
        if i:
            if i == 1:
                M = np.zeros((d + 1 + nnz, d + 1 + nnz))
                M[d + 1:, :d + 1] = S
                M[:d + 1, d + 1:] = ST
            M[:d + 1, :d + 1] = 6.0 * sum(c * outer_product(j) for c, j in zip(Chpeps[v], v[0]))
            if 1.0 / cond(M) < 1E-16:
                step = correction.dot(gradient)
            else:
                b[:d + 1] = -gradient
                step = solve(M, b, assume_a='sym')[:d + 1]
        else:
            step = correction.dot(gradient)

        Cstep = C.dot(step)
        alpha, failed = binary_line_search(Chpeps, Cstep, objective, nsplit=nsplit)
        if failed:
            break
        h += alpha * step
        Ch = C.dot(h)
        if all(Ch < 0.0):
            return h

    return False

import numpy as np
import time
try:
    import Queue as queue
except ImportError:
    import queue


def argmax(backend, A, axis=0):
    if backend.name == 'ctf':
        return abs(A).to_nparray().argmax(axis=axis)
    elif backend.name == 'numpy':
        return abs(A).argmax(axis=axis)


def n_mode_eigendec(backend, T, n, rank, do_flipsign=True):
    """
    Eigendecomposition of mode-n unfolding of a tensor
    """
    dims = T.ndim
    assert dims <= 24  # curret implementation only use a-z for einsum
    str1 = "".join([chr(ord('a') + j) for j in range(n)]) + "y" + "".join(
        [chr(ord('a') + j) for j in range(n + 1, dims)])
    str2 = "".join([chr(ord('a') + j) for j in range(n)]) + "z" + "".join(
        [chr(ord('a') + j) for j in range(n + 1, dims)])
    str3 = "yz"
    einstr = str1 + "," + str2 + "->" + str3

    Y = backend.einsum(einstr, T, T)
    N = Y.shape[0]
    U, _, _ = backend.svd(Y)
    U = U[:, :rank]

    # flip sign
    if do_flipsign:
        U = flipsign(backend, U)
    return U


def ttmc(backend, T, A, transpose=False):
    """
    Tensor times matrix contractions
    """
    dims = T.ndim
    assert dims <= 24  # curret implementation only use a-z for einsum
    X = T.copy()
    for n in range(dims):
        if transpose:
            str1 = "".join([
                chr(ord('a') + j) for j in range(n)
            ]) + "y" + "".join([chr(ord('a') + j) for j in range(n + 1, dims)])
            str2 = "zy"
            str3 = "".join([
                chr(ord('a') + j) for j in range(n)
            ]) + "z" + "".join([chr(ord('a') + j) for j in range(n + 1, dims)])
        else:
            str1 = "".join([
                chr(ord('a') + j) for j in range(n)
            ]) + "y" + "".join([chr(ord('a') + j) for j in range(n + 1, dims)])
            str2 = "yz"
            str3 = "".join([
                chr(ord('a') + j) for j in range(n)
            ]) + "z" + "".join([chr(ord('a') + j) for j in range(n + 1, dims)])
        einstr = str1 + "," + str2 + "->" + str3
        X = backend.einsum(einstr, X, A[n])
    return X


def flipsign(backend, U):
    """
    Flip sign of factor matrices such that largest magnitude
    element will be positive
    """
    midx = argmax(backend, U, axis=0)
    for i in range(U.shape[1]):
        if U[int(midx[i]), i] < 0:
            U[:, i] = -U[:, i]
    return U


def hosvd(backend, T, rank, compute_core=False):
    """
    higher order svd of tensor T
    """
    A = [None for _ in range(T.ndim)]
    dims = range(T.ndim)
    for d in dims:
        A[d] = n_mode_eigendec(backend, T, d, rank)
    if compute_core:
        core = ttmc(backend, T, A, transpose=False)
        return A, core
    else:
        return A


def get_residual(backend, T, A):
    AAT = [None for _ in range(T.ndim)]
    for i in range(T.ndim):
        AAT[i] = backend.dot(A[i], backend.transpose(A[i]))
    nrm = backend.norm(T - ttmc(backend, T, AAT, transpose=False))
    print('residual norm is: ', nrm)
    return nrm


class Tucker_DTALS_Optimizer():
    def __init__(self, backend, T, A):
        self.backend = backend
        self.T = T
        self.A = A
        self.R = A[0].shape[1]
        self.tucker_rank = []
        for i in range(len(A)):
            self.tucker_rank.append(A[i].shape[1])

    def _einstr_builder(self, M, s, ii):
        nd = M.ndim
        assert nd <= 24  # curret implementation only use a-z for einsum
        str1 = "".join([chr(ord('a') + j) for j in range(nd)])
        str2 = (chr(ord('a') + ii)) + "z"
        str3 = "".join([chr(ord('a') + j) for j in range(ii)]) + "z" + "".join(
            [chr(ord('a') + j) for j in range(ii + 1, nd)])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _solve(self, i, s):
        return n_mode_eigendec(self.backend,
                               s[-1][1],
                               i,
                               rank=self.tucker_rank[i],
                               do_flipsign=True)

    def step(self):
        q = queue.Queue()
        for i in range(len(self.A)):
            q.put(i)
        s = [(list(range(len(self.A))), self.T)]
        while not q.empty():
            i = q.get()
            while i not in s[-1][0]:
                s.pop()
                assert (len(s) >= 1)
            while len(s[-1][0]) != 1:
                M = s[-1][1]
                idx = s[-1][0].index(i)
                ii = len(s[-1][0]) - 1
                if idx == len(s[-1][0]) - 1:
                    ii = len(s[-1][0]) - 2

                einstr = self._einstr_builder(M, s, ii)

                N = self.backend.einsum(einstr, M, self.A[ii])

                ss = s[-1][0][:]
                ss.remove(ii)
                s.append((ss, N))
            self.A[i] = self._solve(i, s)
        return self.A


def Tucker_ALS(backend, T, rank, num_iter=1):

    time_all = 0.

    A = hosvd(backend, T, rank)
    get_residual(backend, T, A)

    optimizer = Tucker_DTALS_Optimizer(backend, T, A)
    normT = backend.norm(T)

    for i in range(num_iter):
        t0 = time.time()
        A = optimizer.step()
        t1 = time.time()
        time_all += t1 - t0

    res = get_residual(backend, T, optimizer.A)
    fitness = 1 - res / normT
    print("Tucker residual is", res, "fitness is: ", fitness)
    print("Tucker decomposition took ", time_all, "seconds overall")

    core = ttmc(backend, T, A, transpose=False)
    return A, core


############### example is as follows ##########################
# import tensorbackends as tbs
# backend = tbs.get('numpy')
# T = backend.tensor(backend.random.rand(9, 9, 9))
# mat = backend.tensor(backend.random.rand(9, 9))

# T_new = backend.einsum('ijk,ia,jb,kc->abc', T, mat, mat, mat)
# T = backend.einsum('abc,ia,jb,kc->ijk', T_new, mat, mat, mat)

# A, core = Tucker_ALS(backend, T, rank=8, num_iter=1)

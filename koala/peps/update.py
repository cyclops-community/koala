import tensorbackends
from tensorbackends.interface import ReducedSVD
import numpy as np
import scipy.linalg as la
from .. import gates


class UpdateOption:
    def __str__(self):
        return '{}({})'.format(
            type(self).__name__,
            ','.join('{}={}'.format(k, v) for k, v in vars(self).items())
        )

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return type(self).__name__


class DirectUpdate(UpdateOption):
    def __init__(self, svd_option=None):
        self.svd_option = svd_option


class QRUpdate(UpdateOption):
    def __init__(self, svd_option=None):
        self.svd_option = svd_option


class LocalGramQRUpdate(UpdateOption):
    def __init__(self, rank=None):
        self.rank = rank


class LocalGramQRSVDUpdate(UpdateOption):
    def __init__(self, rank=None):
        self.rank = rank


class DefaultUpdate(UpdateOption):
    def __init__(self, rank=None):
        self.rank = rank


def apply_single_site_operator(state, operator, position):
    state.grid[position] = state.backend.einsum('ijklxp,xy->ijklyp', state.grid[position], operator)


def apply_local_pair_operator(state, operator, positions, update_option):
    if update_option is None:
        update_option = DefaultUpdate()
    if isinstance(update_option, DefaultUpdate):
        apply_local_pair_operator_qr(state, operator, positions, ReducedSVD(update_option.rank))
    elif isinstance(update_option, DirectUpdate):
        apply_local_pair_operator_direct(state, operator, positions, update_option.svd_option)
    elif isinstance(update_option, QRUpdate):
        apply_local_pair_operator_qr(state, operator, positions, update_option.svd_option)
    elif isinstance(update_option, LocalGramQRUpdate):
        apply_local_pair_operator_local_gram_qr(state, operator, positions, update_option.rank)
    elif isinstance(update_option, LocalGramQRSVDUpdate):
        apply_local_pair_operator_local_gram_qr_svd(state, operator, positions, update_option.rank)
    else:
        raise ValueError(f'unknown update option: {update_option}')


def apply_nonlocal_pair_operator(state, operator, positions, update_option):
    assert len(positions) == 2
    x_pos, y_pos = positions

    path = []
    moved_y_pos = [None, None]
    if y_pos[0] < x_pos[0]:
        path.extend(((i,y_pos[1]),(i+1,y_pos[1])) for i in range(y_pos[0], x_pos[0]-1))
        moved_y_pos[0] = x_pos[0] - 1
    elif y_pos[0] > x_pos[0]:
        path.extend(((i,y_pos[1]),(i-1,y_pos[1])) for i in range(y_pos[0], x_pos[0]+1, -1))
        moved_y_pos[0] = x_pos[0] + 1
    else:
        moved_y_pos[0] = x_pos[0]
    if y_pos[1] < x_pos[1]:
        path.extend(((moved_y_pos[0],j),(moved_y_pos[0],j+1)) for j in range(y_pos[1], x_pos[1]-1))
        moved_y_pos[1] = x_pos[1] - 1
    elif y_pos[1] > x_pos[1]:
        path.extend(((moved_y_pos[0],j),(moved_y_pos[0],j-1)) for j in range(y_pos[1], x_pos[1]+1, -1))
        moved_y_pos[1] = x_pos[1] + 1
    else:
        moved_y_pos[1] = x_pos[1]
    if moved_y_pos[0] != x_pos[0] and moved_y_pos[1] < x_pos[1]:
        new_moved_y_pos = [moved_y_pos[0], moved_y_pos[1]+1]
        path.append((tuple(moved_y_pos),tuple(new_moved_y_pos)))
        moved_y_pos = new_moved_y_pos
    elif moved_y_pos[0] != x_pos[0] and moved_y_pos[1] > x_pos[1]:
        new_moved_y_pos = [moved_y_pos[0], moved_y_pos[1]-1]
        path.append((tuple(moved_y_pos),tuple(new_moved_y_pos)))
        moved_y_pos = new_moved_y_pos
    moved_y_pos = tuple(moved_y_pos)

    for u, v in path:
        swap_local_pair(state, u, v, update_option)
    apply_local_pair_operator(state, operator, [x_pos, moved_y_pos], update_option)
    for u, v in reversed(path):
        swap_local_pair(state, u, v, update_option)


def swap_local_pair(state, x_pos, y_pos, update_option):
    if update_option is None:
        update_option = DefaultUpdate()
    if isinstance(update_option, DefaultUpdate):
        swap_local_pair_qr(state, x_pos, y_pos, ReducedSVD(update_option.rank))
    elif isinstance(update_option, DirectUpdate):
        swap_local_pair_direct(state, x_pos, y_pos, update_option.svd_option)
    elif isinstance(update_option, QRUpdate):
        swap_local_pair_qr(state, x_pos, y_pos, update_option.svd_option)
    elif isinstance(update_option, LocalGramQRUpdate):
        swap_local_pair_local_gram_qr(state, x_pos, y_pos, update_option.rank)
    elif isinstance(update_option, LocalGramQRSVDUpdate):
        swap_local_pair_local_gram_qr_svd(state, x_pos, y_pos, update_option.rank)
    else:
        raise ValueError(f'unknown update option: {update_option}')


def truncate(state, update_option):
    identities = {}
    def apply_identity(x_pos, y_pos):
        dims = state.grid[x_pos].shape[4], state.grid[y_pos].shape[4]
        if dims not in identities:
            identities[dims] = state.backend.astensor(np.einsum('xu,yv->xyuv', np.eye(dims[0]), np.eye(dims[1])))
        apply_local_pair_operator(state, identities[dims], (x_pos, y_pos), update_option)

    for i, j in np.ndindex(*state.shape):
        if i < state.shape[0] - 1:
            apply_identity((i, j), (i+1, j))
        if j < state.shape[1] - 1:
            apply_identity((i, j), (i, j+1))


def apply_local_pair_operator_direct(state, operator, positions, svd_option):
    assert len(positions) == 2
    if svd_option is None:
        svd_option = ReducedSVD()
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    if x_pos[0] < y_pos[0]: # [x y]^T
        prod_subscripts = 'abcdxp,cfghyq,xyuv->abndup,nfghvq'
        scale_u_subscripts = 'absdup,s->absdup'
        scale_v_subscripts = 'sbcdvp,s->sbcdvp'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        prod_subscripts = 'abcdxp,efahyq,xyuv->nbcdup,efnhvq'
        scale_u_subscripts = 'sbcdup,s->sbcdup'
        scale_v_subscripts = 'absdvp,s->absdvp'
    elif x_pos[1] < y_pos[1]: # [x y]
        prod_subscripts = 'abcdxp,efgbyq,xyuv->ancdup,efgnvq'
        scale_u_subscripts = 'ascdup,s->ascdup'
        scale_v_subscripts = 'abcsvp,s->abcsvp'
    elif x_pos[1] > y_pos[1]: # [y x]
        prod_subscripts = 'abcdxp,edghyq,xyuv->abcnup,enghvq'
        scale_u_subscripts = 'abcsup,s->abcsup'
        scale_v_subscripts = 'ascdvp,s->ascdvp'
    else:
        assert False

    u, s, v = state.backend.einsumsvd(prod_subscripts, x, y, operator, option=svd_option)
    s = s ** 0.5
    u = state.backend.einsum(scale_u_subscripts, u, s)
    v = state.backend.einsum(scale_v_subscripts, v, s)
    state.grid[x_pos] = u
    state.grid[y_pos] = v


def apply_local_pair_operator_qr(state, operator, positions, svd_option):
    assert len(positions) == 2
    if svd_option is None:
        svd_option = ReducedSVD()
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    if x_pos[0] < y_pos[0]: # [x y]^T
        split_x_subscripts = 'abcdxp->abdi,icxp'
        split_y_subscripts = 'cfghyq->fghj,jcyq'
        recover_x_subscripts = 'abdi,isup,s->absdup'
        recover_y_subscripts = 'fghj,jsvq,s->sfghvq'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        split_x_subscripts = 'abcdxp->bcdi,iaxp'
        split_y_subscripts = 'efahyq->efhj,jayq'
        recover_x_subscripts = 'bcdi,isup,s->sbcdup'
        recover_y_subscripts = 'efhj,jsvq,s->efshvq'
    elif x_pos[1] < y_pos[1]: # [x y]
        split_x_subscripts = 'abcdxp->acdi,ibxp'
        split_y_subscripts = 'efgbyq->efgj,jbyq'
        recover_x_subscripts = 'acdi,isup,s->ascdup'
        recover_y_subscripts = 'efgj,jsvq,s->efgsvq'
    elif x_pos[1] > y_pos[1]: # [y x]
        split_x_subscripts = 'abcdxp->abci,idxp'
        split_y_subscripts = 'edghyq->eghj,jdyq'
        recover_x_subscripts = 'abci,isup,s->abcsup'
        recover_y_subscripts = 'eghj,jsvq,s->esghvq'
    else:
        assert False

    xq, xr = state.backend.einqr(split_x_subscripts, x)
    yq, yr = state.backend.einqr(split_y_subscripts, y)

    u, s, v = state.backend.einsumsvd('ikxp,jkyq,xyuv->isup,jsvq', xr, yr, operator, option=svd_option)
    s = s ** 0.5
    state.grid[x_pos] = state.backend.einsum(recover_x_subscripts, xq, u, s)
    state.grid[y_pos] = state.backend.einsum(recover_y_subscripts, yq, v, s)


def apply_local_pair_operator_local_gram_qr(state, operator, positions, rank):
    assert len(positions) == 2
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    if x_pos[0] < y_pos[0]: # [x y]^T
        gram_x_subscripts = 'abcdxp,abCdXp->xcXC'
        gram_y_subscripts = 'cfghyq,CfghYq->ycYC'
        xq_subscripts = 'abcdxp,xci->abdpi'
        yq_subscripts = 'cfghyq,ycj->fghqj'
        recover_x_subscripts = 'abdpi,isu,s->absdup'
        recover_y_subscripts = 'fghqj,jsv,s->sfghvq'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        gram_x_subscripts = 'abcdxp,AbcdXp->xaXA'
        gram_y_subscripts = 'efahyq,efAhYq->yaYA'
        xq_subscripts = 'abcdxp,xai->bcdpi'
        yq_subscripts = 'efahyq,yaj->efhqj'
        recover_x_subscripts = 'bcdpi,isu,s->sbcdup'
        recover_y_subscripts = 'efhqj,jsv,s->efshvq'
    elif x_pos[1] < y_pos[1]: # [x y]
        gram_x_subscripts = 'abcdxp,aBcdXp->xbXB'
        gram_y_subscripts = 'efgbyq,efgBYq->ybYB'
        xq_subscripts = 'abcdxp,xbi->acdpi'
        yq_subscripts = 'efgbyq,ybj->efgqj'
        recover_x_subscripts = 'acdpi,isu,s->ascdup'
        recover_y_subscripts = 'efgqj,jsv,s->efgsvq'
    elif x_pos[1] > y_pos[1]: # [y x]
        gram_x_subscripts = 'abcdxp,abcDXp->xdXD'
        gram_y_subscripts = 'edghyq,eDghYq->ydYD'
        xq_subscripts = 'abcdxp,xdi->abcpi'
        yq_subscripts = 'edghyq,ydj->eghqj'
        recover_x_subscripts = 'abcpi,isu,s->abcsup'
        recover_y_subscripts = 'eghqj,jsv,s->esghvq'
    else:
        assert False

    def gram_qr_local(backend, a, gram_a_subscripts, q_subscripts):
        gram_a = backend.einsum(gram_a_subscripts, a.conj(), a)
        d, xi = gram_a.shape[:2]

        # local
        gram_a = gram_a.numpy().reshape(d*xi, d*xi)
        w, v = la.eigh(gram_a, overwrite_a=True)
        s = np.clip(w, 0, None) ** 0.5
        s_pinv = np.divide(1, s, out=np.zeros_like(s), where=s!=0)
        r = np.einsum('j,ij->ji', s, v.conj()).reshape(d*xi, d, xi)
        r_inv = np.einsum('j,ij->ij', s_pinv, v).reshape(d, xi, d*xi)

        r = backend.astensor(r)
        r_inv = backend.astensor(r_inv)
        q = backend.einsum(q_subscripts, a, r_inv)
        return q, r

    xq, xr = gram_qr_local(state.backend, x, gram_x_subscripts, xq_subscripts)
    yq, yr = gram_qr_local(state.backend, y, gram_y_subscripts, yq_subscripts)

    u, s, v = state.backend.einsumsvd('ixk,jyk,xyuv->isu,jsv', xr, yr, operator, option=ReducedSVD(rank))
    s = s ** 0.5
    state.grid[x_pos] = state.backend.einsum(recover_x_subscripts, xq, u, s)
    state.grid[y_pos] = state.backend.einsum(recover_y_subscripts, yq, v, s)


def apply_local_pair_operator_local_gram_qr_svd(state, operator, positions, rank):
    assert len(positions) == 2
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    if x_pos[0] < y_pos[0]: # [x y]^T
        gram_x_subscripts = 'abcdxp,abCdXp->xcXC'
        gram_y_subscripts = 'cfghyq,CfghYq->ycYC'
        xq_subscripts = 'abcdxp,xci->abdpi'
        yq_subscripts = 'cfghyq,ycj->fghqj'
        recover_x_subscripts = 'abcdxp,cxsu->absdup'
        recover_y_subscripts = 'cfghyq,cysv->sfghvq'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        gram_x_subscripts = 'abcdxp,AbcdXp->xaXA'
        gram_y_subscripts = 'efahyq,efAhYq->yaYA'
        xq_subscripts = 'abcdxp,xai->bcdpi'
        yq_subscripts = 'efahyq,yaj->efhqj'
        recover_x_subscripts = 'abcdxp,axsu->sbcdup'
        recover_y_subscripts = 'efahyq,aysv->efshvq'
    elif x_pos[1] < y_pos[1]: # [x y]
        gram_x_subscripts = 'abcdxp,aBcdXp->xbXB'
        gram_y_subscripts = 'efgbyq,efgBYq->ybYB'
        xq_subscripts = 'abcdxp,xbi->acdpi'
        yq_subscripts = 'efgbyq,ybj->efgqj'
        recover_x_subscripts = 'abcdxp,bxsu->ascdup'
        recover_y_subscripts = 'efgbyq,bysv->efgsvq'
    elif x_pos[1] > y_pos[1]: # [y x]
        gram_x_subscripts = 'abcdxp,abcDXp->xdXD'
        gram_y_subscripts = 'edghyq,eDghYq->ydYD'
        xq_subscripts = 'abcdxp,xdi->abcpi'
        yq_subscripts = 'edghyq,ydj->eghqj'
        recover_x_subscripts = 'abcdxp,dxsu->abcsup'
        recover_y_subscripts = 'edghyq,dysv->esghvq'
    else:
        assert False

    numpy_backend = tensorbackends.get('numpy')

    def gram_qr_local(backend, a, gram_a_subscripts, q_subscripts):
        gram_a = backend.einsum(gram_a_subscripts, a.conj(), a)
        d, xi = gram_a.shape[:2]

        # local
        gram_a = gram_a.numpy().reshape(d*xi, d*xi)
        w, v = la.eigh(gram_a, overwrite_a=True)
        s = np.clip(w, 0, None) ** 0.5
        s_pinv = np.divide(1, s, out=np.zeros_like(s), where=s!=0)
        r = np.einsum('j,ij->ji', s, v.conj()).reshape(d*xi, d, xi)
        r_inv = np.einsum('j,ij->ij', s_pinv, v).reshape(d, xi, d*xi)
        return numpy_backend.tensor(r), numpy_backend.tensor(r_inv)

    xr, xr_inv = gram_qr_local(state.backend, x, gram_x_subscripts, xq_subscripts)
    yr, yr_inv = gram_qr_local(state.backend, y, gram_y_subscripts, yq_subscripts)

    operator = numpy_backend.tensor(operator.numpy())
    u, s, v = numpy_backend.einsumsvd('ixk,jyk,xyuv->isu,jsv', xr, yr, operator, option=ReducedSVD(rank))
    s **= 0.5
    u = numpy_backend.einsum('xki,isu,s->kxsu', xr_inv, u, s)
    v = numpy_backend.einsum('ykj,jsv,s->kysv', yr_inv, v, s)

    u = state.backend.astensor(u)
    v = state.backend.astensor(v)
    state.grid[x_pos] = state.backend.einsum(recover_x_subscripts, x, u)
    state.grid[y_pos] = state.backend.einsum(recover_y_subscripts, y, v)


def swap_local_pair_direct(state, x_pos, y_pos, svd_option):
    if svd_option is None:
        svd_option = ReducedSVD()

    if x_pos[0] < y_pos[0]: # [x y]^T
        prod_subscripts = 'abcdxp,cfghyq->absdyq,sfghxp'
        scale_u_subscripts = 'absdyq,s->absdyq'
        scale_v_subscripts = 'sfghxp,s->sfghxp'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        prod_subscripts = 'abcdxp,efahyq->sbcdyq,efshxp'
        scale_u_subscripts = 'sbcdyq,s->sbcdyq'
        scale_v_subscripts = 'efshxp,s->efshxp'
    elif x_pos[1] < y_pos[1]: # [x y]
        prod_subscripts = 'abcdxp,efgbyq->ascdyq,efgsxp'
        scale_u_subscripts = 'ascdyq,s->ascdyq'
        scale_v_subscripts = 'efgsxp,s->efgsxp'
    elif x_pos[1] > y_pos[1]: # [y x]
        prod_subscripts = 'abcdxp,edghyq->abcsyq,esghxp'
        scale_u_subscripts = 'abcsyq,s->abcsyq'
        scale_v_subscripts = 'esghxp,s->esghxp'
    else:
        assert False

    x, y = state.grid[x_pos], state.grid[y_pos]
    u, s, v = state.backend.einsumsvd(prod_subscripts, x, y, option=svd_option)
    s = s ** 0.5
    u = state.backend.einsum(scale_u_subscripts, u, s)
    v = state.backend.einsum(scale_v_subscripts, v, s)
    state.grid[x_pos] = u
    state.grid[y_pos] = v


def swap_local_pair_qr(state, x_pos, y_pos, svd_option):
    if svd_option is None:
        svd_option = ReducedSVD()

    if x_pos[0] < y_pos[0]: # [x y]^T
        split_x_subscripts = 'abcdxp->abdi,icxp'
        split_y_subscripts = 'cfghyq->fghj,jcyq'
        recover_x_subscripts = 'abdi,isyq,s->absdyq'
        recover_y_subscripts = 'fghj,jsxp,s->sfghxp'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        split_x_subscripts = 'abcdxp->bcdi,iaxp'
        split_y_subscripts = 'efahyq->efhj,jayq'
        recover_x_subscripts = 'bcdi,isyq,s->sbcdyq'
        recover_y_subscripts = 'efhj,jsxp,s->efshxp'
    elif x_pos[1] < y_pos[1]: # [x y]
        split_x_subscripts = 'abcdxp->acdi,ibxp'
        split_y_subscripts = 'efgbyq->efgj,jbyq'
        recover_x_subscripts = 'acdi,isyq,s->ascdyq'
        recover_y_subscripts = 'efgj,jsxp,s->efgsxp'
    elif x_pos[1] > y_pos[1]: # [y x]
        split_x_subscripts = 'abcdxp->abci,idxp'
        split_y_subscripts = 'edghyq->eghj,jdyq'
        recover_x_subscripts = 'abci,isyq,s->abcsyq'
        recover_y_subscripts = 'eghj,jsxp,s->esghxp'
    else:
        assert False

    x, y = state.grid[x_pos], state.grid[y_pos]

    xq, xr = state.backend.einqr(split_x_subscripts, x)
    yq, yr = state.backend.einqr(split_y_subscripts, y)

    u, s, v = state.backend.einsumsvd('ikxp,jkyq->isyq,jsxp', xr, yr, option=svd_option)
    s = s ** 0.5
    state.grid[x_pos] = state.backend.einsum(recover_x_subscripts, xq, u, s)
    state.grid[y_pos] = state.backend.einsum(recover_y_subscripts, yq, v, s)


def swap_local_pair_local_gram_qr(state, x_pos, y_pos, rank):
    if x_pos[0] < y_pos[0]: # [x y]^T
        gram_x_subscripts = 'abcdxp,abCdXP->xpcXPC'
        gram_y_subscripts = 'cfghyq,CfghYQ->yqcYQC'
        xq_subscripts = 'abcdxp,xpci->abdi'
        yq_subscripts = 'cfghyq,yqcj->fghj'
        recover_x_subscripts = 'abdi,isyq,s->absdyq'
        recover_y_subscripts = 'fghj,jsxp,s->sfghxp'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        gram_x_subscripts = 'abcdxp,AbcdXP->xpaXPA'
        gram_y_subscripts = 'efahyq,efAhYQ->yqaYQA'
        xq_subscripts = 'abcdxp,xpai->bcdi'
        yq_subscripts = 'efahyq,yqaj->efhj'
        recover_x_subscripts = 'bcdi,isyq,s->sbcdyq'
        recover_y_subscripts = 'efhj,jsxp,s->efshxp'
    elif x_pos[1] < y_pos[1]: # [x y]
        gram_x_subscripts = 'abcdxp,aBcdXP->xpbXPB'
        gram_y_subscripts = 'efgbyq,efgBYQ->yqbYQB'
        xq_subscripts = 'abcdxp,xpbi->acdi'
        yq_subscripts = 'efgbyq,yqbj->efgj'
        recover_x_subscripts = 'acdi,isyq,s->ascdyq'
        recover_y_subscripts = 'efgj,jsxp,s->efgsxp'
    elif x_pos[1] > y_pos[1]: # [y x]
        gram_x_subscripts = 'abcdxp,abcDXP->xpdXPD'
        gram_y_subscripts = 'edghyq,eDghYQ->yqdYQD'
        xq_subscripts = 'abcdxp,xpdi->abci'
        yq_subscripts = 'edghyq,yqdj->eghj'
        recover_x_subscripts = 'abci,isyq,s->abcsyq'
        recover_y_subscripts = 'eghj,jsxp,s->esghxp'
    else:
        assert False

    def gram_qr_local(backend, a, gram_a_subscripts, q_subscripts):
        gram_a = backend.einsum(gram_a_subscripts, a.conj(), a)
        d1, d2, xi = gram_a.shape[:3]

        # local
        gram_a = gram_a.numpy().reshape(d1*d2*xi, d1*d2*xi)
        w, v = la.eigh(gram_a, overwrite_a=True)
        s = np.clip(w, 0, None) ** 0.5
        s_pinv = np.divide(1, s, out=np.zeros_like(s), where=s!=0)
        r = np.einsum('j,ij->ji', s, v.conj()).reshape(d1*d2*xi, d1, d2, xi)
        r_inv = np.einsum('j,ij->ij', s_pinv, v).reshape(d1, d2, xi, d1*d2*xi)

        r = backend.astensor(r)
        r_inv = backend.astensor(r_inv)
        q = backend.einsum(q_subscripts, a, r_inv)
        return q, r

    x, y = state.grid[x_pos], state.grid[y_pos]

    xq, xr = gram_qr_local(state.backend, x, gram_x_subscripts, xq_subscripts)
    yq, yr = gram_qr_local(state.backend, y, gram_y_subscripts, yq_subscripts)

    u, s, v = state.backend.einsumsvd('ixpk,jyqk->isyq,jsxp', xr, yr, option=ReducedSVD(rank))
    s = s ** 0.5
    state.grid[x_pos] = state.backend.einsum(recover_x_subscripts, xq, u, s)
    state.grid[y_pos] = state.backend.einsum(recover_y_subscripts, yq, v, s)


def swap_local_pair_local_gram_qr_svd(state, x_pos, y_pos, rank):
    if x_pos[0] < y_pos[0]: # [x y]^T
        gram_x_subscripts = 'abcdxp,abCdXP->xpcXPC'
        gram_y_subscripts = 'cfghyq,CfghYQ->yqcYQC'
        xq_subscripts = 'abcdxp,xpci->abdi'
        yq_subscripts = 'cfghyq,yqcj->fghj'
        recover_x_subscripts = 'abcdxp,cxpsyq->absdyq'
        recover_y_subscripts = 'cfghyq,cyqsxp->sfghxp'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        gram_x_subscripts = 'abcdxp,AbcdXP->xpaXPA'
        gram_y_subscripts = 'efahyq,efAhYQ->yqaYQA'
        xq_subscripts = 'abcdxp,xpai->bcdi'
        yq_subscripts = 'efahyq,yqaj->efhj'
        recover_x_subscripts = 'abcdxp,axpsyq->sbcdyq'
        recover_y_subscripts = 'efahyq,ayqsxp->efshxp'
    elif x_pos[1] < y_pos[1]: # [x y]
        gram_x_subscripts = 'abcdxp,aBcdXP->xpbXPB'
        gram_y_subscripts = 'efgbyq,efgBYQ->yqbYQB'
        xq_subscripts = 'abcdxp,xpbi->acdi'
        yq_subscripts = 'efgbyq,yqbj->efgj'
        recover_x_subscripts = 'abcdxp,bxpsyq->ascdyq'
        recover_y_subscripts = 'efgbyq,byqsxp->efgsxp'
    elif x_pos[1] > y_pos[1]: # [y x]
        gram_x_subscripts = 'abcdxp,abcDXP->xpdXPD'
        gram_y_subscripts = 'edghyq,eDghYQ->yqdYQD'
        xq_subscripts = 'abcdxp,xpdi->abci'
        yq_subscripts = 'edghyq,yqdj->eghj'
        recover_x_subscripts = 'abcdxp,dxpsyq->abcsyq'
        recover_y_subscripts = 'edghyq,dyqsxp->esghxp'
    else:
        assert False

    numpy_backend = tensorbackends.get('numpy')

    def gram_qr_local(backend, a, gram_a_subscripts, q_subscripts):
        gram_a = backend.einsum(gram_a_subscripts, a.conj(), a)
        d1, d2, xi = gram_a.shape[:3]

        # local
        gram_a = gram_a.numpy().reshape(d1*d2*xi, d1*d2*xi)
        w, v = la.eigh(gram_a, overwrite_a=True)
        s = np.clip(w, 0, None) ** 0.5
        s_pinv = np.divide(1, s, out=np.zeros_like(s), where=s!=0)
        r = np.einsum('j,ij->ji', s, v.conj()).reshape(d1*d2*xi, d1, d2, xi)
        r_inv = np.einsum('j,ij->ij', s_pinv, v).reshape(d1, d2, xi, d1*d2*xi)
        return numpy_backend.tensor(r), numpy_backend.tensor(r_inv)

    x, y = state.grid[x_pos], state.grid[y_pos]

    xr, xr_inv = gram_qr_local(state.backend, x, gram_x_subscripts, xq_subscripts)
    yr, yr_inv = gram_qr_local(state.backend, y, gram_y_subscripts, yq_subscripts)

    u, s, v = numpy_backend.einsumsvd('ixpk,jyqk->isyq,jsxp', xr, yr, option=ReducedSVD(rank))
    s **= 0.5
    u = numpy_backend.einsum('xpki,isyq,s->kxpsyq', xr_inv, u, s)
    v = numpy_backend.einsum('yqkj,jsxp,s->kyqsxp', yr_inv, v, s)

    u = state.backend.astensor(u)
    v = state.backend.astensor(v)
    state.grid[x_pos] = state.backend.einsum(recover_x_subscripts, x, u)
    state.grid[y_pos] = state.backend.einsum(recover_y_subscripts, y, v)

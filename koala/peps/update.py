from tensorbackends.interface import ReducedSVD


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


def apply_single_site_operator(state, operator, position):
    state.grid[position] = state.backend.einsum('ijklxp,xy->ijklyp', state.grid[position], operator)


def apply_local_pair_operator(state, operator, positions, update_option):
    if update_option is None:
        update_option = QRUpdate()
    if isinstance(update_option, DirectUpdate):
        return apply_local_pair_operator_direct(state, operator, positions, update_option.svd_option)
    elif isinstance(update_option, QRUpdate):
        return apply_local_pair_operator_qr(state, operator, positions, update_option.svd_option)
    else:
        raise ValueError(f'unknown update option: {update_option}')


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

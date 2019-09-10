from .peps import PEPS

import numpy as np


class QuantumRegister:
    def __init__(self, nrow, ncol, mapping):
        self.state = PEPS(nrow=nrow, ncol=ncol)
        self.mapping = mapping

    def apply_gate(self, name, parameters, qubits):
        tensor = tensorize(name, parameters)
        postitons = [self.mapping(qubit) for qubit in qubits]
        self.state.apply_operator(tensor, postitons)

    def peak(self, qubits, nsamples):
        self.state.peak(qubtis, nsamples)


# =============================================================================
# Gate tensors
# -----------------------------------------------------------------------------
def tensorize(name, parameters):
    def bad_name(parameters):
        raise ValueError(name)
    return tensorize_dispatch.get(name, bad_name)(parameters)

def add_control(array):
    return np.block([[np.eye(2),np.zeros((2,2))],[np.zeros((2,2)),array]]).reshape(2,2,2,2)

H_TENSOR = (np.array([1,1,1,-1],dtype=complex)/np.sqrt(2)).reshape(2,2)
X_TENSOR = np.array([0,1,1,0],dtype=complex).reshape(2,2)
Y_TENSOR = np.array([0,-1j,1j,0],dtype=complex).reshape(2,2)
Z_TENSOR = np.array([1,0,0,-1],dtype=complex).reshape(2,2)
CH_TENSOR = add_control(H_TENSOR)
CX_TENSOR = add_control(X_TENSOR)
CY_TENSOR = add_control(Y_TENSOR)
CZ_TENSOR = add_control(Z_TENSOR)
SWAP_TENSOR = np.array([1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1],dtype=complex).reshape(2,2,2,2)

tensorize_dispatch = {
    'H': lambda _: H_TENSOR,
    'X': lambda _: X_TENSOR,
    'Y': lambda _: Y_TENSOR,
    'Z': lambda _: Z_TENSOR,
    'R': lambda p: np.array([1,0,0,np.exp(1j*p[0])],dtype=complex).reshape(2,2),
    'CH': lambda _: Z_TENSOR,
    'CX': lambda _: CX_TENSOR,
    'CY': lambda _: CY_TENSOR,
    'CZ': lambda _: CZ_TENSOR,
    'CR': lambda p: add_control(np.array([1,0,0,np.exp(1j*p[0])],dtype=complex).reshape(2,2)),
    'SWAP': lambda _: SWAP_TENSOR,
}
# =============================================================================

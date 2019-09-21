
from ..quantum_register import QuantumRegister
from ..backends import get_backend
from ..gates import tensorize
from .peps import PEPS



class PEPSQuantumRegister(QuantumRegister):
    def __init__(self, nrow, ncol, mapping, backend):
        self.state = PEPS.zeros_state(nrow, ncol)
        self.mapping = mapping

    def apply_gate(self, gate):
        tensor = tensorize(gate.name, *gate.parameters)
        postitons = [self.mapping(qubit) for qubit in gate.qubits]
        self.state.apply_operator(tensor, postitons)

    def peak(self, qubits, nsamples):
        self.state.peak(qubtis, nsamples)

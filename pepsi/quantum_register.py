from . import Peps
from .circuit import Gate, Measure, Peek, Sequential
from .tensor_gate import tensorize


class QuantumRegister:
    def __init__(self, row, col, mapping):
        self.state = Peps(row=row, col=col)
        self.mapping = mapping

    def apply(circuit):
        result = []
        if isinstance(circuit, Gate):
            tensor, qubits = tensorize(circuit)
            # self.state.apply(tensor, *(mapping(qubit) for qubit in qubits))
            raise NotImplementedError()
        elif isinstance(circuit, Measure):
            # put measurement output in 'result'
            raise NotImplementedError()
        elif isinstance(circuit, Peek):
            # put peek output in 'result'
            raise NotImplementedError()
        elif isinstance(circuit, Sequential):
            for subcircuit in circuit:
                result.extend(self.apply(subcircuit))
        else:
            raise TypeError(f'Unkown circuit type {type(circuit).__name__}')
        return result


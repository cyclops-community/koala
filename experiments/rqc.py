from collections import namedtuple

import numpy as np
import numpy.linalg as la

Circuit = namedtuple('Circuit', ['gates', 'nrow', 'ncol', 'nlayer', 'two_qubit_gate_name', 'seed'])
Gate = namedtuple('Gate', ['name', 'parameters', 'qubits'])

def generate(nrow, ncol, nlayer, seed):
    """ Generate a random circuit

    Reference:
        https://www.nature.com/articles/s41586-019-1666-5
    """
    one_qubit_gate_names = ['sqrtX', 'sqrtY', 'sqrtW']
    two_qubit_gate_name = 'ISWAP'
    rand_state = np.random.RandomState(seed)
    as_qubit = lambda i, j: i * ncol + j
    qubits = [*range(nrow*ncol)]
    pairs = [[], [], [], []]
    for i, j in np.ndindex(nrow, ncol):
        if i != nrow - 1:
            pairs[i%2].append((as_qubit(i,j), as_qubit(i+1,j)))
        if j != ncol - 1:
            pairs[j%2+2].append((as_qubit(i,j), as_qubit(i,j+1)))
    gates = []
    previous_gate = [None] * len(qubits)
    def add_one_qubit_gates():
        for i in qubits:
            name = rand_state.choice([n for n in one_qubit_gate_names if n != previous_gate[i]])
            previous_gate[i] = name
            gates.append(Gate(name, [], [i]))
    def add_two_qubit_gates(qubits):
        for i, j in qubits:
            gates.append(Gate(two_qubit_gate_name, [], [i, j]))
    for i in range(nlayer):
        add_one_qubit_gates()
        add_two_qubit_gates(pairs[[0,1,2,3,2,3,0,1][i%8]])
    return Circuit(gates, nrow, ncol, nlayer, two_qubit_gate_name, seed)

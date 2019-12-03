import argparse
from collections import namedtuple

import numpy as np

from pepsi import PEPSQuantumRegister

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


def main(args):
    circuit = generate(args.nrow, args.ncol, args.nlayer, args.seed)
    # TODO run simulator and collect data
    from contextlib import redirect_stdout
    with open(args.output_file, 'w+') as f, redirect_stdout(f):
        for gate in circuit.gates:
            print(gate)


def build_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--nrow', help='the number of rows', type=int)
    parser.add_argument('-c', '--ncol', help='the number of columns', type=int)
    parser.add_argument('-l', '--nlayer', help='the number of layers', type=int)
    parser.add_argument('-s', '--seed', help='random circuit seed', type=int, default=0)

    parser.add_argument('-o', '--output-file', help='output file path')

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())

import argparse, time
from itertools import chain
from statistics import mean
from collections import namedtuple

import numpy as np

import tensorbackends
import koala.statevector
import koala.peps

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


def run_statevector(circuit, backend):
    qstate = koala.statevector.computational_zeros(circuit.nrow*circuit.ncol, backend=backend)
    qstate.apply_circuit(circuit.gates)
    return qstate

def run_peps(circuit, threshold, maxrank, backend):
    qstate = koala.peps.computational_zeros(circuit.nrow, circuit.ncol, backend=backend)
    qstate.apply_circuit(circuit.gates, threshold=threshold, maxrank=maxrank)
    return qstate



def main(args):
    circuit = generate(args.nrow, args.ncol, args.nlayer, args.seed)

    t = time.process_time()
    qstate_true = run_statevector(circuit, backend=args.backend)
    statevector_time = time.process_time() - t

    t = time.process_time()
    qstate_peps = run_peps(circuit, backend=args.backend, threshold=args.threshold, maxrank=args.maxrank)
    peps_time = time.process_time() - t

    t = time.process_time()
    qstate = qstate_peps.statevector()
    qstate /= qstate.norm()
    contraction_time = time.process_time() - t

    t = time.process_time()
    fidelity = np.abs(qstate.inner(qstate_true))**2
    fidelity_time = time.process_time() - t

    backend = tensorbackends.get(args.backend)

    if backend.rank == 0:
        print('circuit.nrow', args.nrow)
        print('circuit.ncol', args.ncol)
        print('circuit.nlayer', args.nlayer)
        print('circuit.seed', args.seed)

        print('backend.name', args.backend)
        print('backend.nproc', backend.nproc)

        print('peps.threshold', args.threshold)
        print('peps.maxrank', args.maxrank)

        print('result.statevector_time', statevector_time)
        print('result.peps_time', peps_time)
        print('result.contraction_time', contraction_time)
        print('result.fidelity_time', fidelity_time)
        print('result.fidelity', fidelity)


def build_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--nrow', help='the number of rows', type=int, default=3)
    parser.add_argument('-c', '--ncol', help='the number of columns', type=int, default=3)
    parser.add_argument('-l', '--nlayer', help='the number of layers', type=int, default=4)
    parser.add_argument('-s', '--seed', help='random circuit seed', type=int, default=0)

    parser.add_argument('-b', '--backend', help='the backend to use', choices=['numpy', 'ctf', 'ctfview'], default='numpy')
    parser.add_argument('-th', '--threshold', help='the threshold in trucated SVD when applying gates', type=float, default=1e-5)
    parser.add_argument('-mr', '--maxrank', help='the maxrank in trucated SVD when applying gates', type=int, default=None)

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())

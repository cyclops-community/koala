import argparse, time
from itertools import chain
from statistics import mean
from collections import namedtuple
from contextlib import redirect_stdout

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
    def add_one_qubit_gates(layer):
        for i in qubits:
            name = rand_state.choice([n for n in one_qubit_gate_names if n != previous_gate[i]])
            previous_gate[i] = name
            layer.append(Gate(name, [], [i]))
    def add_two_qubit_gates(qubits, layer):
        for i, j in qubits:
            layer.append(Gate(two_qubit_gate_name, [], [i, j]))
    for i in range(nlayer):
        layer = []
        add_one_qubit_gates(layer)
        add_two_qubit_gates(pairs[[0,1,2,3,2,3,0,1][i%8]], layer)
        gates.append(layer)
    return Circuit(gates, nrow, ncol, nlayer, two_qubit_gate_name, seed)


def get_average_bond_dim(peps):
    s = 0
    for (i,j), tsr in np.ndenumerate(peps.grid):
        if i > 0: s += tsr.shape[0]
        if j < peps.ncol - 1: s += tsr.shape[1]
        if i < peps.nrow - 1: s += tsr.shape[2]
        if j > 0: s += tsr.shape[3]
    return s / (2 * peps.nrow * peps.ncol - peps.nrow - peps.ncol) / 2

def random_peps(nrow, ncol, rank, backend):
    import numpy as np
    import tensorbackends
    backend = tensorbackends.get(backend)
    grid = np.empty((nrow, ncol), dtype=object)
    for i, j in np.ndindex(nrow, ncol):
        shape = (
            rank if i > 0 else 1,
            rank if j < ncol - 1 else 1,
            rank if i < nrow - 1 else 1,
            rank if j > 0 else 1,
            2,
        )
        grid[i, j] = backend.random.uniform(-1,1,shape) + 1j * backend.random.uniform(-1,1,shape)
    return koala.peps.PEPS(grid, backend)

def run_peps(circuit, threshold, maxrank, backend):
    rank = tensorbackends.get(backend).rank
    qstate = random_peps(circuit.nrow, circuit.ncol, maxrank, backend=backend)
    is_ctf = backend in {'ctf', 'ctfview'}
    if is_ctf:
        import ctf
        ctf.initialize_flops_counter()
    for i, layer in enumerate(circuit.gates):
        if rank == 0: print(f'average_bond_dim_{i}', get_average_bond_dim(qstate), flush=True)
        t = time.process_time()
        qstate.apply_circuit(layer, threshold=threshold, maxrank=maxrank)
        t = time.process_time() - t
        if rank == 0: print(f'layer_time_{i}', t, flush=True)
        if rank == 0 and is_ctf:
            print(f'layer_flops_{i}', ctf.get_estimated_flops(), flush=True)
            ctf.initialize_flops_counter()
    return qstate


def main(args):
    circuit = generate(args.nrow, args.ncol, args.nlayer, args.seed)

    t = time.process_time()
    qstate_peps = run_peps(circuit, backend=args.backend, threshold=args.threshold, maxrank=args.maxrank)
    peps_time = time.process_time() - t

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
        print('result.peps_time', peps_time)


def build_cli_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--nrow', help='the number of rows', type=int, default=3)
    parser.add_argument('-c', '--ncol', help='the number of columns', type=int, default=3)
    parser.add_argument('-l', '--nlayer', help='the number of layers', type=int, default=4)
    parser.add_argument('-s', '--seed', help='random circuit seed', type=int, default=0)

    parser.add_argument('-b', '--backend', help='the backend to use', choices=['numpy', 'ctf', 'ctfview'], default='numpy')
    parser.add_argument('-th', '--threshold', help='the threshold in trucated SVD when applying gates', type=float, default=1e-5)
    parser.add_argument('-mr', '--maxrank', help='the maxrank in trucated SVD when applying gates', type=int, default=2)

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    main(parser.parse_args())

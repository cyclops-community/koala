import unittest

import numpy as np
from tensorbackends.utils import test_with_backend

from pepsi import PEPSQuantumRegister, Observable

from .circuit import Circuit, Gate


@test_with_backend()
class TestPEPS(unittest.TestCase):
    def test_norm(self, backend):
        qreg = PEPSQuantumRegister(2, 3, backend=backend)
        qreg.apply_circuit(Circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ]))
        self.assertTrue(np.isclose(qreg.norm(), 1))
        qreg *= 2
        self.assertTrue(np.isclose(qreg.norm(), 2))
        qreg /= 2j
        self.assertTrue(np.isclose(qreg.norm(), 1))

    def test_amplitude(self, backend):
        qreg = PEPSQuantumRegister(2, 3, backend=backend)
        qreg.apply_circuit(Circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ]))
        self.assertTrue(np.isclose(qreg.amplitude([1,0,0,1,0,0]), 1/np.sqrt(2)))
        self.assertTrue(np.isclose(qreg.amplitude([1,1,0,1,1,0]), 1j/np.sqrt(2)))

    def test_probablity(self, backend):
        qreg = PEPSQuantumRegister(2, 3, backend=backend)
        qreg.apply_circuit(Circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ]))
        self.assertTrue(np.isclose(qreg.probability([1,0,0,1,0,0]), 1/2))
        self.assertTrue(np.isclose(qreg.probability([1,1,0,1,1,0]), 1/2))

    def test_expectation(self, backend):
        qreg = PEPSQuantumRegister(2, 3, backend=backend)
        qreg.apply_circuit(Circuit([
            Gate('X', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [2]),
        ]))
        observable = 1.5 * Observable.sum([
            Observable.Z(0) * 2,
            Observable.Z(1), 
            Observable.Z(2) * 2,
            Observable.Z(3),
        ])
        self.assertTrue(np.isclose(qreg.expectation(observable), -3))

    def test_expectation_use_cache(self, backend):
        qreg = PEPSQuantumRegister(2, 3, backend=backend)
        qreg.apply_circuit(Circuit([
            Gate('X', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [2]),
        ]))
        observable = 1.5 * Observable.sum([
            Observable.Z(0) * 2,
            Observable.Z(1), 
            Observable.Z(2) * 2,
            Observable.Z(3),
        ])
        self.assertTrue(np.isclose(qreg.expectation(observable, use_cache=True), -3))

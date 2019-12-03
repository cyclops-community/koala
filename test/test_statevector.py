import unittest

import numpy as np
from tensorbackends.utils import test_with_backend

from pepsi import Observable, statevector

from .gate import Gate


@test_with_backend()
class TestStateVector(unittest.TestCase):
    def test_norm(self, backend):
        qstate = statevector.computational_zeros(6, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ])
        self.assertTrue(np.isclose(qstate.norm(), 1))
        qstate *= 2
        self.assertTrue(np.isclose(qstate.norm(), 2))
        qstate /= 2j
        self.assertTrue(np.isclose(qstate.norm(), 1))

    def test_amplitude(self, backend):
        qstate = statevector.computational_zeros(6, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ])
        self.assertTrue(np.isclose(qstate.amplitude([1,0,0,1,0,0]), 1/np.sqrt(2)))
        self.assertTrue(np.isclose(qstate.amplitude([1,1,0,1,1,0]), 1j/np.sqrt(2)))

    def test_probablity(self, backend):
        qstate = statevector.computational_zeros(6, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ])
        self.assertTrue(np.isclose(qstate.probability([1,0,0,1,0,0]), 1/2))
        self.assertTrue(np.isclose(qstate.probability([1,1,0,1,1,0]), 1/2))

    def test_expectation(self, backend):
        qstate = statevector.computational_zeros(6, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [2]),
        ])
        observable = 1.5 * Observable.sum([
            Observable.Z(0) * 2,
            Observable.Z(1),
            Observable.Z(2) * 2,
            Observable.Z(3),
        ])
        self.assertTrue(np.isclose(qstate.expectation(observable), -3))

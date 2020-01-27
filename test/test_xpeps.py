import unittest

import numpy as np
from tensorbackends.utils import test_with_backend

from koala import Observable, xpeps, statevector, Gate


@test_with_backend()
class TestXPEPS(unittest.TestCase):
    def test_norm(self, backend):
        qstate = xpeps.computational_zeros(2, 3, backend=backend)
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
        qstate = xpeps.computational_zeros(2, 3, backend=backend)
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
        qstate = xpeps.computational_zeros(2, 3, backend=backend)
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
        qstate = xpeps.computational_zeros(2, 3, backend=backend)
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

    # def test_expectation_use_cache(self, backend):
    #     qstate = xpeps.computational_zeros(2, 3, backend=backend)
    #     qstate.apply_circuit([
    #         Gate('X', [], [0]),
    #         Gate('CX', [], [0,3]),
    #         Gate('H', [], [2]),
    #     ])
    #     observable = 1.5 * Observable.sum([
    #         Observable.Z(0) * 2,
    #         Observable.Z(1), 
    #         Observable.Z(2) * 2,
    #         Observable.Z(3),
    #     ])
    #     self.assertTrue(np.isclose(qstate.expectation(observable, use_cache=True), -3))

    def test_add(self, backend):
        psi = xpeps.computational_zeros(2, 3, backend=backend)
        phi = xpeps.computational_ones(2, 3, backend=backend)
        self.assertTrue(np.isclose((psi + phi).norm(), np.sqrt(2)))

    def test_inner(self, backend):
        psi = xpeps.computational_zeros(2, 3, backend=backend)
        psi.apply_circuit([
            Gate('H', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [3]),
        ])
        phi = xpeps.computational_zeros(2, 3, backend=backend)
        self.assertTrue(np.isclose(psi.inner(phi), 0.5))

    def test_statevector(self, backend):
        psi = xpeps.computational_zeros(2, 3, backend=backend)
        psi.apply_circuit([
            Gate('H', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [3]),
        ])
        psi = psi.statevector()
        phi = statevector.computational_zeros(6, backend=backend)
        self.assertTrue(np.isclose(psi.inner(phi), 0.5))

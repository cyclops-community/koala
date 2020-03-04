import unittest

import numpy as np
from tensorbackends.interface import ImplicitRandomizedSVD, ReducedSVD, RandomizedSVD
from tensorbackends.utils import test_with_backend

from koala import Observable, peps, statevector, Gate
from koala.peps import contract_options, Snake, ABMPS, BMPS, Square, TRG


@test_with_backend()
class TestPEPS(unittest.TestCase):
    def test_norm(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
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
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ])
        self.assertTrue(np.isclose(qstate.amplitude([1,0,0,1,0,0]), 1/np.sqrt(2)))
        self.assertTrue(np.isclose(qstate.amplitude([1,1,0,1,1,0]), 1j/np.sqrt(2)))

    def test_amplitude_approx(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ], svd_option=ImplicitRandomizedSVD(rank=2))
        contract_option = peps.BMPS(ReducedSVD(rank=2))
        self.assertTrue(np.isclose(qstate.amplitude([1,0,0,1,0,0], contract_option), 1/np.sqrt(2)))
        self.assertTrue(np.isclose(qstate.amplitude([1,1,0,1,1,0], contract_option), 1j/np.sqrt(2)))

    def test_probablity(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
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
        qstate = peps.computational_zeros(2, 3, backend=backend)
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

    def test_expectation_use_cache(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
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
        self.assertTrue(np.isclose(qstate.expectation(observable, use_cache=True), -3))

    def test_expectation_use_cache_approx(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [2]),
        ], svd_option=ImplicitRandomizedSVD(rank=2))
        observable = 1.5 * Observable.sum([
            Observable.Z(0) * 2,
            Observable.Z(1), 
            Observable.Z(2) * 2,
            Observable.Z(3),
        ])
        contract_option = peps.BMPS(ReducedSVD(rank=2))
        self.assertTrue(np.isclose(qstate.expectation(observable, use_cache=True, contract_option=contract_option), -3))

    def test_add(self, backend):
        psi = peps.computational_zeros(2, 3, backend=backend)
        phi = peps.computational_ones(2, 3, backend=backend)
        self.assertTrue(np.isclose((psi + phi).norm(), np.sqrt(2)))

    def test_inner(self, backend):
        psi = peps.computational_zeros(2, 3, backend=backend)
        psi.apply_circuit([
            Gate('H', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [3]),
        ])
        phi = peps.computational_zeros(2, 3, backend=backend)
        self.assertTrue(np.isclose(psi.inner(phi), 0.5))

    def test_inner_approx(self, backend):
        psi = peps.computational_zeros(2, 3, backend=backend)
        psi.apply_circuit([
            Gate('H', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [3]),
        ], svd_option=ImplicitRandomizedSVD(rank=2))
        phi = peps.computational_zeros(2, 3, backend=backend)
        contract_option = peps.BMPS(ReducedSVD(rank=2))
        self.assertTrue(np.isclose(psi.inner(phi, contract_option), 0.5))

    def test_statevector(self, backend):
        psi = peps.computational_zeros(2, 3, backend=backend)
        psi.apply_circuit([
            Gate('H', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [3]),
        ])
        psi = psi.statevector()
        phi = statevector.computational_zeros(6, backend=backend)
        self.assertTrue(np.isclose(psi.inner(phi), 0.5))

    def test_contract_scalar(self, backend):
        qstate = peps.random(3, 4, 2, backend)
        norm = qstate.norm(contract_option=Snake())
        for contract_option in contract_options:
            if contract_option not in (Snake, TRG):
                for svd_option in (None, ReducedSVD(16), RandomizedSVD(16), ImplicitRandomizedSVD(16)):
                    with self.subTest(contract_option=contract_option.__name__, svd_option=svd_option):
                        self.assertTrue(np.isclose(norm, qstate.norm(contract_option=contract_option(svd_option))))

    def test_contract_vector(self, backend):
        qstate = peps.random(3, 3, 2, backend)
        statevector = qstate.statevector(contract_option=Snake()).tensor.numpy()
        for contract_option in (BMPS,):
            for svd_option in (None, ReducedSVD(16), RandomizedSVD(16), ImplicitRandomizedSVD(16)):
                with self.subTest(contract_option=contract_option.__name__, svd_option=svd_option):
                    self.assertTrue(np.allclose(statevector, qstate.statevector(
                        contract_option=contract_option(svd_option)).tensor.numpy()))

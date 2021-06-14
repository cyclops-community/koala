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
        self.assertTrue(backend.isclose(qstate.norm(), 1))
        qstate *= 2
        self.assertTrue(backend.isclose(qstate.norm(), 2))
        qstate /= 2j
        self.assertTrue(backend.isclose(qstate.norm(), 1))

    def test_trace(self, backend):
        qstate = peps.identity(3, 3, backend=backend)
        self.assertTrue(backend.isclose(qstate.trace(), 2**qstate.nsite))

    def test_amplitude(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ])
        self.assertTrue(backend.isclose(qstate.amplitude([1,0,0,1,0,0]), 1/np.sqrt(2)))
        self.assertTrue(backend.isclose(qstate.amplitude([1,1,0,1,1,0]), 1j/np.sqrt(2)))

    def test_amplitude_approx(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ], update_option=peps.DirectUpdate(ImplicitRandomizedSVD(rank=2)))
        contract_option = peps.BMPS(ReducedSVD(rank=2))
        self.assertTrue(backend.isclose(qstate.amplitude([1,0,0,1,0,0], contract_option), 1/np.sqrt(2)))
        self.assertTrue(backend.isclose(qstate.amplitude([1,1,0,1,1,0], contract_option), 1j/np.sqrt(2)))

    def test_amplitude_qr_update(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ], update_option=peps.QRUpdate(rank=2))
        contract_option = peps.BMPS(ReducedSVD(rank=2))
        self.assertTrue(backend.isclose(qstate.amplitude([1,0,0,1,0,0], contract_option), 1/np.sqrt(2)))
        self.assertTrue(backend.isclose(qstate.amplitude([1,1,0,1,1,0], contract_option), 1j/np.sqrt(2)))

    def test_amplitude_local_gram_qr_update(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ], update_option=peps.LocalGramQRUpdate(rank=2))
        contract_option = peps.BMPS(ReducedSVD(rank=2))
        self.assertTrue(backend.isclose(qstate.amplitude([1,0,0,1,0,0], contract_option), 1/np.sqrt(2)))
        self.assertTrue(backend.isclose(qstate.amplitude([1,1,0,1,1,0], contract_option), 1j/np.sqrt(2)))

    def test_amplitude_local_gram_qr_svd_update(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ], update_option=peps.LocalGramQRSVDUpdate(rank=2))
        contract_option = peps.BMPS(ReducedSVD(rank=2))
        self.assertTrue(backend.isclose(qstate.amplitude([1,0,0,1,0,0], contract_option), 1/np.sqrt(2)))
        self.assertTrue(backend.isclose(qstate.amplitude([1,1,0,1,1,0], contract_option), 1j/np.sqrt(2)))

    def test_amplitude_nonlocal(self, backend):
        update_options = [
            None,
            peps.DirectUpdate(ImplicitRandomizedSVD(rank=2)),
            peps.QRUpdate(rank=2),
            peps.LocalGramQRUpdate(rank=2),
            peps.LocalGramQRSVDUpdate(rank=2),
        ]
        for option in update_options:
            with self.subTest(update_option=option):
                qstate = peps.computational_zeros(2, 3, backend=backend)
                qstate.apply_circuit([
                    Gate('X', [], [0]),
                    Gate('H', [], [1]),
                    Gate('CX', [], [0,5]),
                    Gate('CX', [], [1,3]),
                    Gate('S', [], [1]),
                ], update_option=option)
                self.assertTrue(backend.isclose(qstate.amplitude([1,0,0,0,0,1]), 1/np.sqrt(2)))
                self.assertTrue(backend.isclose(qstate.amplitude([1,1,0,1,0,1]), 1j/np.sqrt(2)))

    def test_amplitude_flip(self, backend):
        update_options = [
            None,
            peps.DirectUpdate(ImplicitRandomizedSVD(rank=2)),
            peps.QRUpdate(rank=2),
            peps.LocalGramQRUpdate(rank=2),
            peps.LocalGramQRSVDUpdate(rank=2),
        ]
        for option in update_options:
            with self.subTest(update_option=option):
                qstate = peps.computational_zeros(2, 3, backend=backend).flip()
                qstate.apply_circuit([
                    Gate('X', [], [0]),
                    Gate('H', [], [1]),
                    Gate('CX', [], [0,5]),
                    Gate('CX', [], [1,3]),
                    Gate('S', [], [1]),
                ], update_option=option, flip=True)
                qstate = qstate.flip()
                self.assertTrue(backend.isclose(qstate.amplitude([1,0,0,0,0,1]), 1/np.sqrt(2)))
                self.assertTrue(backend.isclose(qstate.amplitude([1,1,0,1,0,1]), 1j/np.sqrt(2)))

    def test_probablity(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('H', [], [1]),
            Gate('CX', [], [0,3]),
            Gate('CX', [], [1,4]),
            Gate('S', [], [1]),
        ])
        self.assertTrue(backend.isclose(qstate.probability([1,0,0,1,0,0]), 1/2))
        self.assertTrue(backend.isclose(qstate.probability([1,1,0,1,1,0]), 1/2))

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
        self.assertTrue(backend.isclose(qstate.expectation(observable), -3))

    def test_expectation_single_layer(self, backend):
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
        contract_option = peps.SingleLayer(ImplicitRandomizedSVD(rank=2))
        self.assertTrue(backend.isclose(qstate.expectation(observable, contract_option=contract_option), -3))

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
        self.assertTrue(backend.isclose(qstate.expectation(observable, use_cache=True), -3))
        cache = peps.make_environment_cache(qstate, qstate)
        self.assertTrue(backend.isclose(qstate.expectation(observable, use_cache=cache), -3))
        self.assertTrue(backend.isclose(qstate.norm(cache=cache), 1))

    def test_expectation_use_cache_approx(self, backend):
        qstate = peps.computational_zeros(2, 3, backend=backend)
        qstate.apply_circuit([
            Gate('X', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [2]),
        ], update_option=peps.DirectUpdate(ImplicitRandomizedSVD(rank=2)))
        observable = 1.5 * Observable.sum([
            Observable.Z(0) * 2,
            Observable.Z(1),
            Observable.Z(2) * 2,
            Observable.Z(3),
        ])
        contract_option = peps.BMPS(ReducedSVD(rank=2))
        self.assertTrue(backend.isclose(qstate.expectation(observable, use_cache=True, contract_option=contract_option), -3))

    def test_add(self, backend):
        psi = peps.computational_zeros(2, 3, backend=backend)
        phi = peps.computational_ones(2, 3, backend=backend)
        self.assertTrue(backend.isclose((psi + phi).norm(), np.sqrt(2)))

    def test_inner(self, backend):
        psi = peps.computational_zeros(2, 3, backend=backend)
        psi.apply_circuit([
            Gate('H', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [3]),
        ])
        phi = peps.computational_zeros(2, 3, backend=backend)
        self.assertTrue(backend.isclose(psi.inner(phi), 0.5))

    def test_inner_approx(self, backend):
        psi = peps.computational_zeros(2, 3, backend=backend)
        psi.apply_circuit([
            Gate('H', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [3]),
        ], update_option=peps.DirectUpdate(ImplicitRandomizedSVD(rank=2)))
        phi = peps.computational_zeros(2, 3, backend=backend)
        contract_option = peps.BMPS(ReducedSVD(rank=2))
        self.assertTrue(backend.isclose(psi.inner(phi, contract_option), 0.5))

    def test_statevector(self, backend):
        psi = peps.computational_zeros(2, 3, backend=backend)
        psi.apply_circuit([
            Gate('H', [], [0]),
            Gate('CX', [], [0,3]),
            Gate('H', [], [3]),
        ])
        psi = psi.statevector()
        phi = statevector.computational_zeros(6, backend=backend)
        self.assertTrue(backend.isclose(psi.inner(phi), 0.5))

    def test_contract_scalar(self, backend):
        qstate = peps.random(3, 4, 2, backend=backend)
        norm = qstate.norm(contract_option=Snake())
        for contract_option in contract_options:
            if contract_option is not Snake:
                for svd_option in (None, ReducedSVD(16), RandomizedSVD(16), ImplicitRandomizedSVD(16), ImplicitRandomizedSVD(16, orth_method='local_gram')):
                    with self.subTest(contract_option=contract_option.__name__, svd_option=svd_option):
                        self.assertTrue(backend.isclose(norm, qstate.norm(contract_option=contract_option(svd_option))))

    def test_contract_vector(self, backend):
        qstate = peps.random(3, 3, 2, backend=backend)
        statevector = qstate.statevector(contract_option=Snake())
        for contract_option in [BMPS(None), BMPS(ReducedSVD(16)), BMPS(RandomizedSVD(16)), BMPS(ImplicitRandomizedSVD(16))]:
            with self.subTest(contract_option=contract_option):
                contract_result = qstate.statevector(contract_option=contract_option)
                self.assertTrue(backend.allclose(statevector.tensor, contract_result.tensor))

    def test_truncate(self, backend):
        for phys, dual in [(1,1), (2,1), (2,2)]:
            with self.subTest(phyiscal_dim=phys, dual_dim=dual):
                qstate = peps.random(2, 3, 4, phys, dual, backend=backend)
                self.assertEqual(qstate.get_average_bond_dim(), 4)
                qstate.truncate(peps.DefaultUpdate(rank=2))
                self.assertEqual(qstate.get_average_bond_dim(), 2)

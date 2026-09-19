"""Tests for the classical reference: energy conventions and agreement.

The core scientific assertions of the whole project live here:
  * exact diagonalization of the qubit Hamiltonian matches the FCI value
    (both are exact in this 2e- basis, so this is a strong internal check),
  * E_total = E_electronic + E_nuclear-repulsion is respected everywhere,
  * the VQE estimate respects the variational bound vs the exact eigenvalue.
"""

import numpy as np
import pytest

from src.classical_reference import (
    ReferenceResult,
    build_reference,
    exact_diagonalization,
    fci_reference,
)
from src.hamiltonian import build_qubit_operator
from src.molecule_builder import build_hydrogen
from src.quantum_solver import run_vqe


@pytest.fixture(scope="module")
def problem():
    return build_qubit_operator(build_hydrogen(0.735), mapping="parity")


@pytest.fixture(scope="module")
def ref(problem):
    return build_reference(problem.geometry, qubit_op=problem.qubit_op)


class TestExactReferencesAgree:
    def test_exact_eigenvalue_matches_fci(self, problem, ref):
        # Both are exact limits of this 2-electron problem in STO-3G; the
        # tiny difference is only numerical conditioning of the two routes.
        assert ref.E_exact_elec == pytest.approx(ref.E_fci_elec, abs=1e-6)

    def test_electronic_energy_negative(self, ref):
        assert ref.E_fci_elec < 0.0
        assert ref.E_hf_elec < 0.0

    def test_hf_above_fci(self, ref):
        # Variational hierarchy: HF >= FCI (FCI is the exact limit).
        assert ref.E_hf_elec >= ref.E_fci_elec - 1e-8


class TestEnergyConventions:
    def test_total_is_electronic_plus_nuclear(self, ref):
        assert ref.E_fci_total == pytest.approx(
            ref.E_fci_elec + ref.E_nuc, rel=1e-12
        )
        assert ref.E_exact_total == pytest.approx(
            ref.E_exact_elec + ref.E_nuc, rel=1e-12
        )

    def test_nuclear_repulsion_positive(self, ref):
        assert ref.E_nuc > 0.0

    def test_reference_result_is_documented(self, ref):
        assert isinstance(ref, ReferenceResult)
        assert ref.R_angstrom == pytest.approx(0.735)
        assert ref.basis == "sto3g"

    def test_exact_diagonalization_without_op_returns_none(self):
        r2 = build_reference(build_hydrogen(1.0))
        assert r2.E_exact_elec is None


class TestVQEVersusClassical:
    def test_vqe_respects_variational_bound(self, problem, ref):
        vqe = run_vqe(
            problem.qubit_op,
            num_particles=problem.num_particles,
            nuclear_repulsion_energy=problem.nuclear_repulsion_energy,
            maxiter=200,
        )
        # VQE can never go below the exact ground-state eigenvalue.
        assert vqe.electronic_energy >= ref.E_exact_elec - 1e-6

    def test_vqe_lands_on_classical_value(self, problem, ref):
        vqe = run_vqe(
            problem.qubit_op,
            num_particles=problem.num_particles,
            nuclear_repulsion_energy=problem.nuclear_repulsion_energy,
            maxiter=200,
        )
        assert vqe.electronic_energy == pytest.approx(ref.E_exact_elec, abs=1e-4)


class TestFCIReferenceHelper:
    def test_fci_reference_returns_triplet(self):
        e_fci, e_hf, e_nuc = fci_reference(build_hydrogen(1.0))
        assert len({e_fci, e_hf, e_nuc}) == 3
        assert np.isfinite(e_fci) and np.isfinite(e_hf) and np.isfinite(e_nuc)
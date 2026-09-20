"""Tests for the VQE quantum solver: validity, history, reproducibility."""

import numpy as np
import pytest

from src.classical_reference import exact_diagonalization
from src.hamiltonian import build_qubit_operator
from src.molecule_builder import build_hydrogen
from src.quantum_solver import (
    VQEResult,
    make_ansatz,
    make_estimator,
    make_optimizer,
    run_vqe,
)


@pytest.fixture(scope="module")
def h2_problem():
    geom = build_hydrogen(0.735)
    return build_qubit_operator(geom, mapping="parity")


@pytest.fixture(scope="module")
def vqe_result(h2_problem):
    return run_vqe(
        h2_problem.qubit_op,
        num_particles=h2_problem.num_particles,
        nuclear_repulsion_energy=h2_problem.nuclear_repulsion_energy,
        ansatz="ry",
        optimizer="SLSQP",
        maxiter=200,
        backend="statevector",
    )


class TestVQEResultShape:
    def test_returns_vqe_result(self, vqe_result):
        assert isinstance(vqe_result, VQEResult)

    def test_energy_is_finite_and_negative(self, vqe_result):
        assert np.isfinite(vqe_result.electronic_energy)
        assert vqe_result.electronic_energy < 0.0

    def test_total_is_electronic_plus_nuclear(self, vqe_result):
        expected = (
            vqe_result.electronic_energy + vqe_result.nuclear_repulsion_energy
        )
        assert vqe_result.total_energy == pytest.approx(
            float(expected), rel=1e-12
        )

    def test_parameter_consistency(self, vqe_result, h2_problem):
        assert vqe_result.num_qubits == h2_problem.num_qubits
        assert vqe_result.num_circuit_parameters == len(
            vqe_result.parameter_values
        )
        assert len(vqe_result.optimal_parameters) == len(
            vqe_result.parameter_values
        )


class TestVariationalQuality:
    def test_within_tolerance_of_exact_eigenvalue(self, vqe_result, h2_problem):
        exact = exact_diagonalization(h2_problem.qubit_op)
        gap = vqe_result.electronic_energy - exact
        # Variational principle: VQE must be >= exact eigenvalue, and for the
        # 2-qubit H2 ansatz it must be essentially exact (to optimizer tol).
        assert gap >= -1e-6
        assert gap < 1e-3


class TestHistory:
    def test_history_is_recorded(self, vqe_result):
        assert len(vqe_result.history) > 0

    def test_history_entries_have_expected_keys(self, vqe_result):
        keys = {"iter", "energy"}
        for entry in vqe_result.history:
            assert keys.issubset(entry.keys())

    def test_history_energies_monotonically_approach_optimum(self, vqe_result):
        energies = [entry["energy"] for entry in vqe_result.history]
        last = energies[-1]
        # energy should be descending toward the minimum (allow tiny noise)
        assert all(e + 1e-8 >= last for e in energies)


class TestReproducibility:
    def test_same_seed_same_result(self, h2_problem):
        a = run_vqe(
            h2_problem.qubit_op,
            num_particles=h2_problem.num_particles,
            nuclear_repulsion_energy=h2_problem.nuclear_repulsion_energy,
            maxiter=80,
            seed=7,
        )
        b = run_vqe(
            h2_problem.qubit_op,
            num_particles=h2_problem.num_particles,
            nuclear_repulsion_energy=h2_problem.nuclear_repulsion_energy,
            maxiter=80,
            seed=7,
        )
        assert a.electronic_energy == pytest.approx(b.electronic_energy)


class TestFactories:
    def test_make_estimator_bad_backend(self):
        with pytest.raises(ValueError):
            make_estimator("nope")

    def test_make_optimizer_bad_name(self):
        with pytest.raises(ValueError):
            make_optimizer("downhill")

    def test_make_ansatz_bad_name(self):
        with pytest.raises(ValueError):
            make_ansatz(2, ansatz="circuit-of-magic")

    def test_make_ansatz_ry_free_parameters(self):
        ans = make_ansatz(2, ansatz="ry", seed=1)
        assert len(ans.parameters) > 0

    def test_uccsd_requires_problem_context(self, h2_problem):
        with pytest.raises(ValueError):
            run_vqe(
                h2_problem.qubit_op,
                num_particles=h2_problem.num_particles,
                nuclear_repulsion_energy=h2_problem.nuclear_repulsion_energy,
                ansatz="uccsd",  # missing qubit_mapper / orbitals
            )


class TestAerBackendSmoke:
    @pytest.mark.xfail(
        reason=(
            "Aer 0.15.0 ships only the deprecated V1 Estimator, whose run() "
            "signature is incompatible with qiskit-algorithms VQE on this "
            "machine (TypeError: missing 'observables'). The app therefore "
            "offers only the exact statevector backend. Re-enable this test "
            "when a compatible Aer V2 Estimator path is available."
        ),
        strict=False,
    )
    def test_aer_backend_runs(self, h2_problem):
        """Aer estimator with reduced cost: still a genuine VQE loop."""
        res = run_vqe(
            h2_problem.qubit_op,
            num_particles=h2_problem.num_particles,
            nuclear_repulsion_energy=h2_problem.nuclear_repulsion_energy,
            maxiter=25,
            backend="aer",
            seed=42,
        )
        assert isinstance(res, VQEResult)
        assert np.isfinite(res.electronic_energy)
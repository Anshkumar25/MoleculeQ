"""Variational Quantum Eigensolver (VQE) for MoleculeQ.

This module wraps Qiskit's ``VQE`` (from ``qiskit-algorithms``) with a
small, teaching-friendly interface. It drives the hybrid quantum-classical
loop:

1. prepare a parameterized ansatz circuit,
2. estimate  <psi(theta)| H |psi(theta)>  through a simulator estimator,
3. ask a classical optimizer for better parameters,
4. repeat until the optimizer converges / a budget is exhausted.

Design notes
------------
- Default backend is an exact **statevector estimator** (deterministic,
  fast, and still a genuine quantum-mechanical simulation — the full
  exponential wavefunction is evolved in memory). A shot-noise Aer
  backend is available to demonstrate sampling error.
- Optimizer choice is exposed (SLSQP / COBYLA). SLSQP is the default for
  the interactive app: deterministic, bounded, robust for the 2-4
  parameter H2 ansatz.
- The variational principle means the returned energy is *>=* the exact
  ground-state eigenvalue of the qubit Hamiltonian; how close depends on
  the ansatz, optimizer, initialization and numerical precision. That
  caveat is surfaced to the user in the documentation, not hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SEED = 37271


@dataclass
class VQEResult:
    """Outcome of a VQE run, designed to serialize cleanly for the UI."""

    converged: bool
    electronic_energy: float
    total_energy: float
    nuclear_repulsion_energy: float
    optimal_parameters: dict
    parameter_values: list[float]
    num_evaluations: int
    cost_function_calls: int
    optimizer_name: str
    ansatz_name: str
    num_qubits: int
    circuit_depth: int
    num_circuit_parameters: int
    history: list[dict] = field(default_factory=list)
    exact_ground_state: Optional[float] = None
    error_note: str = ""


# ---------------------------------------------------------------------------
# Backend / estimator selection
# ---------------------------------------------------------------------------
def make_estimator(backend: str = "statevector", seed: int | None = DEFAULT_SEED):
    """Create a VQE-compatible estimator primitive.

    Parameters
    ----------
    backend : str
        ``"statevector"`` (default, exact & deterministic) or ``"aer-sampler"``
        (Aer Estimator with shot noise).
    seed : int | None
        Random seed for the Ansatz initial-point generator etc.
    """
    if backend == "statevector":
        from qiskit.primitives import StatevectorEstimator

        return StatevectorEstimator()
    if backend == "aer":
        from qiskit_aer.primitives import Estimator as AerEstimator

        options = {"seed_simulator": seed} if seed is not None else {}
        return AerEstimator(run_options=options, approximation=True)
    raise ValueError(
        f"Unknown backend {backend!r}; expected 'statevector' or 'aer'."
    )


def make_ansatz(
    num_qubits: int,
    num_spatial_orbitals: int | None = None,
    num_particles: tuple[int, int] | None = None,
    qubit_mapper=None,
    ansatz: str = "ry",
    seed: int | None = DEFAULT_SEED,
):
    """Construct a parameterized ansatz circuit.

    Parameters
    ----------
    num_qubits : int
        Qubit count of the mapped problem.
    ansatz : str
        ``"ry"``  -> hardware-efficient TwoLocal(ry/rz + cnot)
        ``"uccsd"`` -> unitary coupled-cluster singles & doubles
    num_spatial_orbitals / num_particles / qubit_mapper :
        Used by UCCSD (from the molecular problem).
    seed : int
        Seeded to make circuit / initial-point generation reproducible.
    """
    if ansatz == "ry":
        # Qiskit 2.1+ exposes n_local(); fall back to TwoLocal on older
        # versions where the class form is still canonical.
        try:
            from qiskit.circuit.library import n_local

            return n_local(
                num_qubits,
                rotation_blocks=["ry", "rz"],
                entanglement_blocks="cx",
                entanglement="linear",
                reps=1,
            )
        except (ImportError, TypeError):  # pragma: no cover - older qiskit
            from qiskit.circuit.library import TwoLocal

            return TwoLocal(
                num_qubits,
                rotation_blocks=["ry", "rz"],
                entanglement_blocks="cx",
                entanglement="linear",
                reps=1,
                skip_final_rotation_layer=False,
            )

    if ansatz == "uccsd":
        if num_spatial_orbitals is None or num_particles is None or qubit_mapper is None:
            raise ValueError(
                "UCCSD requires num_spatial_orbitals, num_particles and a "
                "qubit mapper (from the molecular problem)."
            )
        from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

        initial = HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=qubit_mapper,
        )
        return UCCSD(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=qubit_mapper,
            initial_state=initial,
            preserve_spin=True,
        )

    raise ValueError(f"Unknown ansatz {ansatz!r}; expected 'ry' or 'uccsd'.")


def make_optimizer(name: str = "SLSQP", maxiter: int = 300, **kwargs):
    """Return a classical optimizer instance from qiskit-algorithms."""
    from qiskit_algorithms.optimizers import COBYLA, SLSQP

    opts = dict(kwargs)
    if name.upper() == "SLSQP":
        return SLSQP(maxiter=maxiter, tol=opts.get("tol", 1e-8))
    if name.upper() == "COBYLA":
        return COBYLA(maxiter=maxiter, tol=opts.get("tol", 1e-6))
    raise ValueError(f"Unknown optimizer {name!r}; expected 'SLSQP' or 'COBYLA'.")


# ---------------------------------------------------------------------------
# VQE runner
# ---------------------------------------------------------------------------
def run_vqe(
    qubit_op,
    *,
    num_particles: tuple[int, int],
    nuclear_repulsion_energy: float,
    ansatz: str = "ry",
    optimizer: str = "SLSQP",
    maxiter: int = 300,
    initial_point: list[float] | np.ndarray | None = None,
    backend: str = "statevector",
    seed: int | None = DEFAULT_SEED,
    qubit_mapper=None,
    num_spatial_orbitals: int | None = None,
) -> VQEResult:
    """Run VQE on the provided qubit operator.

    Parameters
    ----------
    qubit_op : SparsePauliOp
        Mapped molecular qubit Hamiltonian.
    num_particles : tuple[int, int]
        (n_alpha, n_beta) used by nature ansätze and tapers.
    nuclear_repulsion_energy : float
        Constant added to get total molecular energy (documented, never
        silently folded in).
    ansatz : str
        ``"ry"`` or ``"uccsd"``.
    optimizer : str
        ``"SLSQP"`` (default) or ``"COBYLA"``.
    maxiter : int
        Optimizer budget.
    initial_point : array-like | None
        Starting variational parameters (defaults to a small random set).
    backend : str
        ``"statevector"`` (default) or ``"aer"``.
    seed : int | None
        Reproducibility seed.

    Returns
    -------
    VQEResult
    """
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SLSQP

    estimator = make_estimator(backend, seed=seed)
    ans = make_ansatz(
        num_qubits=qubit_op.num_qubits,
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=qubit_mapper,
        ansatz=ansatz,
        seed=seed,
    )
    opt = make_optimizer(optimizer, maxiter=maxiter)
    if not ans.parameters:
        raise ValueError("Ansatz has no free parameters; VQE is undefined.")

    rng = np.random.default_rng(seed)
    if initial_point is None:
        initial_point = 0.01 * rng.standard_normal(len(ans.parameters))
    initial_point = list(np.asarray(initial_point, dtype=float))

    history: list[dict] = []

    def callback(eval_count, parameters, value, metadata):
        # qiskit-algorithms 0.4 calls with (count, params, value, metadata);
        # older versions passed a std value. Handle both.
        if isinstance(metadata, dict):
            std = metadata.get("variance", metadata.get("std", None))
            if std is not None:
                try:
                    std = float(np.sqrt(std))
                except Exception:
                    std = None
        else:
            std = metadata
        history.append(
            {
                "iter": eval_count,
                "energy": float(value.real if hasattr(value, "real") else value),
                "std": float(std) if std is not None else None,
            }
        )

    vqe = VQE(
        estimator,
        ansatz=ans,
        optimizer=opt,
        initial_point=initial_point,
        callback=callback,
    )

    try:
        result = vqe.compute_minimum_eigenvalue(qubit_op)
        eigenvalue = float(result.eigenvalue.real)
        converged = bool(getattr(result, "converged", True))
    except Exception as exc:  # noqa: BLE001 - surface friendly message
        raise RuntimeError(
            f"VQE failed: {exc}. This can happen if the optimizer diverges or "
            "the backend is misconfigured."
        ) from exc

    opt_params = dict(result.optimal_parameters)
    e_elec = eigenvalue
    e_nuc = float(nuclear_repulsion_energy)
    return VQEResult(
        converged=converged,
        electronic_energy=e_elec,
        total_energy=e_elec + e_nuc,
        nuclear_repulsion_energy=e_nuc,
        optimal_parameters=opt_params,
        parameter_values=[float(v) for v in result.optimal_point],
        num_evaluations=int(getattr(result, "cost_function_evals", len(history))),
        cost_function_calls=len(history),
        optimizer_name=optimizer.upper(),
        ansatz_name=ansatz,
        num_qubits=qubit_op.num_qubits,
        circuit_depth=ans.depth(),
        num_circuit_parameters=len(ans.parameters),
        history=history,
        exact_ground_state=None,
    )
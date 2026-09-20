"""Molecular Hamiltonian construction and fermion-to-qubit mapping.

This module builds the qubit Hamiltonian (Pauli operator) that VQE later
minimises, following the canonical Qiskit Nature workflow:

1.  Solve RHF to obtain the molecular-orbital basis.
2.  Transform one- and two-electron integrals into the MO basis.
3.  Build the *ElectronicStructureProblem*/Hamiltonian (`ElectronicEnergy`)
    with Qiskit Nature's second-quantized operators.
4.  Map the fermionic problem to qubits with the **Parity mapping**
    (optionally with the two-qubit reduction / Z2 tapering).

Chemistry driver policy
-----------------------
The default chemistry backend is *PySCFDriver*.  PySCF ships no Windows
wheels, so on Windows MoleculeQ transparently falls back to its bundled,
dependency-free STO-3G H2 integral engine (:mod:`moleculeq.integrals_h2`).
Both backends produce *equivalent* electronic-structure data for STO-3G H2;
only the integral evaluation layer differs. The active backend is always
reported so the app is honest about what computed the numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import integrals_h2 as integrals
from .molecule_builder import MolecularGeometry
from .hartree_fock import solve_rhf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
def pyscf_available() -> bool:
    """Whether the PySCF driver backend is importable on this machine."""
    try:
        import pyscf  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


ACTIVE_CHEMISTRY_BACKEND = "pyscf" if pyscf_available() else "local-sto3g-integrals"


# ---------------------------------------------------------------------------
# Public data bundle
# ---------------------------------------------------------------------------
@dataclass
class QuibitProblem:
    """Everything downstream needs about a mapped molecular problem."""

    qubit_op: object  # qiskit SparsePauliOp
    num_qubits: int
    fermionic_op: object  # FermionicOp (second-quantized)
    num_particles: tuple[int, int]
    nuclear_repulsion_energy: float
    hf_reference_energy_elec: float
    hf_reference_energy_total: float
    basis: str
    mapping: str
    chemistry_backend: str
    geometry: MolecularGeometry


def _build_qubit_operator_nature(
    geometry: MolecularGeometry, basis: str = "sto3g", mapping: str = "parity", **kwargs
) -> QuibitProblem:
    """Qubit operator via Qiskit Nature's ElectronicEnergy + Parity mapping."""
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper

    rhf = solve_rhf(geometry)
    spatial_h1 = rhf["C"].T @ rhf["h_core"] @ rhf["C"]
    G_phys_spatial = np.einsum(
        "pqrs,pi,qk,rj,sl->ijkl",
        rhf["g2_ao"],
        rhf["C"],
        rhf["C"],
        rhf["C"],
        rhf["C"],
    )

    ee = ElectronicEnergy.from_raw_integrals(
        h1_a=spatial_h1, h2_aa=G_phys_spatial, validate=False
    )
    # keep the nuclear repulsion as a documented additive constant
    ee.nuclear_repulsion_energy = rhf["E_nuc"]
    fermionic_op = ee.second_q_op()

    n_alpha, n_beta = 1, 1  # H2 singlet
    if mapping.lower() == "parity":
        mapper = ParityMapper(num_particles=(n_alpha, n_beta))
    elif mapping.lower() == "jordan-wigner":
        mapper = JordanWignerMapper()
    else:
        raise ValueError(f"Unsupported mapping: {mapping!r}")

    qubit_op = mapper.map(fermionic_op)
    # SparsePauliOp coercion (nature may return a PauliSumOp-like)
    try:
        from qiskit.quantum_info import SparsePauliOp

        qubit_op = SparsePauliOp(qubit_op)
    except Exception:
        pass

    return QuibitProblem(
        qubit_op=qubit_op,
        num_qubits=qubit_op.num_qubits,
        fermionic_op=fermionic_op,
        num_particles=(n_alpha, n_beta),
        nuclear_repulsion_energy=rhf["E_nuc"],
        hf_reference_energy_elec=rhf["E_elec"],
        hf_reference_energy_total=rhf["E_total"],
        basis=basis,
        mapping=mapping,
        chemistry_backend=ACTIVE_CHEMISTRY_BACKEND,
        geometry=geometry,
    )


def build_qubit_operator(
    geometry: MolecularGeometry,
    basis: str = "sto3g",
    mapping: str = "parity",
    **kwargs,
) -> QuibitProblem:
    """Main entry point: map the molecular Hamiltonian to qubits.

    Parameters
    ----------
    geometry : MolecularGeometry
        The H2 geometry (from :func:`moleculeq.molecule_builder.build_hydrogen`).
    basis : str
        Currently only ``"sto3g"`` is supported (the minimal basis).
    mapping : str
        ``"parity"`` (default) or ``"jordan-wigner"``.

    Returns
    -------
    QuibitProblem with the Pauli qubit operator and metadata.
    """
    if basis.lower() not in ("sto3g",):
        raise ValueError(
            f"Basis {basis!r} is not available. MoleculeQ uses STO-3G (minimal "
            "basis) for the H2 demonstration."
        )
    if geometry.name != "H2":
        raise ValueError(f"Only H2 is supported, received {geometry.name!r}.")

    # Prefer a genuine PySCF-driven problem when the package is installed.
    if ACTIVE_CHEMISTRY_BACKEND == "pyscf":
        try:
            return _build_qubit_operator_via_pyscf_driver(
                geometry, basis=basis, mapping=mapping, **kwargs
            )
        except Exception as exc:  # pragma: no cover - environment fallback
            logger.warning("PySCF path failed (%s); falling back to local integrals.", exc)

    return _build_qubit_operator_nature(geometry, basis=basis, mapping=mapping, **kwargs)


def _build_qubit_operator_via_pyscf_driver(geometry, basis, mapping, **kwargs):
    """Build the qubit operator through the real PySCFDriver (convenience).

    Only used when PySCF is importable; the local integral path is the
    default on platforms (such as Windows) without PySCF wheels.
    """
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
    from qiskit.quantum_info import SparsePauliOp

    driver = PySCFDriver(atom=geometry.atomic_string(), basis=basis)
    problem = driver.run()

    fermionic_op = problem.hamiltonian.second_q_op()
    n_alpha = problem.num_alpha
    n_beta = problem.num_beta
    if mapping.lower() == "parity":
        mapper = ParityMapper(num_particles=(n_alpha, n_beta))
    else:
        mapper = JordanWignerMapper()
    qubit_op = SparsePauliOp(mapper.map(fermionic_op))

    return QuibitProblem(
        qubit_op=qubit_op,
        num_qubits=qubit_op.num_qubits,
        fermionic_op=fermionic_op,
        num_particles=(n_alpha, n_beta),
        nuclear_repulsion_energy=float(problem.nuclear_repulsion_energy),
        hf_reference_energy_elec=float(problem.hf_energies[0]) if problem.hf_energies else np.nan,
        hf_reference_energy_total=np.nan,
        basis=basis,
        mapping=mapping,
        chemistry_backend="pyscf",
        geometry=geometry,
    )


def qubit_operator_as_matrix(qubit_op) -> np.ndarray:
    """Expose a mapped qubit operator as a dense Hermitian matrix (if small).

    Uses the operator's ``to_matrix`` when available (SparsePauliOp /
    PauliSumOp all provide it), else raises a clear error.
    """
    if hasattr(qubit_op, "to_matrix"):
        mat = qubit_op.to_matrix()
        return np.asarray(mat).astype(complex)
    raise TypeError(
        "Cannot materialize a dense matrix for qubit operator of type "
        f"{type(qubit_op).__name__}."
    )
"""Classical reference calculations that benchmark the quantum result.

Two complementary references are provided:

1. :func:`exact_diagonalization` — eigenvalue of the *mapped qubit
   Hamiltonian*.  This is exactly the objective VQE minimises, so it
   isolates the quality of the variational optimisation with zero basis-set
   ambiguity: VQE should meet it to within optimizer tolerance.

2. :func:`fci_reference` — the exact electronic energy of the continuous
   molecular Hamiltonian (full configuration interaction) in the same basis.
   VQE in a complete ansatz converges to this value, not below it.

Both return **electronic energies**.  The interface always adds the nuclear
repulsion explicitly when a total molecular energy is shown, so the three
labels (electronic / nuclear / total) can never be conflated.

Honesty note: these comparisons validate *the pipeline and the algorithm*.
For a 2 qubit system a laptop diagonalizes exactly faster than any quantum
device could, so agreement here is *not* a demonstration of quantum
advantage — the docs and app say this out loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .molecule_builder import MolecularGeometry
from .hartree_fock import solve_rhf, fci_electronic_energy


@dataclass
class ReferenceResult:
    """Bundle of classical benchmark data for a geometry."""

    R_angstrom: float
    E_fci_elec: float
    E_hf_elec: float
    E_exact_elec: Optional[float]
    E_nuc: float
    basis: str = "sto3g"

    @property
    def E_fci_total(self) -> float:
        return self.E_fci_elec + self.E_nuc

    @property
    def E_exact_total(self) -> float:
        return (self.E_exact_elec if self.E_exact_elec is not None else np.nan) + self.E_nuc


def exact_diagonalization(qubit_op) -> float:
    """Ground eigenvalue of the mapped qubit Hamiltonian (electronic, Ha)."""
    from .hamiltonian import qubit_operator_as_matrix

    mat = qubit_operator_as_matrix(qubit_op)
    eigs = np.linalg.eigvalsh(mat)
    return float(eigs[0])


def fci_reference(geometry: MolecularGeometry) -> tuple[float, float, float]:
    """FCI and HF electronic energies plus nuclear repulsion for the geometry.

    Returns (E_fci_elec, E_hf_elec, E_nuc).
    """
    rhf = solve_rhf(geometry)
    e_fci = fci_electronic_energy(geometry, rhf=rhf)
    return float(e_fci), float(rhf["E_elec"]), float(rhf["E_nuc"])


def build_reference(
    geometry: MolecularGeometry,
    qubit_op=None,
    basis: str = "sto3g",
) -> ReferenceResult:
    """Full classical benchmark for one geometry.

    Parameters
    ----------
    geometry : MolecularGeometry
    qubit_op : SparsePauliOp | None
        If provided (already mapped), the exact diagonalization is included.
    basis : str
    """
    e_fci, e_hf, e_nuc = fci_reference(geometry)
    e_exact = exact_diagonalization(qubit_op) if qubit_op is not None else None
    return ReferenceResult(
        R_angstrom=geometry.bond_distance,
        E_fci_elec=e_fci,
        E_hf_elec=e_hf,
        E_exact_elec=e_exact,
        E_nuc=e_nuc,
        basis=basis,
    )


def energy_error(vqe_elec: float, reference_elec: float) -> dict:
    """Small helper describing the gap between VQE and a reference energy."""
    err = float(vqe_elec) - float(reference_elec)
    return {
        "vqe_elec": float(vqe_elec),
        "reference_elec": float(reference_elec),
        "diff_elec": err,
        "relative_diff": abs(err) / max(abs(float(reference_elec)), 1e-12),
    }
"""Bond-distance energy scan for the H2 molecule.

Every point on the scan curve is produced by the *real* simulation pipeline
(RHF → MO integrals → qubit Hamiltonian → VQE), never fabricated. Individual
point failures are caught and reported rather than silently dropping a fake
value into the curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .molecule_builder import (
    MolecularGeometry,
    build_hydrogen,
    build_molecule,
    validate_bond_distance,
    HEH_MIN_BOND_ANGSTROM, HEH_MAX_BOND_ANGSTROM,
    LIH_MIN_BOND_ANGSTROM, LIH_MAX_BOND_ANGSTROM,
    BEH2_MIN_BOND_ANGSTROM, BEH2_MAX_BOND_ANGSTROM,
    MIN_BOND_ANGSTROM, MAX_BOND_ANGSTROM,
)
from .hartree_fock import fci_electronic_energy, solve_rhf
from .hartree_fock_general import (
    solve_rhf_general,
    fci_electronic_energy_general,
    get_molecule_active_spaces,
    active_space_transformation,
)
from .hamiltonian import build_qubit_operator

logger = logging.getLogger(__name__)


@dataclass
class ScanPoint:
    """Result for a single bond distance in a scan."""

    R_angstrom: float
    vqe_electronic: float | None = None
    vqe_total: float | None = None
    exact_electronic: float | None = None
    fci_electronic: float | None = None
    hf_electronic: float | None = None
    nuclear_repulsion: float | None = None
    backend: str = ""
    failed: bool = False
    error: str = ""


@dataclass
class ScanResult:
    """Collection of per-distance results plus metadata."""

    points: list[ScanPoint] = field(default_factory=list)
    backend: str = ""
    total_points: int = 0
    completed_points: int = 0
    failed_points: int = 0
    failure_reasons: list[str] = field(default_factory=list)

    @property
    def distances(self) -> np.ndarray:
        return np.array([p.R_angstrom for p in self.points])

    @property
    def vqe_totals(self) -> np.ndarray:
        return np.array([p.vqe_total for p in self.points if p.vqe_total is not None])

    @property
    def exact_totals(self) -> np.ndarray:
        # A ScanPoint does not store a direct 'exact total'; reconstruct it
        # from electronic + nuclear repulsion.
        vals = [
            p.exact_electronic + p.nuclear_repulsion
            for p in self.points
            if p.exact_electronic is not None and p.nuclear_repulsion is not None
        ]
        return np.array(vals)

    def lowest_energy_point(self) -> Optional[ScanPoint]:
        ok = [p for p in self.points if p.vqe_total is not None]
        if not ok:
            return None
        return min(ok, key=lambda p: p.vqe_total)


def _scan_distances(r_min: float, r_max: float, n_points: int, molecule_name: str = "H2") -> list[float]:
    min_b, max_b = MIN_BOND_ANGSTROM, MAX_BOND_ANGSTROM
    mol = molecule_name.upper()
    if mol in ("HEH+", "HEHPLUS", "HELIUM HYDRIDE CATION (HEH⁺)"):
        min_b, max_b = HEH_MIN_BOND_ANGSTROM, HEH_MAX_BOND_ANGSTROM
    elif mol in ("LIH", "LITHIUM HYDRIDE (LIH)"):
        min_b, max_b = LIH_MIN_BOND_ANGSTROM, LIH_MAX_BOND_ANGSTROM
    elif mol in ("BEH2", "BERYLLIUM DIHYDRIDE (BEH₂)"):
        min_b, max_b = BEH2_MIN_BOND_ANGSTROM, BEH2_MAX_BOND_ANGSTROM

    r_min = float(validate_bond_distance(r_min, min_b, max_b))
    r_max = float(validate_bond_distance(r_max, min_b, max_b))
    n = int(n_points)
    if n < 2:
        raise ValueError("A scan needs at least 2 points.")
    if n > 60:
        raise ValueError("A scan with more than 60 points would be too slow to run live.")
    if r_max <= r_min:
        raise ValueError("r_max must be strictly larger than r_min.")

    return [float(r) for r in np.linspace(r_min, r_max, n)]


def run_scan(
    r_min: float,
    r_max: float,
    n_points: int = 11,
    *,
    molecule_name: str = "H2",
    run_vqe_fn=None,
    progress_callback=None,
    basis: str = "sto3g",
) -> ScanResult:
    """Run the simulation pipeline at every distance in [r_min, r_max]."""
    distances = _scan_distances(r_min, r_max, n_points, molecule_name=molecule_name)
    backend = "local-sto3g-integrals"

    if run_vqe_fn is None:
        from qiskit_nature.second_q.mappers import ParityMapper
        from .quantum_solver import run_vqe

        def default_vqe(qubit_op, num_particles, enuc, *, maxiter=400, geom=None):
            mapper = ParityMapper(num_particles=num_particles)
            n_spatial = 2
            if geom is not None and geom.name == "BeH2":
                n_spatial = 3
            return run_vqe(
                qubit_op,
                num_particles=num_particles,
                nuclear_repulsion_energy=enuc,
                ansatz="uccsd",
                optimizer="SLSQP",
                maxiter=maxiter,
                qubit_mapper=mapper,
                num_spatial_orbitals=n_spatial,
            )

        run_vqe_fn = default_vqe

    points: list[ScanPoint] = []
    for i, r in enumerate(distances):
        pt = ScanPoint(R_angstrom=r)
        try:
            geom = build_molecule(molecule_name, r)
            prob = build_qubit_operator(geom, basis=basis)
            pt.exact_electronic = _eigmin(prob.qubit_op)
            pt.nuclear_repulsion = prob.nuclear_repulsion_energy
            pt.backend = backend

            if geom.name == "H2":
                rhf = solve_rhf(geom)
                pt.fci_electronic = fci_electronic_energy(geom, rhf=rhf)
                pt.hf_electronic = rhf["E_elec"]
            else:
                rhf = solve_rhf_general(geom)
                fci_tot_elec = fci_electronic_energy_general(geom, rhf=rhf)
                core_idx, active_idx, _ = get_molecule_active_spaces(geom)
                e_core, _, _ = active_space_transformation(rhf, core_idx, active_idx)
                pt.fci_electronic = fci_tot_elec - e_core
                pt.hf_electronic = rhf["E_elec"] - e_core

            import inspect
            sig = inspect.signature(run_vqe_fn)
            if "geom" in sig.parameters:
                vqe = run_vqe_fn(
                    prob.qubit_op,
                    prob.num_particles,
                    prob.nuclear_repulsion_energy,
                    geom=geom,
                )
            else:
                vqe = run_vqe_fn(
                    prob.qubit_op,
                    prob.num_particles,
                    prob.nuclear_repulsion_energy,
                )

            pt.vqe_electronic = vqe.electronic_energy
            pt.vqe_total = vqe.total_energy
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scan point R=%.3f failed: %s", r, exc)
            pt.failed = True
            pt.error = str(exc)[:300]
        points.append(pt)
        if progress_callback is not None:
            progress_callback(i + 1, len(distances))

    completed = sum(not p.failed for p in points)
    failed_reasons = [p.error for p in points if p.failed and p.error]
    return ScanResult(
        points=points,
        backend=backend,
        total_points=len(points),
        completed_points=completed,
        failed_points=len(points) - completed,
        failure_reasons=failed_reasons,
    )


def _eigmin(qubit_op) -> float:
    import numpy as np

    from .classical_reference import exact_diagonalization

    return exact_diagonalization(qubit_op)
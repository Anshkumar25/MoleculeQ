"""Molecule construction and geometry validation for MoleculeQ.

The molecule builder turns user-friendly input (molecule name + bond
distance in angstrom) into a structured molecular geometry that the rest of
the pipeline consumes. Energies are never hard-coded here: only geometry
contracts are produced; all energies are computed downstream from this
geometry.

Units
-----
- Internal geometry coordinates stored in **angstrom** (the convention the
  Qiskit Nature drivers and users expect).
- Conversion to **bohr** (atomic units) provided explicitly and used by the
  integral engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 1 Å = 1.8897261246257702 Bohr  (CODATA)
ANGSTROM_PER_BOHR = 0.52917721067
BOHR_PER_ANGSTROM = 1.0 / ANGSTROM_PER_BOHR

# Allowed scan/bond range for STO-3G H2 (in angstrom).
MIN_BOND_ANGSTROM = 0.25
MAX_BOND_ANGSTROM = 4.00

SUPPORTED_MOLECULES = ("H2",)


class MoleculeBuilderError(ValueError):
    """Raised for invalid molecular geometry requests."""


@dataclass(frozen=True)
class MolecularGeometry:
    """A validated molecular geometry.

    Attributes
    ----------
    name : str            (e.g. "H2")
    symbols : tuple[str]  element symbols, one per atom, in order
    coords : tuple[tuple[float, float, float]]
                          Cartesian coordinates in angstrom
    charge : int          total molecular charge (0 for H2)
    multiplicity : int    2S+1 (1 for H2 singlet)
    bond_distance : float H-H bond distance in angstrom
    """

    name: str
    symbols: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]
    charge: int = 0
    multiplicity: int = 1
    bond_distance: float = field(default=0.0)

    @property
    def num_atoms(self) -> int:
        return len(self.symbols)

    def atomic_string(self) -> str:
        """Geometry line for a driver that accepts `"H 0 0 0; H 0 0 0.735"`."""
        return "; ".join(
            f"{sym} {x:.8f} {y:.8f} {z:.8f}"
            for (sym, (x, y, z)) in zip(self.symbols, self.coords)
        )

    def to_nuclei_z(self) -> tuple[float, float]:
        """z-positions of the two hydrogens in angstrom (bond along z-axis)."""
        if self.num_atoms != 2:
            raise MoleculeBuilderError(
                f"Expected 2 nuclei, got {self.num_atoms} for {self.name}"
            )
        return self.coords[0][2], self.coords[1][2]


def validate_bond_distance(bond_angstrom: float) -> float:
    """Validate a H-H bond distance (angstrom); return it unchanged.

    Raises
    ------
    MoleculeBuilderError for non-finite or out-of-range distances.
    """
    try:
        bond = float(bond_angstrom)
    except (TypeError, ValueError) as exc:
        raise MoleculeBuilderError(
            f"Bond distance must be a number, received {bond_angstrom!r}."
        ) from exc

    if not bond > 0 or not bond < float("inf"):
        raise MoleculeBuilderError(
            f"Bond distance must be positive and finite, got {bond!r}."
        )
    if bond < MIN_BOND_ANGSTROM:
        raise MoleculeBuilderError(
            f"Bond distance {bond:.3f} Å is too small. "
            f"STO-3G supports ≥ {MIN_BOND_ANGSTROM:.2f} Å "
            "(smaller values make the basis linearly dependent)."
        )
    if bond > MAX_BOND_ANGSTROM:
        raise MoleculeBuilderError(
            f"Bond distance {bond:.3f} Å is too large. "
            f"STO-3G supports ≤ {MAX_BOND_ANGSTROM:.2f} Å "
            "(the near-dissociation RHF limit is unreliable beyond this)."
        )
    return bond


def bond_angstrom_to_bohr(bond_angstrom: float) -> float:
    """Convert a bond distance from angstrom to bohr."""
    return bond_angstrom * BOHR_PER_ANGSTROM


def build_hydrogen(bond_angstrom: float) -> MolecularGeometry:
    """Build an H2 molecule with the given bond distance.

    The two hydrogen nuclei are placed on the z-axis, symmetrically around
    the origin, at z = ± (R/2).

    Parameters
    ----------
    bond_angstrom : float
        H-H distance in angstrom.

    Returns
    -------
    MolecularGeometry
    """
    bond = validate_bond_distance(bond_angstrom)
    half = bond / 2.0
    return MolecularGeometry(
        name="H2",
        symbols=("H", "H"),
        coords=((0.0, 0.0, half), (0.0, 0.0, -half)),
        charge=0,
        multiplicity=1,  # singlet
        bond_distance=bond,
    )
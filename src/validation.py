"""Shared input-validation helpers for the MoleculeQ UI and API.

These keep the interactive layer honest: every user-supplied number is
checked *before* it reaches the expensive quantum pipeline, so an invalid
value produces a friendly message instead of a cryptic traceback.
"""

from __future__ import annotations

from typing import Any

from .molecule_builder import (
    MAX_BOND_ANGSTROM,
    MIN_BOND_ANGSTROM,
    HEH_MIN_BOND_ANGSTROM, HEH_MAX_BOND_ANGSTROM,
    LIH_MIN_BOND_ANGSTROM, LIH_MAX_BOND_ANGSTROM,
    BEH2_MIN_BOND_ANGSTROM, BEH2_MAX_BOND_ANGSTROM,
    MoleculeBuilderError,
    validate_bond_distance,
)

VALID_MAPPINGS = ("parity", "jordan-wigner")
VALID_ANSATZE = ("ry", "uccsd")
VALID_OPTIMIZERS = ("SLSQP", "COBYLA")
VALID_BACKENDS = ("statevector", "aer")


def validate_opt_int(name: str, value: Any, low: int, high: int) -> int:
    """Validate an integer option within [low, high]; raise ValueError."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}.") from None
    if not (low <= iv <= high):
        raise ValueError(f"{name} must be between {low} and {high}, got {iv}.")
    return iv


def validate_opt_float(name: str, value: Any, low: float, high: float) -> float:
    """Validate a float option; ensure it is finite and within range."""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}.") from None
    if not (low <= fv <= high):
        raise ValueError(f"{name} must be between {low} and {high}, got {fv:.4g}.")
    return fv


def validate_bond(value: Any) -> float:
    """Validate a bond distance (in angstrom); raise MoleculeBuilderError."""
    return validate_bond_distance(value)


def validate_bond_for_molecule(name: str, value: Any) -> float:
    """Validate a bond distance for a specific molecule."""
    mol = name.upper()
    min_b, max_b = MIN_BOND_ANGSTROM, MAX_BOND_ANGSTROM
    if mol in ("HEH+", "HEHPLUS", "HELIUM HYDRIDE CATION (HEH⁺)"):
        min_b, max_b = HEH_MIN_BOND_ANGSTROM, HEH_MAX_BOND_ANGSTROM
    elif mol in ("LIH", "LITHIUM HYDRIDE (LIH)"):
        min_b, max_b = LIH_MIN_BOND_ANGSTROM, LIH_MAX_BOND_ANGSTROM
    elif mol in ("BEH2", "BERYLLIUM DIHYDRIDE (BEH₂)"):
        min_b, max_b = BEH2_MIN_BOND_ANGSTROM, BEH2_MAX_BOND_ANGSTROM
    return validate_bond_distance(value, min_b, max_b)


def validate_scan_window(r_min: Any, r_max: Any, n_points: Any, molecule_name: str = "H2") -> tuple[float, float, int]:
    """Validate a scan window; return (r_min, r_max, n_points)."""
    a = validate_bond_for_molecule(molecule_name, r_min)
    b = validate_bond_for_molecule(molecule_name, r_max)
    n = validate_opt_int("number of points", n_points, 3, 60)
    if b <= a:
        raise ValueError("The maximum bond distance must exceed the minimum.")
    return a, b, n


def validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    """Validate that ``value`` is one of ``allowed`` (case-insensitive)."""
    low = value.lower()
    for candidate in allowed:
        if low == candidate.lower():
            return candidate
    raise ValueError(f"{name} must be one of {allowed}, got {value!r}.")


def validate_backend(value: str) -> str:
    return validate_choice("backend", value, VALID_BACKENDS)


def validate_ansatz(value: str) -> str:
    return validate_choice("ansatz", value, VALID_ANSATZE)


def validate_optimizer(value: str) -> str:
    return validate_choice("optimizer", value, VALID_OPTIMIZERS)


def validate_mapping(value: str) -> str:
    return validate_choice("mapping", value, VALID_MAPPINGS)


def friendly_scan_bounds(molecule_name: str = "H2") -> str:
    mol = molecule_name.upper()
    min_b, max_b = MIN_BOND_ANGSTROM, MAX_BOND_ANGSTROM
    if mol in ("HEH+", "HEHPLUS", "HELIUM HYDRIDE CATION (HEH⁺)"):
        min_b, max_b = HEH_MIN_BOND_ANGSTROM, HEH_MAX_BOND_ANGSTROM
    elif mol in ("LIH", "LITHIUM HYDRIDE (LIH)"):
        min_b, max_b = LIH_MIN_BOND_ANGSTROM, LIH_MAX_BOND_ANGSTROM
    elif mol in ("BEH2", "BERYLLIUM DIHYDRIDE (BEH₂)"):
        min_b, max_b = BEH2_MIN_BOND_ANGSTROM, BEH2_MAX_BOND_ANGSTROM
    return (
        f"Bond distances must lie in [{min_b:.2f}, {max_b:.2f}] Å "
        "(STO-3G basis validity range)."
    )
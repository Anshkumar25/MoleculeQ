"""General Restricted Hartree-Fock and FCI solver with Active Space support.

Handles closed-shell molecules (HeH+, LiH, BeH2, etc.) while leaving
`src/hartree_fock.py` completely untouched for H2.
"""

from __future__ import annotations

import numpy as np

from .molecule_builder import MolecularGeometry, BOHR_PER_ANGSTROM
from .integrals_general import compute_general_integrals


def get_molecule_electron_count(geometry: MolecularGeometry) -> int:
    """Return the total number of electrons for a geometry based on atomic numbers and charge."""
    z_map = {"H": 1, "He": 2, "Li": 3, "Be": 4}
    n_elec = sum(z_map[sym] for sym in geometry.symbols) - geometry.charge
    return n_elec


def get_molecule_active_spaces(geometry: MolecularGeometry) -> tuple[list[int], list[int], tuple[int, int]]:
    """Return (core_indices, active_indices, active_particles) for a molecule.

    Active spaces:
    - H2: core=[], active=[0, 1] (2 active electrons in 2 MOs)
    - HeH+: core=[], active=[0, 1] (2 active electrons in 2 MOs)
    - LiH: core=[0] (Li 1s frozen), active=[1, 2] (2 active electrons in 2 MOs: HOMO & LUMO)
    - BeH2: core=[0] (Be 1s frozen), active=[1, 2, 3] (4 active electrons in 3 MOs: HOMO-1, HOMO, LUMO)
    """
    name = geometry.name
    if name == "HeH+":
        return [], [0, 1], (1, 1)
    if name == "LiH":
        return [0], [1, 2], (1, 1)
    if name == "BeH2":
        return [0], [1, 2, 3], (2, 2)
    # Default fallback: 0 core, all MOs active
    n_elec = get_molecule_electron_count(geometry)
    n_alpha = n_elec // 2
    n_beta = n_elec // 2
    return [], list(range(n_alpha + 1)), (n_alpha, n_beta)


def solve_rhf_general(
    geometry: MolecularGeometry,
    *,
    max_iter: int = 500,
    damping: float = 0.4,
    tol: float = 1e-11,
) -> dict:
    """Solve RHF for any closed-shell molecule using STO-3G general integrals."""
    coords_bohr = tuple(
        (x * BOHR_PER_ANGSTROM, y * BOHR_PER_ANGSTROM, z * BOHR_PER_ANGSTROM)
        for (x, y, z) in geometry.coords
    )

    ints = compute_general_integrals(geometry.symbols, coords_bohr)
    n = ints["n_ao"]
    S = ints["S"]
    h_core = ints["h_core"]
    g2 = ints["g2_ao"]

    n_elec = get_molecule_electron_count(geometry)
    if n_elec % 2 != 0:
        raise ValueError(f"solve_rhf_general currently supports closed-shell singlets, got {n_elec} electrons.")

    n_occ = n_elec // 2

    # Orthonormalizer X = S^{-1/2}
    w, V = np.linalg.eigh(S)
    X = V @ np.diag(1.0 / np.sqrt(w)) @ V.T

    density = np.zeros((n, n))
    fock = h_core.copy()
    converged = False
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1
        fock = h_core.copy()
        for mu in range(n):
            for nu in range(n):
                coul = 0.0
                exch = 0.0
                for lam in range(n):
                    for sig in range(n):
                        coul += density[lam, sig] * g2[mu, nu, lam, sig]
                        exch += 0.5 * density[lam, sig] * g2[mu, sig, lam, nu]
                fock[mu, nu] += coul - exch

        energies, C_ortho = np.linalg.eigh(X.T @ fock @ X)
        C = X @ C_ortho

        new_density = np.zeros((n, n))
        for i in range(n_occ):
            new_density += 2.0 * np.outer(C[:, i], C[:, i])

        max_diff = float(np.max(np.abs(new_density - density)))
        density = (
            new_density
            if it == 0
            else (1.0 - damping) * new_density + damping * density
        )
        if max_diff < tol:
            converged = True
            break

    E_elec = 0.5 * float(np.sum(density * (h_core + fock)))
    E_nuc = ints["enuc"]
    E_total = E_elec + E_nuc

    return {
        "n_ao": n,
        "n_elec": n_elec,
        "S": S,
        "h_core": h_core,
        "g2_ao": g2,
        "C": C,
        "orbital_energies": energies,
        "density": density,
        "E_elec": float(E_elec),
        "E_nuc": float(E_nuc),
        "E_total": float(E_total),
        "converged": converged,
        "iterations": iterations,
    }


def active_space_transformation(
    rhf: dict, core_indices: list[int], active_indices: list[int]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Transform MO integrals into an active space.

    Returns
    -------
    e_core : float
        Electronic energy contribution from frozen core orbitals.
    h1_act : np.ndarray (n_act, n_act)
        Effective 1-electron Hamiltonian in active spatial MO basis.
    g2_act_phys : np.ndarray (n_act, n_act, n_act, n_act)
        2-electron integrals in active spatial MO basis (physicists notation).
    """
    C = rhf["C"]
    h_ao = rhf["h_core"]
    g2_ao = rhf["g2_ao"]

    h1_mo = C.T @ h_ao @ C
    g2_mo_chem = np.einsum("pqrs,pi,qj,rk,sl->ijkl", g2_ao, C, C, C, C)

    # Core electronic energy
    e_core = 0.0
    for i in core_indices:
        e_core += 2.0 * h1_mo[i, i]
        for j in core_indices:
            e_core += 2.0 * g2_mo_chem[i, i, j, j] - g2_mo_chem[i, j, j, i]

    # Effective active 1-electron Hamiltonian
    n_act = len(active_indices)
    h1_act = np.zeros((n_act, n_act))
    for p_idx, p in enumerate(active_indices):
        for q_idx, q in enumerate(active_indices):
            val = h1_mo[p, q]
            for i in core_indices:
                val += 2.0 * g2_mo_chem[p, q, i, i] - g2_mo_chem[p, i, i, q]
            h1_act[p_idx, q_idx] = val

    # Active 2-electron integrals (chemists -> physicists)
    g2_act_chem = np.zeros((n_act, n_act, n_act, n_act))
    for p_idx, p in enumerate(active_indices):
        for q_idx, q in enumerate(active_indices):
            for r_idx, r in enumerate(active_indices):
                for s_idx, s in enumerate(active_indices):
                    g2_act_chem[p_idx, q_idx, r_idx, s_idx] = g2_mo_chem[p, q, r, s]

    g2_act_phys = np.transpose(g2_act_chem, (0, 2, 1, 3))
    return float(e_core), h1_act, g2_act_phys


def fci_electronic_energy_general(geometry: MolecularGeometry, rhf: dict | None = None) -> float:
    """Exact ground electronic energy in the active space for general molecules."""
    if rhf is None:
        rhf = solve_rhf_general(geometry)

    core_idx, active_idx, (n_alpha_act, n_beta_act) = get_molecule_active_spaces(geometry)
    e_core, h1_act, g2_act_phys = active_space_transformation(rhf, core_idx, active_idx)

    n_act = len(active_idx)
    n_spin = 2 * n_act

    # Build spin-orbital 1-body and 2-body tensors
    h1_spin = np.zeros((n_spin, n_spin))
    for p in range(n_spin):
        u, s_p = p // 2, p % 2
        for q in range(n_spin):
            v, s_q = q // 2, q % 2
            if s_p == s_q:
                h1_spin[p, q] = h1_act[u, v]

    g2_spin_phys = np.zeros((n_spin, n_spin, n_spin, n_spin))
    for i in range(n_spin):
        u, s_i = i // 2, i % 2
        for j in range(n_spin):
            v, s_j = j // 2, j % 2
            for k in range(n_spin):
                w, s_k = k // 2, k % 2
                for l in range(n_spin):
                    x, s_l = l // 2, l % 2
                    if s_i == s_k and s_j == s_l:
                        g2_spin_phys[i, j, k, l] = g2_act_phys[u, v, w, x]

    # Enumerate Slater determinants in active space sector (n_alpha_act, n_beta_act)
    alpha_orbs = [2 * u for u in range(n_act)]
    beta_orbs = [2 * u + 1 for u in range(n_act)]

    from itertools import combinations

    determinants: list[frozenset[int]] = []
    for a_comb in combinations(alpha_orbs, n_alpha_act):
        for b_comb in combinations(beta_orbs, n_beta_act):
            determinants.append(frozenset(a_comb + b_comb))

    m = len(determinants)

    def slater_condon(I: frozenset[int], J: frozenset[int]) -> float:
        if I == J:
            val = 0.0
            for i in I:
                val += h1_spin[i, i]
            occ_list = sorted(I)
            for a in range(len(occ_list)):
                for b in range(a + 1, len(occ_list)):
                    i, j = occ_list[a], occ_list[b]
                    val += g2_spin_phys[i, j, i, j] - g2_spin_phys[i, j, j, i]
            return val

        diff_from_I = list(I - J)
        diff_to_J = list(J - I)
        common = I & J
        if len(diff_from_I) == 1 and len(diff_to_J) == 1:
            i_occ = diff_from_I[0]
            v_virt = diff_to_J[0]
            val = h1_spin[i_occ, v_virt]
            for j in sorted(common):
                val += g2_spin_phys[i_occ, j, v_virt, j] - g2_spin_phys[i_occ, j, j, v_virt]
            return val

        if len(diff_from_I) == 2 and len(diff_to_J) == 2:
            i, j = sorted(diff_from_I)
            v, w = sorted(diff_to_J)
            val = g2_spin_phys[i, j, v, w] - g2_spin_phys[i, j, w, v]
            return val

        return 0.0

    hmat = np.zeros((m, m))
    for a, det_a in enumerate(determinants):
        for b, det_b in enumerate(determinants):
            hmat[a, b] = slater_condon(det_a, det_b)

    eigs = np.linalg.eigvalsh(hmat)
    return float(eigs[0] + e_core)

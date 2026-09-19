"""Restricted Hartree-Fock and full configuration-interaction for H2.

This module contains the small classical electronic-structure routines that
MoleculeQ needs:

1. **RHF** (:func:`solve_rhf`): iteratively solves the Roothaan-Hall
   equations for H2 in the STO-3G basis and returns the molecular-orbital
   coefficients.  The MOs define the basis in which the second-quantized /
   qubit Hamiltonian is written.

2. **FCI** (:func:`fci_electronic_energy`): the exact ground electronic
   energy of the molecule in the (2 spatial orbital, 2 electron) space. This
   is the *chemistry* benchmark: exact diagonalization of the continuous
   molecular Hamiltonian in the chosen basis.  VQE should converge close to
   (never below) this value.

3. **MO-basis integrals** (:func:`mo_integrals`): one- and two-electron
   integrals transformed from the AO basis into the RHF MO basis with the
   spin-orbital structure the second-quantized Hamiltonian needs.

All energies here are **electronic energies** (they exclude nuclear
repulsion).  For total molecular energies add the nuclear repulsion term
(1/R for two protons) reported by the integral engine.

Convention notes
----------------
- Two-electron integrals are kept in **chemists' notation** (ij|kl).
- RHF orbitals are real, so transient factors are avoided.
- The RHF density-mixing parameter and convergence threshold are exposed
  for reproducibility.
"""

from __future__ import annotations

import numpy as np

from .molecule_builder import MolecularGeometry
from . import integrals_h2 as integrals


def solve_rhf(
    geometry: MolecularGeometry,
    *,
    max_iter: int = 500,
    damping: float = 0.4,
    tol: float = 1e-11,
) -> dict:
    """Solve restricted Hartree-Fock for H2 in the STO-3G basis.

    Parameters
    ----------
    geometry : MolecularGeometry
        The validated H2 geometry.
    max_iter : int
        Maximum SCF cycles.
    damping : float
        Fraction of the previous density kept in each mixing step.
    tol : float
        Convergence threshold on the maximum density-matrix change.

    Returns
    -------
    dict with keys:
        R_bohr, n_ao, S, h_core, g2_ao,  C (MO coeffs, 2x2), orbital_energies,
        density, E_elec, E_nuc, E_total, converged, iterations
    """
    if geometry.name != "H2" or geometry.num_atoms != 2:
        raise ValueError(
            f"solve_rhf currently supports H2 only, got {geometry.name!r}"
        )

    R_bohr = geometry.bond_distance * integrals.BOHR_PER_ANGSTROM
    ints = integrals.h2_integrals(R_bohr)

    n = 2  # 2 basis functions
    S = integrals.ao_overlap_matrix(ints)
    h_core = integrals.ao_core_hamiltonian(ints)
    g2 = integrals.ao_two_electron_tensor(ints)

    # Orthonormalizer  X = S^{-1/2}
    w, V = np.linalg.eigh(S)
    X = V @ np.diag(1.0 / np.sqrt(w)) @ V.T

    density = np.zeros((n, n))
    fock = h_core.copy()
    converged = False
    iterations = 0
    for it in range(max_iter):
        iterations = it + 1
        # Build Fock matrix:  F = h_core + 2J - K (closed shell)
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
        new_density = 2.0 * np.outer(C[:, 0], C[:, 0])  # 1 occupied MO (2 e-)

        max_diff = float(np.max(np.abs(new_density - density)))
        density = (
            new_density
            if it == 0
            else (1.0 - damping) * new_density + damping * density
        )
        if max_diff < tol:
            converged = True
            break

    # Electronic energy:  E = 1/2 tr[ P (h + F) ]
    E_elec = 0.5 * float(np.sum(density * (h_core + fock)))
    E_nuc = ints["enuc"]
    E_total = E_elec + E_nuc

    return {
        "R_bohr": R_bohr,
        "n_ao": n,
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


def mo_integrals(rhf: dict) -> tuple[np.ndarray, np.ndarray]:
    """Transform AO-basis integrals into the RHF MO spin-orbital basis.

    Returns
    -------
    h1_mo : np.ndarray (4, 4)
        One-electron integrals in the 4 spin-orbital basis (chemists'
        ordering: spin orbitals (0a, 0b, 1a, 1b) — spin up block first).
    g2_mo_phys : np.ndarray (4, 4, 4, 4)
        Two-electron integrals in physicists' ordering <ij|kl>.
    """
    C = rhf["C"]
    n = C.shape[0]
    # spin-orbital transform: h1_spin[p, q] with p,q in {0a,0b,1a,1b}
    h1_spin = np.zeros((2 * n, 2 * n))
    # Transform AO->MO spatial: h_mo = C^T h_ao C
    h_mo = C.T @ rhf["h_core"] @ C

    # map spatial pairs: orbital u (0,1) -> spinorbitals 2u (alpha), 2u+1 (beta)
    for p in range(2 * n):
        u = p // 2
        s_p = p % 2
        for q in range(2 * n):
            v = q // 2
            s_q = q % 2
            if s_p == s_q:
                h1_spin[p, q] = h_mo[u, v]

    # two-electron: phys <ij|kl> = (ik|jl)_chem.
    g2_ao = rhf["g2_ao"]
    # transform each spatial index pair
    def ao2mo(t_ao: np.ndarray) -> np.ndarray:
        # t_ao[i,j,k,l] spatial chemists -> spatial physicists via P: _idx
        # phys <ij|kl> = sum_{pqrs} C_pi C_qk C_rj C_sl * (pq|rs)_chem
        return np.einsum("pqrs,pi,qk,rj,sl->ijkl", t_ao, C, C, C, C)

    g2_mo_phys_spatial = ao2mo(g2_ao)  # (2,2,2,2) phys order spatial

    # Expand to spin orbitals.  Phys <ij|kl>: i,j (bra, spin s1,s2) k,l (ket).
    # with real orbitals we use:  <ij|kl> = (ik|jl)
    # Standard spin rule: integrals where spin-fulc embedding alpha/beta
    # are zero unless s_i == s_k and s_j == s_l.
    g2_spin_phys = np.zeros((2 * n, 2 * n, 2 * n, 2 * n))
    for i in range(2 * n):
        u = i // 2
        s_i = i % 2
        for j in range(2 * n):
            v = j // 2
            s_j = j % 2
            for k in range(2 * n):
                w = k // 2
                s_k = k % 2
                for l in range(2 * n):
                    x = l // 2
                    s_l = l % 2
                    if s_i == s_k and s_j == s_l:
                        g2_spin_phys[i, j, k, l] = g2_mo_phys_spatial[u, v, w, x]
    return h1_spin, g2_spin_phys


def fci_electronic_energy(geometry: MolecularGeometry, rhf: dict | None = None) -> float:
    """Full-CI ground electronic energy for H2 (2 electrons, 2 spatial MOs).

    Represents the molecular Hamiltonian in the RHF MO spin-orbital basis
    and diagonalizes the full many-body matrix restricted to the physical
    sector (N=2 electrons, Sz=0).  This is the exact ground electronic
    energy in the STO-3G basis.

    Parameters
    ----------
    geometry : MolecularGeometry
    rhf : dict, optional
        Precomputed RHF solve. If None, a fresh solve is done.

    Returns
    -------
    float : E_FCI (electronic energy, Ha).
    """
    if rhf is None:
        rhf = solve_rhf(geometry)
    h1_spin, g2_spin_phys = mo_integrals(rhf)
    n_spin = h1_spin.shape[0]

    # enumerate determinants with N=2, Sz=0
    def occupations_ok(occ: list[int]) -> bool:
        if len(occ) != 2:
            return False
        sz = sum(1 if i % 2 == 0 else -1 for i in occ)
        return sz == 0

    determinants: list[frozenset[int]] = []
    for i in range(n_spin):
        for j in range(i + 1, n_spin):
            occ = (i, j)
            if occupations_ok(occ):
                determinants.append(frozenset(occ))

    det_index = {d: k for k, d in enumerate(determinants)}
    m = len(determinants)

    def slater_condon(I: frozenset[int], J: frozenset[int]) -> float:
        # physical sector matrix element <I|H|J>
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

        # single excitation: I = O u {i}, J = O u {v}
        common = I & J
        diff_from_I = list(I - J)
        diff_to_J = list(J - I)
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
    return float(eigs[0])
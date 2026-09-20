"""Minimal-basis STO-3G one- and two-electron integrals for the H2 molecule.

WHY THIS MODULE EXISTS
----------------------
Qiskit Nature normally builds molecular problems through a classical
chemistry *driver* (PySCFDriver, PSI4Driver, ...). PySCF, the recommended
drfiver, is a compiled package that ships wheels for Linux/macOS but **not
for Windows**. To keep MoleculeQ fully runnable on this Windows machine we
implement a small, dependency-free STO-3G integral engine for the
hydrogen molecule.

This is **not** a fake or hard-coded result: the one- and two-electron
integrals below are *computed at runtime* for the requested internuclear
distance R, following the closed-form Gaussian-product formulas from
Szabo & Ostlund, *Modern Quantum Chemistry* (Appendices A & B and the
H2 problems). The output represents the true STO-3G electronic-structure
problem for H2 and is validated in the test-suite against the widely
reproduced textbook reference values (e.g. S_12(1.4 Bohr) = 0.6643).

If PySCF becomes available on the machine, :mod:`moleculeq.hamiltonian`
prefers the PySCFDriver and this module is used only as a documented
fallback.

UNIT CONVENTION
---------------
All quantities in this module are in **Hartree atomic units**: energies in
Hartree (Ha), distances in Bohr (a0).  Keep conversion to Å where the
interface interacts with the user (1 a0 = 0.52917721067 Å).

INTEGRAL DEFINITIONS
--------------------
For the two nuclei at +R/2 and -R/2 on the z-axis, basis functions
chi_A (centered on H_A) and chi_B (centered on H_B), STO-3G contraction of
three s-Gaussians:

    S      = <chi_A | chi_B>                       (overlap)
    T      = <chi_A | -1/2 nabla^2 | chi_B>        (kinetic energy)
    V_C    = <chi_A | -1/|r - R_C| | chi_B>        (nuclear attraction, C = A or B)
    (ij|kl) = <chi_i chi_j | 1/r_12 | chi_k chi_l> (two-electron, chemists' notation)

h1 (core Hamiltonian) = T + V_A + V_B.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# STO-3G contraction for hydrogen 1s (scaled to zeta = 1.24)
# ---------------------------------------------------------------------------
# Gaussian exponents (already include the zeta^2 scaling for H) and
# normalized contraction coefficients (Hehre / Pople STO-3G table, as used
# in Szabo & Ostlund).
H_EXP: list[float] = [3.42525091, 0.62391373, 0.16885540]
H_COEF: list[float] = [0.154329, 0.535328, 0.444635]

BOHR_PER_ANGSTROM = 1.8897261246257702
ANGSTROM_PER_BOHR = 1.0 / BOHR_PER_ANGSTROM


# ---------------------------------------------------------------------------
# Gaussian primitive helpers (unnormalized primitives; normalization is
# applied through the contraction coefficients and prefactors below).
# ---------------------------------------------------------------------------
def _norm_gaussian(exponent: float) -> float:
    """Normalization constant N such that N * exp(-a r^2) is normalized."""
    return pow(2.0 * exponent / math.pi, 0.75)


def _gauss_pair_like(a: float, b: float, rab2: float) -> float:
    """The Gaussian product-theorem prefactor for two s-type primitives.

    A primitive pair g_a, g_b at squared distance rab2 has
        <g_a | g_b> = (pi/(a+b))^{3/2} exp(- ab/(a+b) * rab2).
    """
    p = a + b
    reduced = a * b / p
    return pow(math.pi / p, 1.5) * math.exp(-reduced * rab2)


# ---------------------------------------------------------------------------
# One-center / two-center primitive integrals (s-orbitals on z-axis)
# ---------------------------------------------------------------------------
def _s_overlap(a: float, b: float, rab2: float) -> float:
    """Overlap of two (unnormalized) s-Gaussians."""
    return _gauss_pair_like(a, b, rab2)


def _s_kinetic(a: float, b: float, rab2: float) -> float:
    """Kinetic energy <g_a | -1/2 nabla^2 | g_b> (unnormalized gaussians)."""
    p = a + b
    q = a * b / p
    pref = _gauss_pair_like(a, b, rab2)
    # < -1/2 laplacian > for two s-type gaussians:
    #   pref * [ 3*q - 2*q^2 * rab2 ]   (see Helgaker / Szabo-Ostlund).
    return pref * (3.0 * q - 2.0 * q * q * rab2)


def _s_nuclear_attraction(a: float, b: float, rab2: float, rc: float, rp: float, p: float) -> float:
    """Nuclear attraction -< g_a | 1/|r - C| | g_b > for a nucleus at z = rc.

    Closed form for two s-type primitives centered at +R/2,-R/2 (centers of
    mass R_P = (a*za + b*zb)/p) and nuclear charge located at rc:
        V = -(2 pi / p) e^{-ab/p * R_AB^2} * F0( p * |R_P - C|^2 )
    F0 is the Boys function.  All centers lie on the z-axis so the squared
    distance is simply (rp - rc)^2.
    """
    z = rp - rc
    arg = p * z * z
    pref = -(2.0 * math.pi / p) * math.exp(-a * b / p * rab2)
    return pref * _boys0(arg)


def _boys0(x: float) -> float:
    """Boys function F0(x) computed in a numerically stable way."""
    if x < 1e-8:
        return 1.0 - x / 3.0
    return 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))


# ---------------------------------------------------------------------------
# Two-electron four-center integral for four s-type primitives
# (all centers on the z-axis).  Closed form via Gaussian-product theorem:
#   (ab|cd) = (2 pi^(5/2) / (p^2 sqrt(p+q))) * exp(-q_ab R_ab^2 - q_cd R_cd^2)
#             * F0( (p + q) * |R_pq|^2 )
# with the reduce-exponent/prefactor machinery of Szabo-Ostlund Appendix B.
# ---------------------------------------------------------------------------
def _s_two_electron(*, ea, eb, ec, ed, za, zb, zc, zd):
    """Two-electron repulsion of four unnormalized s-gaussians (all z-axis).

    ea..ed : primitive exponents
    za..zd : center positions (z coordinate)
    Returns (chemists') (AB|CD).

    Closed form (derived in this module's docstring / test notebook via
    Poisson's integral of the Coulomb kernel):

        p = ea+eb,  q = ec+ed,   K_AB = e^{-(ea·eb/ p)·|A-B|^2},
        K_CD = e^{-(ec·ed/ q)·|C-D|^2},    R_pq = |R_P - R_Q|,
        (AB|CD) = K_AB K_CD (8 pi^(7/2) / (p q sqrt(p+q)))
                                        * F0( (p q/(p+q)) R_pq^2 )
    """
    p = ea + eb
    k_ab = math.exp(-ea * eb / p * (za - zb) ** 2)
    rp = (ea * za + eb * zb) / p

    q = ec + ed
    k_cd = math.exp(-ec * ed / q * (zc - zd) ** 2)
    rq = (ec * zc + ed * zd) / q

    rpq2 = (rp - rq) ** 2

    pref = 2.0 * math.pi ** 2.5 / (p * q * math.sqrt(p + q))
    return pref * k_ab * k_cd * _boys0(p * q / (p + q) * rpq2)


def _contracted_overlap(R: float) -> float:
    """Overlap S_12 of the two contracted H STO-3G orbitals at distance R."""
    n = len(H_EXP)
    val = 0.0
    for i in range(n):
        for j in range(n):
            val += (
                H_COEF[i]
                * H_COEF[j]
                * _norm_gaussian(H_EXP[i])
                * _norm_gaussian(H_EXP[j])
                * _s_overlap(H_EXP[i], H_EXP[j], R * R)
            )
    return val


def _contracted_kinetic(R: float) -> float:
    """Kinetc matrix elements: T_AA (diag) and T_AB (off-diag)."""
    n = len(H_EXP)
    t_aa = 0.0
    t_ab = 0.0
    for i in range(n):
        for j in range(n):
            nij = (
                H_COEF[i]
                * H_COEF[j]
                * _norm_gaussian(H_EXP[i])
                * _norm_gaussian(H_EXP[j])
            )
            t_aa += nij * _s_kinetic(H_EXP[i], H_EXP[j], 0.0)
            t_ab += nij * _s_kinetic(H_EXP[i], H_EXP[j], R * R)
    return np.array([t_aa, t_ab])


def _contracted_nuclear_attraction(R: float) -> tuple[float, float, float, float]:
    """Nuclear attraction matrix elements.

    Returns (V_AA_A, V_AB_A, V_AA_B, V_AB_B):
      V_AA_A = <chi_A| -1/|r-R_A| |chi_A>   (own nucleus)
      V_AB_A = <chi_A| -1/|r-R_A| |chi_B>   (off-diagonal, nucleus A)
      V_AA_B = <chi_A| -1/|r-R_B| |chi_A>   (opposite nucleus)
      V_AB_B = <chi_A| -1/|r-R_B| |chi_B>
    Centers: H_A at +R/2, H_B at -R/2.
    """
    n = len(H_EXP)
    za = R / 2.0  # nuclear center (and orbital A) at +R/2
    zb = -R / 2.0  # nuclear center (and orbital B) at -R/2
    v_aa_a = v_ab_a = v_aa_b = v_ab_b = 0.0
    for i in range(n):
        for j in range(n):
            nij = (
                H_COEF[i]
                * H_COEF[j]
                * _norm_gaussian(H_EXP[i])
                * _norm_gaussian(H_EXP[j])
            )
            p = H_EXP[i] + H_EXP[j]
            # --- both gaussians on center A ---
            rp_aa = za  # product center = za
            # --- gaussian i on A, j on B ---
            rp_ab = (H_EXP[i] * za + H_EXP[j] * zb) / p
            # nucleus at A (z=+R/2)
            v_aa_a += nij * _s_nuclear_attraction(H_EXP[i], H_EXP[j], 0.0, za, rp_aa, p)
            v_ab_a += nij * _s_nuclear_attraction(H_EXP[i], H_EXP[j], R * R, za, rp_ab, p)
            # nucleus at B (z=-R/2)
            v_aa_b += nij * _s_nuclear_attraction(H_EXP[i], H_EXP[j], 0.0, zb, rp_aa, p)
            v_ab_b += nij * _s_nuclear_attraction(H_EXP[i], H_EXP[j], R * R, zb, rp_ab, p)
    return (v_aa_a, v_ab_a, v_aa_b, v_ab_b)


def _contracted_two_electron(R: float) -> tuple[float, float, float]:
    """The three independent (ij|kl) integrals of the H2 minimal basis.

    Only three are unique under the 8-fold permutational symmetry of a
    2-center / 1-function-per-center STO-3G basis:
      (11|11)  one-center two-electron repulsion (on center A)
      (11|22)  inter-center Coulomb repulsion
      (12|12)  exchange-type integral
    """
    n = len(H_EXP)
    za = R / 2.0
    zb = -R / 2.0
    g_11_11 = g_11_22 = g_12_12 = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    coef = (
                        H_COEF[i]
                        * H_COEF[j]
                        * H_COEF[k]
                        * H_COEF[l]
                        * _norm_gaussian(H_EXP[i])
                        * _norm_gaussian(H_EXP[j])
                        * _norm_gaussian(H_EXP[k])
                        * _norm_gaussian(H_EXP[l])
                    )
                    # (11|11): all four on center A (or B)
                    g_11_11 += coef * _s_two_electron(
                        ea=H_EXP[i], eb=H_EXP[j], ec=H_EXP[k], ed=H_EXP[l],
                        za=za, zb=za, zc=za, zd=za,
                    )
                    # (11|22): i,j on A ; k,l on B
                    g_11_22 += coef * _s_two_electron(
                        ea=H_EXP[i], eb=H_EXP[j], ec=H_EXP[k], ed=H_EXP[l],
                        za=za, zb=za, zc=zb, zd=zb,
                    )
                    # (12|12): i,k on A ; j,l on B
                    g_12_12 += coef * _s_two_electron(
                        ea=H_EXP[i], eb=H_EXP[j], ec=H_EXP[k], ed=H_EXP[l],
                        za=za, zb=zb, zc=za, zd=zb,
                    )
    return (g_11_11, g_11_22, g_12_12)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def h2_integrals(R_bohr: float) -> dict[str, float]:
    """Return the full minimal-basis H2 STO-3G integral set at distance R.

    Parameters
    ----------
    R_bohr : float
        H-H distance in Bohr (a0).  Must be > 0.

    Returns
    -------
    dict with keys:
        S, T11, T12, V11A, V12A, V11B, V12B,
        g_11_11, g_11_22, g_12_12, enuc
    All in Hartree atomic units.  V11A = <chiA| -1/|r-RA| |chiA> etc.
    """
    if R_bohr <= 0:
        raise ValueError(f"H-H distance must be positive, got R={R_bohr:.4f} Bohr")

    S = _contracted_overlap(R_bohr)
    t11, t12 = _contracted_kinetic(R_bohr)
    v11a, v12a, v11b, v12b = _contracted_nuclear_attraction(R_bohr)
    g11_11, g11_22, g12_12 = _contracted_two_electron(R_bohr)
    enuc = 1.0 / R_bohr  # Z1*Z2/R  for two protons

    return {
        "S": S,
        "T11": t11,
        "T12": t12,
        "V11A": v11a,
        "V12A": v12a,
        "V11B": v11b,
        "V12B": v12b,
        "g_11_11": g11_11,
        "g_11_22": g11_22,
        "g_12_12": g12_12,
        "enuc": enuc,
        "R_bohr": R_bohr,
    }


def ao_core_hamiltonian(ints: dict[str, float]) -> np.ndarray:
    """Core (one-electron) Hamiltonian in AO basis, 2x2 symmetric matrix.

    h11 = T11 + V11A + V11B ; h12 = T12 + V12A + V12B ; h22 = h11.
    """
    h11 = ints["T11"] + ints["V11A"] + ints["V11B"]
    h12 = ints["T12"] + ints["V12A"] + ints["V12B"]
    return np.array([[h11, h12], [h12, h11]])


def ao_overlap_matrix(ints: dict[str, float]) -> np.ndarray:
    """2x2 AO overlap matrix with the contraction diagonal = 1."""
    S = ints["S"]
    return np.array([[1.0, S], [S, 1.0]])


def ao_two_electron_tensor(ints: dict[str, float]) -> np.ndarray:
    """Full two-electron tensor (ij|kl), chemists' notation, 2x2x2x2.

    Computed directly with the four-center primitive formula for **every**
    combination of the two AO centers, so three-center integrals such as
    (11|12) are included (they are non-zero in general).
    """
    R = ints["R_bohr"]
    center = {0: R / 2.0, 1: -R / 2.0}
    n_ao = 2
    n_prim = len(H_EXP)
    ten = np.zeros((n_ao, n_ao, n_ao, n_ao))
    # (ij|kl): these are OVER-barrier values; the loop is small (16 * 3^4).
    for i, j, k, l in np.ndindex(n_ao, n_ao, n_ao, n_ao):
        acc = 0.0
        for a, b, c, d in np.ndindex(n_prim, n_prim, n_prim, n_prim):
            coef = (
                H_COEF[a] * H_COEF[b] * H_COEF[c] * H_COEF[d]
                * _norm_gaussian(H_EXP[a])
                * _norm_gaussian(H_EXP[b])
                * _norm_gaussian(H_EXP[c])
                * _norm_gaussian(H_EXP[d])
            )
            acc += coef * _s_two_electron(
                ea=H_EXP[a], eb=H_EXP[b], ec=H_EXP[c], ed=H_EXP[d],
                za=center[i], zb=center[j], zc=center[k], zd=center[l],
            )
        ten[i, j, k, l] = acc
    return ten


# Reference values from the literature (Szabo & Ostlund) used by the tests.
REFERENCE_OVERLAP_1_4 = 0.6643
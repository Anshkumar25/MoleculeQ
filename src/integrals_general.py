"""General STO-3G one- and two-electron integral engine for s and p orbitals.

Supports H, He, Li, Be using McMurchie-Davidson Gaussian primitive evaluation
and Boys functions. Leaves `src/integrals_h2.py` completely untouched for H2.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.special import gammainc, gamma

# ---------------------------------------------------------------------------
# STO-3G Basis Parameters for H, He, Li, Be
# ---------------------------------------------------------------------------
H_EXP = [3.42525091, 0.62391373, 0.16885540]
H_COEF = [0.15432897, 0.53532814, 0.44463454]

HE_EXP = [6.36242139, 1.15892300, 0.31364979]
HE_COEF = [0.15432897, 0.53532814, 0.44463454]

LI_1S_EXP = [16.11957475, 2.93637175, 0.79465395]
LI_1S_COEF = [0.15432897, 0.53532814, 0.44463454]
LI_2SP_EXP = [0.63568758, 0.14754949, 0.04754564]
LI_2S_COEF = [-0.09996723, 0.39951283, 0.70011547]
LI_2P_COEF = [0.15591627, 0.60768372, 0.39195739]

BE_1S_EXP = [30.15612140, 5.49272300, 1.48624100]
BE_1S_COEF = [0.15432897, 0.53532814, 0.44463454]
BE_2SP_EXP = [1.31471700, 0.30537000, 0.09840900]
BE_2S_COEF = [-0.11651400, 0.44421600, 0.67756000]
BE_2P_COEF = [0.15591627, 0.60768372, 0.39195739]


# ---------------------------------------------------------------------------
# Mathematical primitives & Boys function
# ---------------------------------------------------------------------------
def boys(m: int, x: float) -> float:
    """Boys function F_m(x) = int_0^1 t^{2m} exp(-x t^2) dt."""
    if x < 1e-8:
        return 1.0 / (2 * m + 1) - x / (2 * m + 3)
    return 0.5 * (x ** (-(m + 0.5))) * gamma(m + 0.5) * gammainc(m + 0.5, x)


def norm_prim(alpha: float, l: int, m: int, n: int) -> float:
    """Normalization factor for a Cartesian primitive x^l y^m z^n exp(-alpha r^2)."""
    nom = (2.0 ** (2 * (l + m + n) + 1.5)) * (alpha ** (l + m + n + 1.5))
    den = (
        math.pi ** 1.5
        * math.factorial(2 * l)
        * math.factorial(2 * m)
        * math.factorial(2 * n)
        / ((4.0 ** (l + m + n)) * math.factorial(l) * math.factorial(m) * math.factorial(n))
    )
    return math.sqrt(nom / den)


class BasisFunction:
    """A contracted Gaussian basis function with angular momentum (l, m, n)."""

    def __init__(
        self,
        center: tuple[float, float, float] | np.ndarray,
        l: int,
        m: int,
        n: int,
        alphas: list[float],
        coefs: list[float],
    ):
        self.center = np.array(center, dtype=float)
        self.l = l
        self.m = m
        self.n = n
        self.alphas = np.array(alphas, dtype=float)
        norms = np.array([norm_prim(a, l, m, n) for a in self.alphas])
        self.coefs = np.array(coefs, dtype=float) * norms

        # Normalize contracted basis function to unit overlap
        s_self = 0.0
        for i in range(len(self.alphas)):
            for j in range(len(self.alphas)):
                s_self += self.coefs[i] * self.coefs[j] * prim_overlap(
                    self.alphas[i], self.l, self.m, self.n, self.center,
                    self.alphas[j], self.l, self.m, self.n, self.center
                )
        self.coefs /= math.sqrt(s_self)


# ---------------------------------------------------------------------------
# McMurchie-Davidson Expansion & Hermite Coulomb Integrals
# ---------------------------------------------------------------------------
def E_coefficients(l1: int, l2: int, a: float, b: float, Ax: float, Bx: float) -> np.ndarray:
    p = a + b
    Px = (a * Ax + b * Bx) / p
    XPA = Px - Ax
    XPB = Px - Bx
    XAB = Ax - Bx

    E = np.zeros((l1 + 1, l2 + 1, l1 + l2 + 1))
    E[0, 0, 0] = math.exp(-a * b / p * (XAB**2))

    for i in range(l1 + 1):
        for j in range(l2 + 1):
            if i == 0 and j == 0:
                continue
            for t in range(i + j + 1):
                val = 0.0
                if i > 0:
                    if t - 1 >= 0:
                        val += 0.5 / p * E[i - 1, j, t - 1]
                    val += XPA * E[i - 1, j, t]
                    if t + 1 <= i + j:
                        val += (t + 1) * E[i - 1, j, t + 1]
                elif j > 0:
                    if t - 1 >= 0:
                        val += 0.5 / p * E[i, j - 1, t - 1]
                    val += XPB * E[i, j - 1, t]
                    if t + 1 <= i + j:
                        val += (t + 1) * E[i, j - 1, t + 1]
                E[i, j, t] = val
    return E


def R_table(max_tuv: int, p: float, PC: np.ndarray) -> np.ndarray:
    X, Y, Z = PC
    RPC2 = X**2 + Y**2 + Z**2
    R = np.zeros((max_tuv + 1, max_tuv + 1, max_tuv + 1, max_tuv + 1))
    for N in range(max_tuv + 1):
        R[0, 0, 0, N] = ((-2 * p) ** N) * boys(N, p * RPC2)
    for N in range(max_tuv - 1, -1, -1):
        for i in range(max_tuv + 1 - N):
            for j in range(max_tuv + 1 - N - i):
                for k in range(max_tuv + 1 - N - i - j):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    if k > 0:
                        val = (k - 1) * R[i, j, k - 2, N + 1] if k > 1 else 0.0
                        val += Z * R[i, j, k - 1, N + 1]
                        R[i, j, k, N] = val
                    elif j > 0:
                        val = (j - 1) * R[i, j - 2, k, N + 1] if j > 1 else 0.0
                        val += Y * R[i, j - 1, k, N + 1]
                        R[i, j, k, N] = val
                    elif i > 0:
                        val = (i - 1) * R[i - 2, j, k, N + 1] if i > 1 else 0.0
                        val += X * R[i - 1, j, k, N + 1]
                        R[i, j, k, N] = val
    return R


# ---------------------------------------------------------------------------
# Primitive Integrals
# ---------------------------------------------------------------------------
def prim_overlap(a: float, l1: int, m1: int, n1: int, A: np.ndarray,
                 b: float, l2: int, m2: int, n2: int, B: np.ndarray) -> float:
    Ex = E_coefficients(l1, l2, a, b, A[0], B[0])
    Ey = E_coefficients(m1, m2, a, b, A[1], B[1])
    Ez = E_coefficients(n1, n2, a, b, A[2], B[2])
    return Ex[l1, l2, 0] * Ey[m1, m2, 0] * Ez[n1, n2, 0] * ((math.pi / (a + b)) ** 1.5)


def prim_kinetic(a: float, l1: int, m1: int, n1: int, A: np.ndarray,
                  b: float, l2: int, m2: int, n2: int, B: np.ndarray) -> float:
    p = a + b

    def s1d(i: int, j: int, ax: float, bx: float) -> float:
        if i < 0 or j < 0:
            return 0.0
        return E_coefficients(i, j, a, b, ax, bx)[i, j, 0] * math.sqrt(math.pi / p)

    def t1d(i: int, j: int, ax: float, bx: float) -> float:
        term1 = b * (2 * j + 1) * s1d(i, j, ax, bx)
        term2 = -2 * (b**2) * s1d(i, j + 2, ax, bx)
        term3 = -0.5 * j * (j - 1) * s1d(i, j - 2, ax, bx) if j >= 2 else 0.0
        return term1 + term2 + term3

    Sx = s1d(l1, l2, A[0], B[0])
    Sy = s1d(m1, m2, A[1], B[1])
    Sz = s1d(n1, n2, A[2], B[2])

    Tx = t1d(l1, l2, A[0], B[0])
    Ty = t1d(m1, m2, A[1], B[1])
    Tz = t1d(n1, n2, A[2], B[2])

    return Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz


def prim_nuc(a: float, l1: int, m1: int, n1: int, A: np.ndarray,
             b: float, l2: int, m2: int, n2: int, B: np.ndarray,
             C: np.ndarray, Zc: float) -> float:
    p = a + b
    P = (a * A + b * B) / p
    Ex = E_coefficients(l1, l2, a, b, A[0], B[0])
    Ey = E_coefficients(m1, m2, a, b, A[1], B[1])
    Ez = E_coefficients(n1, n2, a, b, A[2], B[2])

    max_tuv = (l1 + l2) + (m1 + m2) + (n1 + n2)
    R = R_table(max_tuv, p, P - C)

    val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                val += Ex[l1, l2, t] * Ey[m1, m2, u] * Ez[n1, n2, v] * R[t, u, v, 0]
    return -Zc * (2.0 * math.pi / p) * val


def prim_eri(a: float, l1: int, m1: int, n1: int, A: np.ndarray,
             b: float, l2: int, m2: int, n2: int, B: np.ndarray,
             c: float, l3: int, m3: int, n3: int, C: np.ndarray,
             d: float, l4: int, m4: int, n4: int, D: np.ndarray) -> float:
    p = a + b
    q = c + d
    alpha = p * q / (p + q)
    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q

    Ex = E_coefficients(l1, l2, a, b, A[0], B[0])
    Ey = E_coefficients(m1, m2, a, b, A[1], B[1])
    Ez = E_coefficients(n1, n2, a, b, A[2], B[2])

    Fx = E_coefficients(l3, l4, c, d, C[0], D[0])
    Fy = E_coefficients(m3, m4, c, d, C[1], D[1])
    Fz = E_coefficients(n3, n4, c, d, C[2], D[2])

    max_tuv = (l1 + l2 + l3 + l4) + (m1 + m2 + m3 + m4) + (n1 + n2 + n3 + n4)
    R = R_table(max_tuv, alpha, P - Q)

    val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                for tau in range(l3 + l4 + 1):
                    for mu in range(m3 + m4 + 1):
                        for nu in range(n3 + n4 + 1):
                            sign = (-1.0) ** (tau + mu + nu)
                            term = (
                                Ex[l1, l2, t]
                                * Ey[m1, m2, u]
                                * Ez[n1, n2, v]
                                * Fx[l3, l4, tau]
                                * Fy[m3, m4, mu]
                                * Fz[n3, n4, nu]
                                * sign
                                * R[t + tau, u + mu, v + nu, 0]
                            )
                            val += term
    return (2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))) * val


# ---------------------------------------------------------------------------
# Public Molecule Basis Builders & Integral Evaluator
# ---------------------------------------------------------------------------
def build_molecule_basis(symbols: tuple[str, ...], coords_bohr: tuple[tuple[float, float, float], ...]):
    """Build basis functions and nuclear list from atomic symbols and coords in Bohr."""
    basis: list[BasisFunction] = []
    nuclei: list[tuple[np.ndarray, float]] = []

    charge_map = {"H": 1.0, "He": 2.0, "Li": 3.0, "Be": 4.0}

    for sym, coord in zip(symbols, coords_bohr):
        c = np.array(coord, dtype=float)
        z_nuc = charge_map[sym]
        nuclei.append((c, z_nuc))

        if sym == "H":
            basis.append(BasisFunction(c, 0, 0, 0, H_EXP, H_COEF))
        elif sym == "He":
            basis.append(BasisFunction(c, 0, 0, 0, HE_EXP, HE_COEF))
        elif sym == "Li":
            basis.append(BasisFunction(c, 0, 0, 0, LI_1S_EXP, LI_1S_COEF))
            basis.append(BasisFunction(c, 0, 0, 0, LI_2SP_EXP, LI_2S_COEF))
            basis.append(BasisFunction(c, 1, 0, 0, LI_2SP_EXP, LI_2P_COEF))
            basis.append(BasisFunction(c, 0, 1, 0, LI_2SP_EXP, LI_2P_COEF))
            basis.append(BasisFunction(c, 0, 0, 1, LI_2SP_EXP, LI_2P_COEF))
        elif sym == "Be":
            basis.append(BasisFunction(c, 0, 0, 0, BE_1S_EXP, BE_1S_COEF))
            basis.append(BasisFunction(c, 0, 0, 0, BE_2SP_EXP, BE_2S_COEF))
            basis.append(BasisFunction(c, 1, 0, 0, BE_2SP_EXP, BE_2P_COEF))
            basis.append(BasisFunction(c, 0, 1, 0, BE_2SP_EXP, BE_2P_COEF))
            basis.append(BasisFunction(c, 0, 0, 1, BE_2SP_EXP, BE_2P_COEF))
        else:
            raise ValueError(f"Unsupported atom symbol: {sym}")

    return basis, nuclei


def compute_general_integrals(symbols: tuple[str, ...], coords_bohr: tuple[tuple[float, float, float], ...]):
    """Compute AO-basis integrals (S, H_core, g2_ao, E_nuc) for any supported molecule."""
    basis, nuclei = build_molecule_basis(symbols, coords_bohr)
    n = len(basis)

    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n))
    g2 = np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(n):
            b1, b2 = basis[i], basis[j]
            s_val = t_val = v_val = 0.0
            for a in range(len(b1.alphas)):
                for b in range(len(b2.alphas)):
                    c12 = b1.coefs[a] * b2.coefs[b]
                    s_val += c12 * prim_overlap(
                        b1.alphas[a], b1.l, b1.m, b1.n, b1.center,
                        b2.alphas[b], b2.l, b2.m, b2.n, b2.center,
                    )
                    t_val += c12 * prim_kinetic(
                        b1.alphas[a], b1.l, b1.m, b1.n, b1.center,
                        b2.alphas[b], b2.l, b2.m, b2.n, b2.center,
                    )
                    for C, Zc in nuclei:
                        v_val += c12 * prim_nuc(
                            b1.alphas[a], b1.l, b1.m, b1.n, b1.center,
                            b2.alphas[b], b2.l, b2.m, b2.n, b2.center,
                            C, Zc,
                        )
            S[i, j] = s_val
            T[i, j] = t_val
            V[i, j] = v_val

    h_core = T + V

    # 8-fold permutational symmetry for real 2-electron integrals:
    # (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (kl|ji) = (lk|ij) = (lk|ji)
    for i in range(n):
        for j in range(i + 1):
            ij = i * (i + 1) // 2 + j
            for k in range(n):
                for l in range(k + 1):
                    kl = k * (k + 1) // 2 + l
                    if ij >= kl:
                        b1, b2, b3, b4 = basis[i], basis[j], basis[k], basis[l]
                        eri_val = 0.0
                        for a in range(len(b1.alphas)):
                            for b in range(len(b2.alphas)):
                                for c in range(len(b3.alphas)):
                                    for d in range(len(b4.alphas)):
                                        coef = b1.coefs[a] * b2.coefs[b] * b3.coefs[c] * b4.coefs[d]
                                        eri_val += coef * prim_eri(
                                            b1.alphas[a], b1.l, b1.m, b1.n, b1.center,
                                            b2.alphas[b], b2.l, b2.m, b2.n, b2.center,
                                            b3.alphas[c], b3.l, b3.m, b3.n, b3.center,
                                            b4.alphas[d], b4.l, b4.m, b4.n, b4.center,
                                        )
                        # Set all 8 symmetry permutations
                        g2[i, j, k, l] = eri_val
                        g2[j, i, k, l] = eri_val
                        g2[i, j, l, k] = eri_val
                        g2[j, i, l, k] = eri_val
                        g2[k, l, i, j] = eri_val
                        g2[l, k, i, j] = eri_val
                        g2[k, l, j, i] = eri_val
                        g2[l, k, j, i] = eri_val

    # Nuclear repulsion energy
    e_nuc = 0.0
    num_nuc = len(nuclei)
    for i in range(num_nuc):
        for j in range(i + 1, num_nuc):
            c_i, z_i = nuclei[i]
            c_j, z_j = nuclei[j]
            dist = float(np.linalg.norm(c_i - c_j))
            e_nuc += (z_i * z_j) / dist

    return {
        "n_ao": n,
        "S": S,
        "h_core": h_core,
        "g2_ao": g2,
        "enuc": e_nuc,
    }

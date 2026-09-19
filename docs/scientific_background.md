# Scientific Background

*Companion to MoleculeQ — the physics and chemistry behind the app.*

MoleculeQ is an interactive demonstration of quantum molecular simulation.
This document explains, in increasing depth, *why* molecules are hard to
simulate, *how* quantum computers are being researched for that problem, and
exactly *what* the app computes.

## 1. Why molecular simulation matters

Almost all of chemistry lives in the electrons: where they sit, how they
move, how they are shared. Predicting the electronic state of a molecule lets
chemists reason about bond breaking, reaction pathways, spectroscopy,
catalysis, and materials properties — the backbone of pharmaceutical, energy
and materials R&D.

The molecule gives its energy through a **Hamiltonian** operator *Ĥ*
describing the kinetic energy of the electrons and their Coulomb
interactions with the nuclei and each other (for a fixed nuclear geometry,
the nuclei are treated as a frozen "external potential"). The **electronic
ground state** is the wavefunction that minimizes the energy expectation
value

```
E₀ = ⟨ψ|Ĥ|ψ⟩  (minimized over all electron wavefunctions ψ)
```

and E₀ is the number chemistry is built on.

## 2. Why it gets hard as systems grow

A wavefunction for *n* electrons is not a point but a *distribution over
configurations*. Representing it exactly with *m* basis functions requires
roughly *mⁿ* amplitude coefficients. Because the numbers multiply, cost grows
**exponentially** with the number of particles. This scaling is the reason a
small molecule can be treated exactly with classical computers but larger,
interesting ones quickly exceed what exact classical diagonalization can do —
and why approximating methods (Hartree–Fock, DFT, coupled-cluster, …) exist.

MoleculeQ includes both sides of this story: it uses an exact classical
reference **and** a quantum algorithm, so the two can be compared live.

## 3. Why quantum computers

A quantum computer stores its state in a superposition over *2ⁿ* amplitudes
simultaneously. A simulation that would need *mⁿ* classical numbers can, in
principle, be encoded in the amplitudes of *O(n·log m)* qubits. Simulating a
quantum system on a quantum computer is therefore a *natural* encoding,
rather than a brute-force one.

This is famously captured by Feynman's observation that "nature isn't
classical, dammit, and if you want to make a simulation of nature, you'd
better make it quantum mechanical." Whether this advantage becomes practical
is an open research question; MoleculeQ demonstrates the *construction* of
such a simulation on the simplest molecule, without pretending the scaling
question is resolved.

## 4. The molecule: H₂

Hydrogen is the simplest molecule: two nuclei, two electrons. It is the
it plays the role of `print("Hello, World!")` in quantum chemistry — because:

- its physics is rich enough to exercise every step of a real quantum-chemistry workflow,
- its minimal-basis problem is tiny enough to be solved exactly as a benchmark,
- and yet it shows the full structure: molecular orbitals, electron correlation, and a binding curve.

### The minimum basis (STO-3G)

Each hydrogen atom contributes one **Slater-type orbital**, itself fit by
three Gaussians. The molecule then has 2 spatial orbitals and 2 electrons.

### The RHF step

Restricted Hartree–Fock self-consistently finds the best single-determinant
wavefunction for the two electrons. It yields molecular orbitals and the
best **mean-field** energy. The app layers the exact answer (FCI) above it,
so the gap between mean-field and correlated energy is visible.

### Full Configuration Interaction (FCI)

FCI is the exact electronic energy *within the chosen basis*: all possible
electron configurations — the occupied state, two single excitations, and the
double excitation — are diagonalized. For H₂/STO-3G, half of the electron
correlation comes from the double excitation alone.

## 5. Mapping chemistry to qubits

### Fermionic Hamiltonian

The second-quantized electronic Hamiltonian is written in terms of creation
and annihilation operators:

```
Ĥ = Σᵢⱼ hᵢⱼ aᵢ⁺aⱼ  +  ½ Σᵢⱼₖₗ gᵢⱼₖₗ aᵢ⁺aⱼ⁺aₖaₗ  +  E_nuc
```

with *h* the one-electron integrals and *g* the two-electron integrals in the
molecular-orbital basis. This operator acts on a space of 4 sites
(2 spin-up × 2 spin-down orbitals) for H₂.

### Fermion → qubit mapping

Electron operators obey anticommutation relations that map onto Pauli
operators via a mapping such as **Jordan–Wigner** or **Parity**. MoleculeQ
uses the **Parity mapping**, which reduces the qubit count for H₂ to **2
qubits** by encoding parity (occupation symmetry) directly in the qubit
labels. The result is a compact Pauli operator,

```
Ĥ_qubit = g₀ I + g₁ Z₀ + g₂ Z₁ + g₃ Z₀Z₁ + g₄ X₀X₁  (+ constant offset)
```

whose spectrum, up to the constant offset, is exactly the electronic spectrum
of the molecule.

## 6. VQE — the quantum algorithm

The **Variational Quantum Eigensolver** is a hybrid quantum-classical
algorithm:

1. **Ansatz** — a parameterized quantum circuit prepares a trial state
   |ψ(θ)⟩. MoleculeQ offers two:
   - `ry`: a hardware-efficient `TwoLocal(ry, rz; cx; linear)` circuit — a
     generic ansatz, no chemistry knowledge encoded;
   - `uccsd`: unitary coupled-cluster singles & doubles — starts from the
     Hartree–Fock state and exponentiates excitations, encoding chemistry.
2. **Measurement** — the energy ⟨ψ(θ)|Ĥ|ψ(θ)⟩ is estimated with an exact
   **statevector estimator** (the full *2ⁿ*-amplitude wavefunction is
   evolved in memory — a real quantum-mechanical simulation, just without
   hardware noise).
3. **Optimization** — a classical optimizer (SLSQP or COBYLA) proposes new θ
   to lower the energy, then repeats from step 2.

### The variational principle

For any trial state, E(θ) ≥ E₀. VQE can therefore *approach* the ground-state
energy from above, but never overshoot below it. How close it gets depends on
the ansatz expressivity, the optimizer, and the initialization. MoleculeQ
ships the classical exact answer to make that gap visible — normally VQE
lands on it to within ~10⁻⁶ Ha for the H₂/ry ansatz.

## 7. The energy convention

The pipeline separates three quantities explicitly and honestly:

| Quantity | Meaning |
|---|---|
| E_electronic | eigenvalue of the electronic (qubit) Hamiltonian — the quantum result |
| E_nuclear-repulsion, E_nuc | constant ∝ 1/R between the two nuclei |
| E_total | E_electronic + E_nuc — the binding curve quantity |

At short R, E_nuc blows up (protons repel); at large R the curve flattens as
the molecule dissociates. The minimum of E_total vs R is the model's
**equilibrium bond length** — reported as such, with the caveat that it is an
STO-3G value, not the high-accuracy library value (~0.741 Å experiment).

## 8. What the app shows — and what it does not

**Shows:** a genuine end-to-end quantum chemistry workflow on 2 qubits —
geometry → integrals → fermionic Hamiltonian → qubit Hamiltonian → VQE →
classical benchmark → binding curve — every number produced by the real
pipeline.

**Does not show:** drug discovery, quantum advantage, or fault-tolerant
scaling. This is an educational demonstration; classical exact methods beat
this demo's VQE on runtime for a 2-qubit problem by orders of magnitude, and
that is the honest framing the app presents.

## References

1. A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love,
   A. Aspuru-Guzik, J. L. O'Brien, *A variational eigenvalue solver on a
   photonic quantum processor*, **Nature Communications 5**, 4213 (2014).
2. Y. Cao, J. Romero, J. P. Olson, M. Degroote, P. D. Johnson, M. Kieferová,
   I. D. Kivlichan, T. Menke, B. Peropadre, N. P. D. Sawaya, S. Sim, L.
   Veis, A. Aspuru-Guzik, *Quantum Chemistry in the Age of Quantum
   Computing*, **Chemical Reviews 119**, 10856 (2019).
3. P. J. J. O'Malley *et al.*, *Scalable quantum simulation of molecular
   energies*, **Physical Review X 6**, 031007 (2016) — the canonical H₂-on-
   hardware result.
4. D. A. McQuarrie, *Quantum Chemistry* — textbook background on the
   electronic Hamiltonian, HF and CI.
5. Qiskit / Qiskit Nature documentation: https://qiskit.org and
   https://qiskit.org/ecosystem/nature/
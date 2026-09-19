# Expo Demo Script

*Step-by-step expo presentation for MoleculeQ, written against the real numbers
the pipeline produces on this machine. Verify each number live before the
demo — the app may drift slightly with package versions.*

---

## Part A — The live demo (audience-facing, ~6 minutes)

**Setup.** Have the app running and on the **Home** page with the default
geometry (R ≈ 0.735 Å). Ensure the machine is on a stable desktop, no
screensaver, full-screen the browser.

### Step 1 — The million-dollar question *(30 s)*

> "Why do we care? Chemistry is *electrons* — where they sit decides how
> drugs bind, how materials behave. For big molecules even the world's best
> classical computers can't track the electron wavefunction exactly. It goes
> exponential."

Point at the H₂ drawing: *"This is our molecule. Two protons, two electrons."*

### Step 2 — A molecule you can break (Stretch the bond) *(45 s)*

Slowly drag the **bond-distance slider** from 0.735 Å out to ~3 Å and back.

> "Watch the picture — it isn't a photo, it's computed from the slider. Pull
> the nuclei apart and the molecule is headed toward hand-dissociation:
> two separate hydrogen atoms."

Move the slider to **0.6 Å** (squeezed).

> "Squeeze them and the protons start repelling — so the energy goes up."

### Step 3 — The honest ask *(20 s)*

> "Every geometry turns into a *math problem*: find the lowest energy the
> electrons can have. For a real chem-rival molecule that's many thousands
> of coupled equations. We'll solve the smallest honest version of it — on a
> quantum algorithm."

### Step 4 — Press ▶ Run simulation *(30 s)*

Air-quote the VQE as you click:

> "This is **VQE** — the Variational Quantum Eigensolver. A trial
> wavefunction makes a guess, a quantum step measures its energy, a classical
> optimizer asks it to guess better, and we loop. That hybrid rhythm — quantum
> + classical — is the heart of near-term quantum computing."

The spinner runs for about a second, then success.

### Step 5 — The answer is real, and benchmarked *(45 s)*

Open **Simulate**. Show:

- the **Hamiltonian** (2 qubits, the Z₀, Z₁, Z₀Z₁, X₀X₁ terms);
- the **circuit** (the actual ansatz with its tuned parameters);
- the **convergence chart** — "watch the line settle";
- the energy metrics: *electronic −1.857 Ha; total −1.137 Ha.*

> "A real quantum estimate. But a scientist never trusts a number. Where's
> the check?"

Switch to **Results**.

### Step 6 — Benchmarked, honestly *(45 s)*

Point at the comparison table:

> "Classical exact diagonalization of the same 2-qubit Hamiltonian:
> **−1.857276 Ha**. VQE: **−1.857274 Ha**. The difference is **0.000002
> Hartree** — the bracket on the 'variational principle'. VQE can't go below
> the truth; it just gets as close as the optimizer allows."

> "Do we beat the laptop? No. This is *validation*, not a race — and the
> point is the pipeline, built for the day the problem is too big for that
> exact-diagonalization trick."

### Step 7 — The whole energy curve (wow, but optional) *(60 s)*

On **Results**, check *"Compute the energy scan"* (15 points takes ~30–60 s).
Show the progress bar, then the curve with the red star at the minimum.

> "Every point is the full pipeline running again — chemicals → matrices →
> qubits → VQE. The curve bottoms out near **0.81 Å**. That's this model's
> equilibrium bond length." *(Science caveat, see Step 8.)*

### Step 8 — The scientist's caveat *(30 s)*

> "Note the star is the *STO-3G model's* favorite bond length, not the
> textbook 0.741 Å. Quantum chemistry is always a model: here we used the
> simplest basis set, and the app tells you so. Honesty about the model is
> the healthiest habit a computational scientist has."

### Step 9 — Close the loop *(20 s)*

> "MoleculeQ is a tiny molecule, a simulated chip, and an always-on classical
> referee. It doesn't discover drugs or prove quantum advantage — it shows
> the *mechanism*: geometry to Hamiltonian to qubits to hybrid optimization,
> then a check. That mechanism is what quantum chemistry research is building
> on."

---

## Part B — 12-section presentation outline

A slide-deck skeleton that follows the demo (for a longer talk or panel):

1. **Ask** — "Where do electrons sit, and why does it matter?" *(drugs,
   materials, catalysis)*
2. **The Blocker** — exact wavefunctions grow exponentially; classical limits.
3. **The Idea** — simulate quantum mechanics with quantum machines (Feynman).
4. **The Molecule** — H₂: simplest real chemistry; a molecule, not a toy.
5. **From geometry to operator** — RHF → integrals → fermionic Hamiltonian.
6. **Onto qubits** — fermion→qubit mapping; Parity → the 2-qubit Pauli operator.
7. **VQE** — the hybrid loop; ansatz, measurement, optimizer; variational principle.
8. **The Demo video / live run** — the workflow end-to-end (Part A).
9. **Results we trust** — exact/FCI benchmark agreement (~10⁻⁶ Ha), convergence plots.
10. **Honesty wall** — limitations: minimal basis, simulator, no advantage claim; STO-3G equilibrium vs experiment.
11. **Why it matters** — same pipeline concepts scale toward molecules beyond exact classical reach; near-term hardware direction.
12. **Q&A best-hits** —
    - "What qubit count?" → 2 qubits (parity) / 4 fermionic sites.
    - "Real hardware?" → same algorithm; this run is exact simulation for honesty & speed.
    - "Can I run it?" → `streamlit run app.py`; see README.
    - "Does it beat classical?" → no, deliberately; it's validated *against* classical.

---

## Part C — Speaker cheat-sheet (numbers)

| Quantity | Value (this machine, default geometry) |
|---|---|
| Bond distance (default) | 0.735 Å |
| Chemistry backend | local-sto3g-integrals (Windows; PySCF path on unix) |
| Fermionic sites / qubits | 4 sites → 2 qubits (Parity), 5 Pauli terms |
| VQE electronic energy | −1.857274 Ha |
| Exact (eigenevalue) electronic | −1.857276 Ha |
| FCI electronic energy | −1.857276 Ha |
| Nuclear repulsion | +0.719969 Ha |
| Total energy | −1.137305 Ha |
| VQE − exact | ≈ +1.5×10⁻⁶ Ha (optimizer tolerance) |
| Scan minimum (uccsd curve) | ≈ 0.807 Å (STO-3G model) |
| Experimental equilibrium | ≈ 0.741 Å (used only for the caveat) |

> ⚠️ **Before the day:** run full pipeline + a scan once to confirm these
> match on the expo machine; package versions can change the last digits.
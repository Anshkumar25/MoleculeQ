# MoleculeQ 🧪

> **Understanding molecules today — unlocking scientific possibilities tomorrow.**

MoleculeQ is an interactive, beginner-friendly web app for **quantum
molecular simulation**. You move the two hydrogen nuclei of an H₂ molecule
apart or together, press *Run*, and the app genuinely builds the molecular
Hamiltonian for that geometry, maps it onto **qubits**, and estimates the
molecule's ground-state electronic energy with the **Variational Quantum
Eigensolver (VQE)** — the same hybrid quantum-classical algorithm that
near-term quantum processors run.

No quantum computing background is required to use the app.

## Why this matters

Molecular simulation underpins drug discovery, materials science, and
catalysis. Exact many-electron solutions on classical computers scale
**exponentially** with system size, which is why quantum computers — which
natively represent waves of probability — are being researched for chemistry.
MoleculeQ demonstrates the *idea* end-to-end on the simplest possible
molecule (H₂), honestly: a small-scale educational demo that runs on a
laptop, not a claim of quantum advantage.

## What it does

| Feature | Detail |
|---|---|
| **Interactive molecule** | Two hydrogen nuclei drawn live from the bond-distance slider |
| **Real quantum pipeline** | RHF → MO integrals → fermionic Hamiltonian → **qubit** Hamiltonian → VQE |
| **Circuit visualization** | The actual ansatz circuit (with optimal parameters bound in) |
| **Convergence analysis** | Per-iteration energy history of the optimizer |
| **Energy curve** | Bond-distance vs energy, recomputed point-by-point by VQE |
| **Classical benchmark** | Exact diagonalization of the qubit Hamiltonian + FCI reference |
| **Learn page** | Plain-English explanations of every concept |

### Energy convention (kept honest)

- **E_electronic** — the electronic Hamiltonian eigenvalue VQE estimates (the interesting, quantum part).
- **E_nuclear-repulsion** — a constant added by convention.
- **E_total** = E_electronic + E_nuclear-repulsion.

All energies are reported in **Hartree** (1 Ha = 27.2114 eV).

## Tech stack

- **Frontend:** [Streamlit](https://streamlit.io) (`app.py`)
- **Quantum:** [Qiskit](https://qiskit.org) + [Qiskit Nature](https://qiskit.org/ecosystem/nature/) + Qiskit Algorithms
- **Visualization:** matplotlib (molecule/circuit), Plotly (charts)
- **Chemistry integrals:** bundled dependency-free STO-3G H₂ integral engine (used on Windows where PySCF has no wheels); PySCF-backed path available where PySCF is installed
- **Exact references:** SciPy eigendecomposition + a full-configuration-interaction routine

## Install

Requires **Python 3.11**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
venv\Scripts\activate.bat            # cmd
\.venv\Scripts\activate.ps1          # PowerShell

# 2. Install dependencies
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the printed URL (default `http://localhost:8501`).

**Typical workflow**

1. **Home** — read what the app computes; watch the molecule update as you
   drag the bond-distance slider in the sidebar.
2. Adjust the **ansatz**, **optimizer**, and **iterations** in the sidebar.
3. Press **▶ Run simulation** — VQE runs for ~1 s on a laptop.
4. **Simulate** — inspect the qubit Hamiltonian, the ansatz circuit, and the
   convergence chart.
5. **Results** — compare against the classical references and (optionally)
   compute the full bond-distance energy curve.
6. **Learn** — every concept explained without jargon.

## Project layout

```
moleculeq/
├── app.py                    # Streamlit application (thin UI layer only)
├── .streamlit/config.toml    # Visual theme + server settings
├── requirements.txt
├── src/
│   ├── molecule_builder.py   # geometry, units (Å ↔ Bohr), validation
│   ├── hartree_fock.py       # RHF + FCI electronic energies (STO-3G)
│   ├── integrals_h2.py       # dependency-free STO-3G H₂ integral engine
│   ├── hamiltonian.py        # fermionic → qubit Hamiltonian (parity/JW)
│   ├── quantum_solver.py     # ansatz, optimizer, VQE runner (statevector)
│   ├── classical_reference.py# exact diagonalization + FCI benchmark
│   ├── energy_scan.py        # bond-distance energy scan (real pipeline)
│   ├── visualization.py      # molecule diagram, circuit, charts
│   └── validation.py         # shared input validation
├── tests/                    # pytest suite (61 tests)
└── docs/                     # design + background docs
```

## Methodology

For every geometry R, the pipeline is (nothing is faked, ever):

1. **Geometry** — build H₂ with nuclei at z = ±R/2 (Å), validate the range.
2. **RHF** — solve restricted Hartree–Fock for molecular orbitals in STO-3G.
3. **Fermionic Hamiltonian** — build the one-/two-electron operators.
4. **Qubit mapping** — Parity mapping (+ 2-qubit reduction gives 2 qubits).
5. **VQE** — minimize ⟨ψ(θ)|H|ψ(θ)⟩ over the ansatz parameters θ with a
   classical optimizer, using the exact statevector estimator.
6. **Classical reference** — exact diagonalization and FCI of the same
   problem for an always-on benchmark.

The **variational principle** guarantees VQE's energy is ≥ the true
ground-state eigenvalue; the comparison table shows exactly how close it
got.

## Testing

```bash
pytest tests/ -q
```

The suite validates geometry building, energy conventions
(E_total = E_elec + E_nuc), the VQE–exact agreement, scan point counts and
partial-failure tolerance, and all input validators. One test is
`xfail`-marked because the installed Aer's deprecated V1 Estimator cannot be
driven by this VQE version (the app offers only the working statevector
backend).

## Scientific honesty

- This is a **small-scale, educational demonstration** on H₂ in a minimal
  (STO-3G) basis. It does **not** discover drugs, beat a laptop, or prove
  quantum advantage.
- Classical methods remain extremely effective for molecules this small;
  whether quantum computers help for real molecules is an **open question**
  the demo deliberately does not prejudge.
- Energies are computed from the real quantum pipeline — never fabricated —
  and always benchmarked against a classical reference.

## Limitations

- Single molecule (H₂), minimal basis, simulated exactly (no device noise).
- The equilibrium bond length reported is that of the **STO-3G model**, which
  differs slightly from high-accuracy or experimental values.
- No hardware, no error mitigation, no scaling story — deliberately.

## Future work

- More molecules + basis sets (HeH⁺, LiH, H₂O; STO-3G → 6-31G).
- A flight toward real hardware via Qiskit Runtime backends.
- Error-mitigation and shot-noise demonstrations once EstimatorV2 lands.
- Pedagogy extras: interactive Hamiltonian terms, ansatz comparison, noise sliders.

## References

- Peruzzo et al., *A variational eigenvalue solver on a photonic quantum processor*, Nat. Commun. **5**, 4213 (2014).
- Cao et al., *Quantum chemistry in the age of quantum computing*, Chem. Rev. **119**, 10856 (2019).
- Qiskit & Qiskit Nature documentation. See `docs/scientific_background.md`.
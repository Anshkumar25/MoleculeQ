# Architecture

*How MoleculeQ is built: modules, data flow, and the numbers that move between them.*

MoleculeQ separates concerns cleanly: a **thin Streamlit UI** drives a
**pure-Python science layer** (`src/`) that never imports Streamlit. The
science layer is fully testable headlessly (the pytest suite runs with no UI).

```
┌─────────────┐  widget settings   ┌──────────────┐   dataclasses    ┌─────────────┐
│  app.py     │ ──────────────────▶│   src/*      │ ───────────────▶ │  src/viz    │
│ (Streamlit) │ ◀──────────────────│  (pipeline)  │ ◀─ figures ───── │  + plotly   │
└─────────────┘  cached results    └──────────────┘                  └─────────────┘
```

## Layer 1 — The Streamlit UI (`app.py`)

`app.py` is the only file with a UI dependency. It is intentionally *thin*:
every scientific action is delegated to `src/`; the UI never computes an
energy itself.

- **Sidebar** (`render_sidebar`) — controls: molecule (fixed to H₂ this
  version), bond distance slider, ansatz, optimizer, iterations, backend,
  Run button, Reset.
- **`cached_pipeline`** — `@st.cache_data`-wrapped closure around the
  single-geometry pipeline. Inputs are primitives (hashable), so a rerun with
  unchanged settings is served instantly from cache.
- **Pages** — Home (intro + live molecule), Simulate (Hamiltonian, circuit,
  convergence, energies), Results (benchmark table + optional energy scan),
  Learn (educational content).
- **Error surfacing** — `_safe()` catches domain errors and shows them
  as friendly messages, with a collapsible developer-details expander (no
  raw tracebacks by default).
- **`.streamlit/config.toml`** — dark scientific theme + server defaults.

## Layer 2 — The science pipeline (`src/`)

Each module is a small, documented unit. Data for a geometry flows top to
bottom:

```
molecule_builder  →  hartree_fock (RHF + FCI)
        │               │
        │               ▼
        └─────────▶  hamiltonian  (fermionic op + parity mapping)
                                │
                                ▼
                    quantum_solver (VQE)
                                │
              ┌─────────────────┼───────────────┐
              ▼                 ▼               ▼
      classical_reference   energy_scan   visualization
      (exact + FCI)         (the curve)   (figures)
```

### Module contract (what each returns)

| Module | Input → Output | Notes |
|---|---|---|
| `molecule_builder` | bond Å → `MolecularGeometry` | validates range; Å-convention; Å↔Bohr helpers |
| `integrals_h2` | geometry → one/two-electron integrals | dependency-free STO-3G engine (Windows default) |
| `hartree_fock` | geometry → `{E_elec, E_nuc, E_total, C, h_core, g2_ao, …}`; `fci_electronic_energy` | mean-field + exact-in-basis energy |
| `hamiltonian` | geometry → `QuibitProblem` | backends: PySCF if importable, else local engine; Parity (default) or JW mapping; 2 qubits / 5 Pauli terms for H₂ |
| `quantum_solver` | qubit_op + settings → `VQEResult` | statevector VQE (exact); SLSQP/COBYLA; `ry`/`uccsd` ansätze; returns energy history, circuit depth, params |
| `classical_reference` | geometry (+ qubit_op) → `ReferenceResult` | exact diagonalization + FCI/HF; E_total conventions |
| `energy_scan` | [R_min, R_max], N → `ScanResult` | runs the full pipeline per point; per-point try/except → partial failure tolerance; never fabricates |
| `visualization` | geometry/circuit/results → figures | matplotlib (Agg) for molecule+circuit, plotly for convergence/scan |
| `validation` | any user input → validated value | single source of friendly error messages |

### `src/hamiltonian.py` — the two chemistry backends

`ACTIVE_CHEMISTRY_BACKEND` is detected once at import:

- **`pyscf`** when PySCF imports (Linux/macOS typically) — drives the real
  `PySCFDriver` from Qiskit Nature.
- **`local-sto3g-integrals`** otherwise (this Windows machine) — calls the
  bundled, dependency-free integral engine and RHF. The active backend is
  always reported to the user in the UI (honesty about *what* computed the
  numbers).

Both paths produce equivalent STO-3G H₂ electronic data; only the integral
evaluation layer differs.

### `src/quantum_solver.py` — the VQE design

- **Estimator:** `StatevectorEstimator` (exact, deterministic). An Aer
  backend option exists but is `xfail`-marked in tests because this
  environment's Aer 0.15 ships only a deprecated V1 Estimator that the
  installed qiskit-algorithms VQE cannot drive — the UI, accordingly, offers
  the statevector backend.
- **Ansätze:** `ry` → `TwoLocal(ry, rz; cx; linear, reps=1)` (general);
  `uccsd` → Nature's `UCCSD` on a Hartree–Fock initial state (chemically
  motivated). UCCSD is used for scan points by default because it is
  essentially exact for 2e-/2-orbital H₂ across the curve.
- **Optimizer:** SLSQP (deterministic, default) or COBYLA. Both are genuine
  qiskit-algorithms optimizers; per-iteration callback builds `history`.
- **Reproducibility:** a fixed default seed drives initial-point generation.

## Layer 3 — Caching & performance

- **Single point** — `st.cache_data` keyed on (bond, ansatz, optimizer,
  maxiter, backend, mapping). One VQE ≈ 0.2–1 s on a laptop.
- **Scan** — cached in `st.session_state` keyed by the scanned window; opt-in
  in the UI (a 15-point scan ≈ 30–60 s, shown via `st.progress`).

The scientific content of every return value is *never* stored as a
hard-coded result: cached objects are live pipeline outputs (dataclasses
holding computed energies/operators).

## Energy conventions (single source of truth)

`E_total = E_electronic + E_nuclear_repulsion`, with the nuclear-repulsion
term kept separate and documented everywhere it appears (solver result,
reference result, scan point, UI metrics). Units are Hartree (Ha) throughout;
UI shows eV equivalents (1 Ha = 27.2114 eV).

## Testing strategy

- `tests/conftest.py` puts the repo root on `sys.path`.
- Unit tests cover geometry/validation (fast, pure Python), VQE validity and
  reproducibility, energy conventions, scan failure tolerance, and the
  exact-reference agreement (the scientific heart: exact qubit-Hamiltonian
  eigenvalue matches FCI to ~10⁻⁶ Ha).
- A blind-spot is the AppTest-bare-mode Windows native segfault seen when a
  *long scan* runs inside Streamlit's test harness; the pipeline itself is
  verified standalone (smoke scripts in `scratch/`) and every app page is
  verified exception-free via AppTest. One manual click-through in a browser
  is the recommended final check.
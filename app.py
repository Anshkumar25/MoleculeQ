"""MoleculeQ — Streamlit dashboard.

Explore molecules through quantum computing: build H2, change the bond
distance, run a genuine Variational Quantum Eigensolver (VQE) on the mapped
Hamiltonian, and compare with classical references.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import warnings

import numpy as np

import streamlit as st

warnings.filterwarnings("ignore")  # quiet benign scipy/qiskit warnings

# ruff: noqa: E402  - keep imports clustered for readability

from src.molecule_builder import (
    MAX_BOND_ANGSTROM,
    MIN_BOND_ANGSTROM,
    HEH_MIN_BOND_ANGSTROM, HEH_MAX_BOND_ANGSTROM,
    LIH_MIN_BOND_ANGSTROM, LIH_MAX_BOND_ANGSTROM,
    BEH2_MIN_BOND_ANGSTROM, BEH2_MAX_BOND_ANGSTROM,
    MoleculeBuilderError,
    build_hydrogen,
    build_molecule,
)
from src.hamiltonian import build_qubit_operator
from src.quantum_solver import run_vqe
from src import validation
from src.validation import (
    validate_bond,
    validate_scan_window,
    validate_ansatz,
    validate_optimizer,
    validate_backend,
    validate_mapping,
)

st.set_page_config(
    page_title="MoleculeQ — Quantum Molecular Simulation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

MOLECULES = {
    "Hydrogen (H₂)": "H2",
    "Helium Hydride Cation (HeH⁺)": "HeH+",
    "Lithium Hydride (LiH)": "LiH",
    "Beryllium Dihydride (BeH₂)": "BeH2",
}

PAGES = ["🏠 Home", "🧪 Simulate", "📊 Results", "🎓 Learn"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _safe(fn):
    """Run f() and surface friendly errors to the UI (no raw tracebacks)."""
    try:
        return fn()
    except (ValueError, MoleculeBuilderError) as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong: {exc}")
        with st.expander("Developer details"):
            import traceback

            st.code(traceback.format_exc())
    return None


@st.cache_data(show_spinner=False)
def cached_pipeline(
    molecule_key: str,
    bond_angstrom: float,
    ansatz: str,
    optimizer: str,
    maxiter: int,
    backend: str,
    mapping: str,
) -> dict | None:
    """Run the full single-geometry pipeline; returns a serializable dict."""
    return _run_pipeline(
        molecule_key, bond_angstrom, ansatz, optimizer, maxiter, backend, mapping
    )


def _run_pipeline(molecule_key, bond, ansatz, optimizer, maxiter, backend, mapping):
    geom = build_molecule(molecule_key, bond)
    prob = build_qubit_operator(geom, mapping=mapping)
    qm = None
    n_spatial = 2
    if geom.name == "BeH2":
        n_spatial = 3

    if ansatz == "uccsd":
        from qiskit_nature.second_q.mappers import ParityMapper

        qm = ParityMapper(num_particles=prob.num_particles)
    vqe = run_vqe(
        prob.qubit_op,
        num_particles=prob.num_particles,
        nuclear_repulsion_energy=prob.nuclear_repulsion_energy,
        ansatz=ansatz,
        optimizer=optimizer,
        maxiter=maxiter,
        backend=backend,
        qubit_mapper=qm,
        num_spatial_orbitals=n_spatial if ansatz == "uccsd" else None,
    )
    from src.classical_reference import build_reference

    ref = build_reference(geom, qubit_op=prob.qubit_op)
    return {
        "geom": geom,
        "prob": prob,
        "vqe": vqe,
        "ref": ref,
        "ansatz_circuit": _make_bound_circuit(prob, vqe, ansatz, qm, n_spatial),
    }


def _make_bound_circuit(prob, vqe, ansatz, qubit_mapper, n_spatial=2):
    """Ansatz with its optimal parameters bound in (for display)."""
    from src.quantum_solver import make_ansatz

    ans = make_ansatz(
        prob.num_qubits,
        num_spatial_orbitals=n_spatial if ansatz == "uccsd" else None,
        num_particles=prob.num_particles,
        qubit_mapper=qubit_mapper,
        ansatz=ansatz,
    )
    try:
        params = list(ans.parameters)
        value_dict = dict(zip(params, vqe.parameter_values))
        return ans.assign_parameters(value_dict)
    except Exception:  # noqa: BLE001
        return ans


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## 🧪 MoleculeQ")
        st.caption("Understand molecules through quantum computing")

        st.markdown("---")
        st.markdown("### Molecule")
        molecule = st.selectbox("Species", list(MOLECULES.keys()))
        mol_key = MOLECULES[molecule]

        if mol_key == "H2":
            min_b, max_b, val_b = MIN_BOND_ANGSTROM, MAX_BOND_ANGSTROM, 0.735
        elif mol_key == "HeH+":
            min_b, max_b, val_b = HEH_MIN_BOND_ANGSTROM, HEH_MAX_BOND_ANGSTROM, 0.772
        elif mol_key == "LiH":
            min_b, max_b, val_b = LIH_MIN_BOND_ANGSTROM, LIH_MAX_BOND_ANGSTROM, 1.595
        else:
            min_b, max_b, val_b = BEH2_MIN_BOND_ANGSTROM, BEH2_MAX_BOND_ANGSTROM, 1.326

        bond_angstrom = st.slider(
            "Bond distance (Å)",
            min_value=min_b,
            max_value=max_b,
            value=val_b,
            step=0.005,
            help=f"Relevant bond length for {molecule}. Changing this "
            "rebuilds the molecular Hamiltonian and re-runs the algorithm.",
        )

        st.markdown("---")
        st.markdown("### Quantum algorithm")
        ansatz = st.selectbox(
            "Ansatz",
            ["ry (hardware-efficient)", "uccsd (quantum chemistry)"],
            index=0,
            help="The parameterized circuit used to represent the trial "
            "ground-state wavefunction.",
        )
        ansatz_key = "ry" if ansatz.startswith("ry") else "uccsd"
        optimizer = st.selectbox(
            "Optimizer", ["SLSQP", "COBYLA"], index=0
        )
        maxiter = st.slider("Optimizer iterations", 50, 800, 300, step=10)
        backend = st.selectbox(
            "Simulator backend",
            ["statevector (exact)"],
            index=0,
            help="The statevector estimator evolves the full quantum "
            "wavefunction exactly and deterministically — the honest default "
            "for this small demonstration.",
        )
        backend_key = "statevector"

        st.markdown("---")
        action = st.button("▶  Run simulation", type="primary", use_container_width=True)
        if st.button("↺ Reset to defaults", use_container_width=True):
            for key in ("last_bond", "results"):
                st.session_state.pop(key, None)
            st.rerun()

    return dict(
        molecule_key=MOLECULES[molecule],
        bond=bond_angstrom,
        ansatz=ansatz_key,
        optimizer=optimizer,
        maxiter=maxiter,
        backend=backend_key,
        run_pressed=action,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home(settings):
    st.title("MoleculeQ — Explore Molecules Through Quantum Computing")
    st.caption("_Understand molecules today; unlock scientific possibilities tomorrow._")

    col_l, col_r = st.columns([3, 2], gap="large")
    with col_l:
        st.markdown(
            """
**What does this app do?**

It lets you build a tiny real molecule — **hydrogen (H₂)** — and then push its
two nuclei together or apart. For each geometry it does something genuinely
quantum: it writes down the molecular *Hamiltonian* (the electron energy
operator), maps it onto **qubits**, and estimates the molecule's lowest
electronic energy with the **Variational Quantum Eigensolver (VQE)** — a
hybrid quantum-classical algorithm genuine quantum processors run, here
simulated honestly on a statevector simulator.

**What you will see:**

- A live picture of H₂ that changes as you move the bond-distance slider.
- The qubit Hamiltonian and the quantum circuit that represents the trial state.
- The VQE energy converging iteration by iteration.
- A bond-distance ↔ energy curve computed point-by-point by real quantum
  simulation.
- A classical reference (exact diagonalization + full CI) that the app is
  graded against — because an honest small demo always tells you how close
  it got.
            """
        )
        st.info(
            "**Scope, honestly stated:** this is a small-scale educational "
            "demonstration with a minimal basis set, run on a simulator. It "
            "does *not* discover drugs, beat a laptop, or prove quantum "
            "advantage — it shows *how* quantum simulation is designed."
        )

    with col_r:
        geom = build_molecule(settings["molecule_key"], settings["bond"])
        from src.visualization import plot_molecule

        fig = plot_molecule(geom, energy_text=f"{settings['bond']:.3f} Å")
        st.pyplot(fig)
        st.caption(
            f"Current geometry: {settings['bond']:.3f} Å  ·  "
            f"{settings['ansatz'].upper()} ansatz · {settings['optimizer']} optimizer"
        )

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Molecule", settings["molecule_key"], f"{geom.num_atoms} atom(s)")
    c2.metric("Basis set", "STO-3G", "minimal basis")
    c3.metric("Algorithm", "VQE", "hybrid quantum-classical")
    c4.metric("Charge", f"{geom.charge}", f"multiplicity {geom.multiplicity}")

    st.markdown(
        """
### How to use
1. Use the **sidebar** to select a molecule, set the bond distance and algorithm options.
2. Press **▶ Run simulation** to run VQE on the current geometry.
3. Open the **Simulate** and **Results** pages to see the circuit, the
   convergence and the comparison with classical references.
4. Open **Learn** for plain-English explanations of every concept.
        """
    )


def page_simulate(settings):
    st.title("Simulation — one molecule, one quantum solve")
    st.caption("Everything on this page is produced live by the running pipeline.")

    run = settings.pop("run_pressed", False)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Next run settings:** {settings['molecule_key']} · bond={settings['bond']:.3f} Å · "
        f"{settings['ansatz']} · {settings['optimizer']} · {settings['maxiter']} iters"
    )

    if run:
        with st.spinner("Running VQE on the quantum simulator…"):
            result = _safe(
                lambda: cached_pipeline(
                    settings["molecule_key"],
                    settings["bond"],
                    settings["ansatz"],
                    settings["optimizer"],
                    settings["maxiter"],
                    settings["backend"],
                    "parity",
                )
            )
        if result is not None:
            st.session_state["results"] = result
            st.session_state["last_bond"] = settings["bond"]
            st.success("Simulation complete. See Results page for analysis.")
        else:
            st.stop()

    result = st.session_state.get("results")
    bond_now = settings["bond"]

    if result is None:
        st.warning(
            "No simulation has been run yet. Use the sidebar to configure a "
            "geometry and press **▶ Run simulation**."
        )
        return

    changed = abs(result["geom"].bond_distance - bond_now) > 1e-9 or result["geom"].name != settings["molecule_key"]
    if changed:
        st.warning(
            f"You selected {settings['molecule_key']} at {bond_now:.3f} Å but the last run was for "
            f"{result['geom'].name} at {result['geom'].bond_distance:.3f} Å. Press ▶ Run to re-solve."
        )

    geom, prob, vqe, ref = result["geom"], result["prob"], result["vqe"], result["ref"]

    st.write("### Molecular geometry")
    c = st.columns(3)
    c[0].metric("Species", geom.name)
    c[1].metric("Bond length", f"{geom.bond_distance:.3f} Å")
    c[2].metric("Charge / spin", f"{geom.charge} / {geom.multiplicity}")

    st.write("### Qubit Hamiltonian")
    c = st.columns(2)
    with c[0]:
        st.markdown(
            f"- **Mapping:** {prob.mapping}\n"
            f"- **Qubits:** {prob.num_qubits}\n"
            f"- **Pauli terms:** {len(prob.qubit_op)}\n"
            f"- **Fermionic terms:** {len(prob.fermionic_op)}"
        )
    with c[1]:
        st.code(str(prob.qubit_op), language="text")

    st.write("### Variational ansatz")
    c = st.columns(3)
    c[0].metric("Ansatz", vqe.ansatz_name.upper())
    c[1].metric("Circuit depth", vqe.circuit_depth)
    c[2].metric("Optimizer calls", vqe.cost_function_calls)

    st.subheader("Quantum circuit (with optimal parameters)")
    from src.visualization import circuit_figure

    fig, is_summary = circuit_figure(result["ansatz_circuit"])
    st.pyplot(fig)
    if is_summary:
        st.caption("Visual circuit exceeds display limits; showing a metric summary.")

    st.write("### VQE progress and final energy")
    from src.visualization import plot_convergence

    if vqe.history:
        st.plotly_chart(plot_convergence(vqe.history), use_container_width=True)
    else:
        st.info("No per-iteration history was captured by this optimizer.")

    c = st.columns(3)
    c[0].metric(
        "Electronic energy",
        f"{vqe.electronic_energy:.6f} Ha",
        f"{vqe.electronic_energy * 27.2114:.4f} eV",
    )
    c[1].metric(
        "Nuclear repulsion",
        f"{vqe.nuclear_repulsion_energy:.6f} Ha",
        "+ (constant, added by convention)",
    )
    c[2].metric(
        "Total molecular energy",
        f"{vqe.total_energy:.6f} Ha",
        f"{vqe.total_energy * 27.2114:.4f} eV",
    )
    st.caption(
        "E_total = E_electronic + E_nuclear-repulsion. All energies in Hartree "
        "(1 Ha = 27.2114 eV)."
    )


def page_results(settings):
    st.title("Results — quantum vs classical, and the full energy landscape")

    result = st.session_state.get("results")
    if result is None:
        st.warning("Run a simulation first (Simulate page).")
        return

    geom, prob, vqe, ref = result["geom"], result["prob"], result["vqe"], result["ref"]

    st.write("### Single-point comparison")
    from src.visualization import comparison_rows

    rows = comparison_rows(ref, vqe)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.write("### Energy conventions")
    c = st.columns(3)
    c[0].metric("E_electronic (VQE)", f"{vqe.electronic_energy:.6f} Ha")
    c[1].metric("E_nuclear repulsion", f"{vqe.nuclear_repulsion_energy:.6f} Ha")
    c[2].metric("E_total", f"{vqe.total_energy:.6f} Ha")

    st.markdown(
        "> **What this does and does not prove.** The VQE estimate matches the "
        "classical exact eigenvalue of the qubit Hamiltonian — validating the "
        "pipeline and the algorithm. It does **not** show quantum advantage: "
        "for small qubit systems a laptop diagonalizes the operator instantly. The "
        "comparison is a *benchmark of the quantum optimisation*, not a race."
    )

    st.write("### Bond-distance energy curve")
    run_scan_here = st.checkbox(
        "Compute the energy scan for this molecule (re-runs VQE at many distances)",
        value=False,
        help="Each point is computed by the real simulation pipeline from "
        "scratch. On a typical laptop this takes 30–60 seconds.",
    )
    if run_scan_here:
        st.caption(
            "Each point is computed by the real simulation pipeline — chemistry "
            "integrals → qubit Hamiltonian → VQE — from scratch."
        )
        mol_key = settings["molecule_key"]
        if mol_key == "H2":
            s_min, s_max, d_min, d_max = MIN_BOND_ANGSTROM, MAX_BOND_ANGSTROM, 0.6, 3.5
        elif mol_key == "HeH+":
            s_min, s_max, d_min, d_max = HEH_MIN_BOND_ANGSTROM, HEH_MAX_BOND_ANGSTROM, 0.5, 2.5
        elif mol_key == "LiH":
            s_min, s_max, d_min, d_max = LIH_MIN_BOND_ANGSTROM, LIH_MAX_BOND_ANGSTROM, 1.0, 3.5
        else:
            s_min, s_max, d_min, d_max = BEH2_MIN_BOND_ANGSTROM, BEH2_MAX_BOND_ANGSTROM, 0.9, 3.0

        left, right, npts = st.columns([1, 1, 1])
        r_min = left.slider("Scan min (Å)", s_min, 2.0, d_min, 0.05)
        r_max = right.slider("Scan max (Å)", 1.0, s_max, d_max, 0.05)
        n = npts.slider("Points", 5, 30, 15)

        try:
            a = validate_scan_window(r_min, r_max, n, molecule_name=mol_key)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        scan = st.session_state.get("scan")
        scan_key = f"{mol_key}-{a[0]:.3f}-{a[1]:.3f}-{a[2]}"
        if scan is None or st.session_state.get("scan_key") != scan_key:
            progress = st.progress(0.0, text=f"Scanning {mol_key} bond distances…")
            scan = _run_scan_safely(a[0], a[1], a[2], progress, molecule_name=mol_key)
            if scan is not None:
                st.session_state["scan"] = scan
                st.session_state["scan_key"] = scan_key
        else:
            st.success("Scan already cached for this window.")

        if scan is not None:
            from src.energy_scan import ScanResult
            from src.visualization import plot_scan_curve

            st.plotly_chart(plot_scan_curve(scan, y_quantity="total"), use_container_width=True)
            low = scan.lowest_energy_point()
            if low is not None:
                c = st.columns(3)
                c[0].metric("Lowest computed energy", f"{low.vqe_total:.5f} Ha")
                c[1].metric("At bond distance", f"{low.R_angstrom:.3f} Å")
                c[2].metric("Failed points", f"{scan.failed_points} / {scan.total_points}")
            if scan.failed_points:
                st.warning(
                    f"{scan.failed_points} scan point(s) failed and were skipped: "
                    f"{'; '.join(scan.failure_reasons[:3])}"
                )
            st.markdown(
                "The **minimum** is the equilibrium bond length *of this STO-3G "
                "model* (a slightly different value from high-accuracy coupled-cluster "
                "or experimental curves)."
            )


def _run_scan_safely(r_min, r_max, n, progress, molecule_name="H2"):
    from src.energy_scan import run_scan

    try:
        scan = run_scan(
            r_min,
            r_max,
            n,
            molecule_name=molecule_name,
            progress_callback=lambda done, total: progress.progress(done / total),
        )
        progress.empty()
        return scan
    except Exception as exc:  # noqa: BLE001
        progress.empty()
        st.error(str(exc))
        return None


def page_learn():
    st.title("Learn — understand every concept in plain English")

    sections = {
        "What is a molecule?": """
A molecule is a group of atoms held together by shared or transferred
electrons. For chemists the interesting part is *where the electrons sit*:
that decides the molecule's shape, colour, reactivity — everything. The
electron arrangement is called the molecule's **electronic structure**.
""",
        "What is an electron?": """
An electron is a tiny, negatively charged particle that moves around atoms.
Electrons are *quantum particles*: they don't have fixed positions like tiny
planets — they are described by a wavefunction, a wave of probability spread
through space. Because of this, a "snapshot" of a molecule is really a
description of these waves.
""",
        "What is a qubit?": """
A bit is a 0 or a 1. A **qubit** (quantum bit) is a physical two-state system
that can be in a mixture of both states at once — a *superposition* — and can
be entangled with other qubits. Two qubits can hold information that scales
like 2² classical values; n qubits like 2ⁿ. That exponential space is what
makes quantum computers interesting for simulating quantum systems like
molecules.
""",
        "What is a molecular Hamiltonian?": """
The Hamiltonian is the *energy operator* of the molecule. For a fixed
nuclear geometry it describes how the kinetic energy of the electrons and
the electron–electron / electron–nucleus interactions combine. The electron
wavefunction that minimises ⟨ψ|H|ψ⟩ is the molecule's **ground state**, and
that minimum is the **ground-state electronic energy** — the number our app
estimates. (A separate, constant term accounts for nucleus–nucleus
repulsion.)
""",
        "What is VQE?": """
**Variational Quantum Eigensolver** — a hybrid quantum-classical algorithm.
It has three actors:

1. a *parameterized* quantum circuit (the **ansatz**), prepared by qubits;
2. a *quantum* step: measure ⟨ψ(θ)|H|ψ(θ)⟩ for the current parameters θ;
3. a *classical* step: an optimizer nudges θ to lower the measured energy.

Repeat. The energy can never go below the true ground-state energy
(**variational principle**), so the lowest value we reach is our best
estimation. VQE is a leading candidate for chemistry on near-term quantum
hardware.
""",
        "Why is quantum simulation important?": """
Exact many-electron solutions on classical computers scale exponentially
with system size. Quantum computers natively store exponentially many
amplitudes, so simulating the many-body *wavefunction* is their natural
strength. This project demonstrates the *idea* on the simplest possible
molecule; the same building blocks scale (in principle) towards molecules
that defeat exact classical diagonalization.
""",
        "What can this project show — and not show?": """
**Shows:** a real end-to-end quantum chemistry workflow — build a molecule in
code, map its Hamiltonian to qubits, run VQE, compare with an exact classical
answer — on H₂ in a minimal basis.

**Does not show:** drug discovery, beating classical computers, or quantum
advantage. Our H₂/STO-3G problem is solvable exactly on any laptop in
milliseconds; the point is the *pipeline*, not the record.
""",
        "Why does the energy curve have a minimum?": """
At very short bond lengths the two protons repel each other strongly and the
total energy is high. Pull them apart and attraction binds the molecule — the
energy drops to a minimum at the **equilibrium bond length**. Pull further
and the molecule dissociates into two separate hydrogen atoms, so the energy
levels off. The location of that minimum depends on the model (basis set); we
show the STO-3G prediction.
""",
    }

    for title, text in sections.items():
        with st.expander(title, expanded=(title == "What is a molecule?")):
            st.markdown(text)

    st.markdown("---")
    st.markdown(
        "### Quick glossary\n"
        "- **Hartree (Ha)**: atomic unit of energy (1 Ha = 27.2114 eV).\n"
        "- **STO-3G**: a minimal Gaussian basis set — 3 Gaussians per atomic orbital.\n"
        "- **FCI (full configuration interaction)**: exact electronic energy in a given basis.\n"
        "- **Fermion → qubit mapping**: how electron operators are translated to Pauli operators.\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    settings = render_sidebar()

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Navigate")
        page = st.radio("Page", PAGES, label_visibility="collapsed")

    settings["run_pressed"] = settings.get("run_pressed", False)

    if page == PAGES[0]:
        page_home(settings)
    elif page == PAGES[1]:
        page_simulate(settings)
    elif page == PAGES[2]:
        page_results(settings)
    else:
        page_learn()


if __name__ == "__main__":
    main()
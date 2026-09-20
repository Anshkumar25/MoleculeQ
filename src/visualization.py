"""Visualization helpers for MoleculeQ.

All plotting is lazy: functions return matplotlib or plotly figure objects
that the Streamlit layer renders. No figures are opened at module import
time, so this module stays testable in a headless environment.
"""

from __future__ import annotations

import matplotlib
import warnings as _warnings

matplotlib.use("Agg")  # no display backend: figures are rendered to bytes/UI

import numpy as np


# ---------------------------------------------------------------------------
# Environment-safe imports
# ---------------------------------------------------------------------------
def _safe_import_plotly():
    import plotly.graph_objects as go

    return go


def _safe_import_qiskit_draw():
    from qiskit import QuantumCircuit

    return QuantumCircuit


# ---------------------------------------------------------------------------
# Molecule diagram
# ---------------------------------------------------------------------------
def plot_molecule(geometry, energy_text: str | None = None, ax=None):
    """Draw a 2D molecular diagram that reflects the current geometry and bond distance."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    color_map = {
        "H": "#1f7abf",
        "He": "#9c27b0",
        "Li": "#e65100",
        "Be": "#43a047",
    }
    edge_map = {
        "H": "#0e4a7a",
        "He": "#6a1b9a",
        "Li": "#b71c1c",
        "Be": "#1b5e20",
    }

    if geometry.num_atoms == 2:
        (x1, y1, z1), (x2, y2, z2) = geometry.coords[0], geometry.coords[1]
        sym1, sym2 = geometry.symbols[0], geometry.symbols[1]

        c1 = color_map.get(sym1, "#1f7abf")
        e1 = edge_map.get(sym1, "#0e4a7a")
        c2 = color_map.get(sym2, "#1f7abf")
        e2 = edge_map.get(sym2, "#0e4a7a")

        r1 = 0.28 if sym1 in ("Li", "Be") else (0.24 if sym1 == "He" else 0.22)
        r2 = 0.22

        ax.add_patch(Circle((z1, 0), r1, color=c1, ec=e1, lw=2, zorder=3))
        ax.add_patch(Circle((z2, 0), r2, color=c2, ec=e2, lw=2, zorder=3))

        # bond
        ax.plot([z1, z2], [0, 0], color="#888", lw=3, zorder=1)

        # electron density hints
        for zc in np.linspace(z1, z2, 9):
            ax.add_patch(Circle((zc, 0.34), 0.09, color="#ffb84d", alpha=0.5, zorder=2))
            ax.add_patch(Circle((zc, -0.34), 0.09, color="#ffb84d", alpha=0.5, zorder=2))

        ax.set_xlim(min(z1, z2) - 0.8, max(z1, z2) + 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect("equal")
        ax.axis("off")

        mid = (z1 + z2) / 2
        ax.annotate(
            "",
            xy=(z1, -0.55),
            xytext=(z2, -0.55),
            arrowprops=dict(arrowstyle="<->", lw=1.2, color="#333"),
        )
        ax.text(mid, -0.72, f"{geometry.bond_distance:.3f} Å", ha="center", fontsize=11)

        if energy_text:
            ax.text(mid, 0.42, energy_text, ha="center", fontsize=10, color="#1b5e20")

        ax.text(z1, 0.0, sym1, ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=4)
        ax.text(z2, 0.0, sym2, ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=4)
        return fig

    # 3-atom linear arrangement: BeH2 (H - Be - H)
    z_be = geometry.coords[0][2]
    z_h1 = geometry.coords[1][2]
    z_h2 = geometry.coords[2][2]

    c_be, e_be = color_map["Be"], edge_map["Be"]
    c_h, e_h = color_map["H"], edge_map["H"]

    ax.add_patch(Circle((z_be, 0), 0.28, color=c_be, ec=e_be, lw=2, zorder=3))
    ax.add_patch(Circle((z_h1, 0), 0.22, color=c_h, ec=e_h, lw=2, zorder=3))
    ax.add_patch(Circle((z_h2, 0), 0.22, color=c_h, ec=e_h, lw=2, zorder=3))

    ax.plot([z_h2, z_h1], [0, 0], color="#888", lw=3, zorder=1)

    for zc in np.linspace(z_h2, z_h1, 15):
        ax.add_patch(Circle((zc, 0.34), 0.09, color="#ffb84d", alpha=0.5, zorder=2))
        ax.add_patch(Circle((zc, -0.34), 0.09, color="#ffb84d", alpha=0.5, zorder=2))

    ax.set_xlim(min(z_h1, z_h2) - 0.8, max(z_h1, z_h2) + 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.annotate(
        "",
        xy=(z_be, -0.55),
        xytext=(z_h1, -0.55),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="#333"),
    )
    ax.text((z_be + z_h1) / 2, -0.72, f"R = {geometry.bond_distance:.3f} Å", ha="center", fontsize=11)

    if energy_text:
        ax.text(z_be, 0.42, energy_text, ha="center", fontsize=10, color="#1b5e20")

    ax.text(z_be, 0.0, "Be", ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=4)
    ax.text(z_h1, 0.0, "H", ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=4)
    ax.text(z_h2, 0.0, "H", ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=4)
    return fig


# ---------------------------------------------------------------------------
# VQE convergence
# ---------------------------------------------------------------------------
def plot_convergence(history: list[dict], title: str = "VQE convergence"):
    """Plotly figure of optimizer iteration vs energy."""
    go = _safe_import_plotly()
    if not history:
        raise ValueError("No convergence history to plot.")

    iters = [h["iter"] for h in history]
    energies = [h["energy"] for h in history]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=energies,
            mode="lines+markers",
            marker=dict(size=4, color="#1f7abf"),
            line=dict(width=2, color="#1f7abf"),
            name="energy at iteration",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Cost-function evaluation (iteration)",
        yaxis_title="Electronic energy (Hartree)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Scan curve
# ---------------------------------------------------------------------------
def plot_scan_curve(scan, *, y_quantity: str = "total") -> "object":
    """Plot the bond-distance vs energy curve from a ScanResult.

    y_quantity : 'total' (default; physical curve) or 'electronic'.
    Marks the lowest continuous curve point and labels the model-dependent
    minimum.
    """
    go = _safe_import_plotly()
    from .energy_scan import ScanResult

    if not isinstance(scan, ScanResult):
        raise TypeError("scan must be a ScanResult")

    ylabel = "Total molecular energy (Hartree)" if y_quantity == "total" else "Electronic energy (Hartree)"

    points_total = [
        p for p in scan.points if p.vqe_total is not None and p.vqe_electronic is not None
    ]
    if not points_total:
        raise ValueError("No successful scan points to plot.")

    if y_quantity == "total":
        y_vqe = np.array([p.vqe_total for p in points_total])
        y_exact = np.array(
            [(p.exact_electronic + p.nuclear_repulsion) for p in points_total]
        )
    else:
        y_vqe = np.array([p.vqe_electronic for p in points_total])
        y_exact = np.array([p.exact_electronic for p in points_total])

    x = np.array([p.R_angstrom for p in points_total])
    mask = ~np.isnan(y_exact)
    if mask.all() is False:
        # still keep arrays aligned
        y_exact = np.where(np.isfinite(y_exact), y_exact, np.nan)

    order = np.argsort(x)
    x, y_vqe, y_exact = x[order], y_vqe[order], y_exact[order]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_exact,
            mode="lines",
            line=dict(width=1.5, dash="dash", color="#999"),
            name="Classical exact reference",
            hovertemplate="R= %{x:.3f} Å<br>E= %{y:.5f} Ha<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_vqe,
            mode="lines+markers",
            marker=dict(size=6, color="#1f7abf"),
            line=dict(width=2.2, color="#1f7abf"),
            name="VQE (quantum estimate)",
            hovertemplate="R= %{x:.3f} Å<br>E= %{y:.5f} Ha<extra></extra>",
        )
    )

    # discrete grid minimum (of the VQE curve)
    imin = int(np.nanargmin(y_vqe))
    fig.add_trace(
        go.Scatter(
            x=[x[imin]],
            y=[y_vqe[imin]],
            mode="markers",
            marker=dict(size=11, symbol="star", color="#d32f2f"),
            name="Lowest computed energy",
        )
    )
    fig.add_annotation(
        x=x[imin],
        y=y_vqe[imin],
        text=f"model equilibrium ≈ {x[imin]:.2f} Å",
        showarrow=True,
        arrowhead=2,
        ay=-40,
        font=dict(color="#d32f2f"),
    )

    fig.update_layout(
        title="Electronic energy of H₂ as a function of bond distance",
        xaxis_title="H–H bond distance (Å)",
        yaxis_title=ylabel,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Quantum circuit rendering
# ---------------------------------------------------------------------------
def circuit_figure(circuit, max_qubits_renderable: int = 6):
    """Render a QuantumCircuit to a matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")

    if circuit is None:
        raise ValueError("No circuit to visualize.")

    nq = circuit.num_qubits

    if nq > max_qubits_renderable:
        return _circuit_summary(circuit), True

    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")

            fig = circuit.draw(
                output="mpl",
                fold=-1,
                scale=0.8
            )

        if fig is None:
            raise ValueError("Qiskit returned no matplotlib figure.")

        return fig, False

    except Exception as exc:
        raise RuntimeError(
            f"Qiskit matplotlib circuit rendering failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _circuit_summary(circuit):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, min(6, 1.2 + 0.8 * circuit.num_qubits)))
    ax.axis("off")
    title = [
        f"Quantum circuit summary",
        f"Qubits: {circuit.num_qubits}   Gates: {circuit.size()}   Depth: {circuit.depth()}",
        f"Parameterized gates: {sum(1 for instr in circuit.data if instr.operation.is_parameterized())}",
    ]
    ax.text(0.02, 0.98, "\n".join(title), va="top", ha="left", fontsize=12)

    # simple gate-column listing
    gate_counts: dict = {}
    for instr in circuit.data:
        name = instr.operation.name
        gate_counts[name] = gate_counts.get(name, 0) + 1
    if gate_counts:
        listing = "  |  ".join(f"{name} × {c}" for name, c in sorted(gate_counts.items()))
        ax.text(0.02, 0.55, listing, va="top", ha="left", fontsize=11, color="#333")
    ax.text(0.02, 0.28, "Circuit too large to draw graphically; metrics above.", va="top", fontsize=10, color="#888")
    return fig


# ---------------------------------------------------------------------------
# Quantum-classical comparison
# ---------------------------------------------------------------------------
def comparison_rows(result, vqe) -> list[dict]:
    """Rows describing the quantum vs classical comparison for the UI table."""
    rows = [
        {
            "Quantity": "Quantum-estimated electronic energy (VQE)",
            "Value": f"{vqe.electronic_energy:.6f} Ha",
        },
        {
            "Quantity": "Classical exact qubit-Hamiltonian eigenvalue",
            "Value": f"{result.E_exact_elec:.6f} Ha" if result.E_exact_elec is not None else "—",
        },
        {
            "Quantity": "Classical FCI electronic energy (chemistry benchmark)",
            "Value": f"{result.E_fci_elec:.6f} Ha",
        },
    ]
    if result.E_exact_elec is not None:
        d = vqe.electronic_energy - result.E_exact_elec
        rows.append({"Quantity": "VQE − exact qubit-Hamiltonian", "Value": f"{d:+.2e} Ha"})
        rows.append(
            {
                "Quantity": "VQE quality",
                "Value": "Matches classical reference" if abs(d) < 1e-3 else "Above reference (see note)",
            }
        )
    return rows
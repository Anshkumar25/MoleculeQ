"""Tests for new molecules: HeH+, LiH, and BeH2."""

import pytest
import numpy as np

from src.molecule_builder import (
    build_molecule,
    build_heh_plus,
    build_lih,
    build_beh2,
    MoleculeBuilderError,
    HEH_MIN_BOND_ANGSTROM, HEH_MAX_BOND_ANGSTROM,
    LIH_MIN_BOND_ANGSTROM, LIH_MAX_BOND_ANGSTROM,
    BEH2_MIN_BOND_ANGSTROM, BEH2_MAX_BOND_ANGSTROM,
)
from src.integrals_general import compute_general_integrals
from src.hartree_fock_general import (
    solve_rhf_general,
    fci_electronic_energy_general,
    get_molecule_active_spaces,
)
from src.hamiltonian import build_qubit_operator
from src.classical_reference import fci_reference, build_reference
from src.quantum_solver import run_vqe
from src.energy_scan import run_scan
from src.validation import validate_bond_for_molecule


class TestMoleculeBuilderNew:
    def test_build_heh_plus(self):
        geom = build_heh_plus(0.772)
        assert geom.name == "HeH+"
        assert geom.symbols == ("He", "H")
        assert geom.num_atoms == 2
        assert geom.charge == 1
        assert geom.multiplicity == 1
        assert geom.bond_distance == pytest.approx(0.772)

    def test_build_lih(self):
        geom = build_lih(1.595)
        assert geom.name == "LiH"
        assert geom.symbols == ("Li", "H")
        assert geom.num_atoms == 2
        assert geom.charge == 0
        assert geom.multiplicity == 1
        assert geom.bond_distance == pytest.approx(1.595)

    def test_build_beh2(self):
        geom = build_beh2(1.326)
        assert geom.name == "BeH2"
        assert geom.symbols == ("Be", "H", "H")
        assert geom.num_atoms == 3
        assert geom.charge == 0
        assert geom.multiplicity == 1
        assert geom.coords[0] == (0.0, 0.0, 0.0)
        assert geom.coords[1] == (0.0, 0.0, 1.326)
        assert geom.coords[2] == (0.0, 0.0, -1.326)

    def test_generic_dispatcher(self):
        g1 = build_molecule("H2", 0.735)
        assert g1.name == "H2"

        g2 = build_molecule("HeH+", 0.772)
        assert g2.name == "HeH+"

        g3 = build_molecule("LiH", 1.595)
        assert g3.name == "LiH"

        g4 = build_molecule("BeH2", 1.326)
        assert g4.name == "BeH2"

    def test_validation_ranges(self):
        with pytest.raises(MoleculeBuilderError):
            validate_bond_for_molecule("HeH+", HEH_MIN_BOND_ANGSTROM - 0.05)
        with pytest.raises(MoleculeBuilderError):
            validate_bond_for_molecule("LiH", LIH_MAX_BOND_ANGSTROM + 0.1)
        with pytest.raises(MoleculeBuilderError):
            validate_bond_for_molecule("BeH2", BEH2_MIN_BOND_ANGSTROM - 0.01)


class TestElectronicStructureNew:
    def test_heh_plus_rhf_and_fci(self):
        geom = build_heh_plus(0.772)
        rhf = solve_rhf_general(geom)
        assert rhf["converged"] is True
        e_fci = fci_electronic_energy_general(geom, rhf=rhf)
        assert e_fci <= rhf["E_elec"] + 1e-6

    def test_lih_rhf_and_fci(self):
        geom = build_lih(1.595)
        rhf = solve_rhf_general(geom)
        assert rhf["converged"] is True
        e_fci = fci_electronic_energy_general(geom, rhf=rhf)
        assert e_fci <= rhf["E_elec"] + 1e-6

    def test_beh2_rhf_and_fci(self):
        geom = build_beh2(1.326)
        rhf = solve_rhf_general(geom)
        assert rhf["converged"] is True
        e_fci = fci_electronic_energy_general(geom, rhf=rhf)
        assert e_fci <= rhf["E_elec"] + 1e-6


class TestQubitMappingAndVQE:
    def test_heh_plus_qubit_operator(self):
        geom = build_heh_plus(0.772)
        prob = build_qubit_operator(geom, mapping="parity")
        assert prob.num_qubits == 2
        vqe = run_vqe(
            prob.qubit_op,
            num_particles=prob.num_particles,
            nuclear_repulsion_energy=prob.nuclear_repulsion_energy,
            ansatz="ry",
            optimizer="SLSQP",
            maxiter=100,
        )
        assert vqe.converged is True

    def test_lih_qubit_operator_and_vqe(self):
        geom = build_lih(1.595)
        prob = build_qubit_operator(geom, mapping="parity")
        assert prob.num_qubits == 2
        vqe = run_vqe(
            prob.qubit_op,
            num_particles=prob.num_particles,
            nuclear_repulsion_energy=prob.nuclear_repulsion_energy,
            ansatz="ry",
            optimizer="SLSQP",
            maxiter=100,
        )
        assert vqe.converged is True

    def test_beh2_qubit_operator_and_vqe(self):
        geom = build_beh2(1.326)
        prob = build_qubit_operator(geom, mapping="parity")
        assert prob.num_qubits == 4
        vqe = run_vqe(
            prob.qubit_op,
            num_particles=prob.num_particles,
            nuclear_repulsion_energy=prob.nuclear_repulsion_energy,
            ansatz="ry",
            optimizer="SLSQP",
            maxiter=100,
        )
        assert vqe.converged is True


class TestEnergyScanNew:
    def test_heh_plus_scan(self):
        scan = run_scan(0.5, 1.5, 3, molecule_name="HeH+")
        assert scan.total_points == 3
        assert scan.completed_points == 3

    def test_lih_scan(self):
        scan = run_scan(1.2, 2.0, 3, molecule_name="LiH")
        assert scan.total_points == 3
        assert scan.completed_points == 3

    def test_beh2_scan(self):
        scan = run_scan(1.0, 1.8, 3, molecule_name="BeH2")
        assert scan.total_points == 3
        assert scan.completed_points == 3

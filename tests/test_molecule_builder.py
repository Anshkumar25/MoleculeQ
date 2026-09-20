"""Tests for the molecule builder: geometry, units, validation, errors."""

import numpy as np
import pytest

from src.molecule_builder import (
    MAX_BOND_ANGSTROM,
    MIN_BOND_ANGSTROM,
    MoleculeBuilderError,
    MolecularGeometry,
    bond_angstrom_to_bohr,
    build_hydrogen,
    validate_bond_distance,
)


class TestBuildHydrogen:
    def test_valid_geometry(self):
        geom = build_hydrogen(0.735)
        assert geom.name == "H2"
        assert geom.symbols == ("H", "H")
        assert geom.num_atoms == 2
        assert geom.charge == 0
        assert geom.multiplicity == 1
        assert geom.bond_distance == pytest.approx(0.735)

    def test_nuclei_symmetric_about_origin(self):
        geom = build_hydrogen(1.0)
        z1, z2 = geom.to_nuclei_z()
        assert z1 == pytest.approx(+0.5)
        assert z2 == pytest.approx(-0.5)

    def test_bond_along_z_axis(self):
        geom = build_hydrogen(2.0)
        coords = np.array(geom.coords)
        # x and y are exactly zero at both nuclei
        assert np.allclose(coords[:, :2], 0.0)
        # separation equals requested distance
        assert np.abs(coords[0, 2] - coords[1, 2]) == pytest.approx(2.0)

    def test_atomic_string_format(self):
        geom = build_hydrogen(0.735)
        s = geom.atomic_string()
        assert s.count("H") == 2
        # two entries separated by '; '
        assert len(s.split("; ")) == 2


class TestUnits:
    def test_angstrom_to_bohr(self):
        # 1 A == 1.8897261... bohr
        assert bond_angstrom_to_bohr(1.0) == pytest.approx(1.8897261246257702)
        assert bond_angstrom_to_bohr(0.0) == 0.0

    def test_inverse_round_trip(self):
        r = 0.9
        back = bond_angstrom_to_bohr(r) * 0.52917721067
        assert back == pytest.approx(r, rel=1e-12)


class TestValidationErrors:
    def test_too_small(self):
        with pytest.raises(MoleculeBuilderError):
            validate_bond_distance(MIN_BOND_ANGSTROM - 0.01)

    def test_too_large(self):
        with pytest.raises(MoleculeBuilderError):
            validate_bond_distance(MAX_BOND_ANGSTROM + 0.1)

    def test_zero_and_negative(self):
        with pytest.raises(MoleculeBuilderError):
            validate_bond_distance(0.0)
        with pytest.raises(MoleculeBuilderError):
            validate_bond_distance(-0.5)

    def test_non_numeric(self):
        with pytest.raises(MoleculeBuilderError):
            validate_bond_distance("not-a-number")
        with pytest.raises(MoleculeBuilderError):
            validate_bond_distance(None)

    def test_nan_and_inf(self):
        with pytest.raises(MoleculeBuilderError):
            validate_bond_distance(float("nan"))
        with pytest.raises(MoleculeBuilderError):
            validate_bond_distance(float("inf"))

    def test_build_hydrogen_propagates_error(self):
        with pytest.raises(MoleculeBuilderError):
            build_hydrogen(-1.0)

    def test_round_trip_float(self):
        r = validate_bond_distance(1.2345)
        assert r == pytest.approx(1.2345)


class TestMolecularGeometryContract:
    def test_to_nuclei_z_rejects_wrong_count(self):
        geom = MolecularGeometry(
            name="X",
            symbols=("H",),
            coords=((0.0, 0.0, 0.0),),
            bond_distance=0.0,
        )
        with pytest.raises(MoleculeBuilderError):
            geom.to_nuclei_z()

    def test_frozen_dataclass(self):
        geom = build_hydrogen(0.7)
        with pytest.raises(Exception):
            geom.bond_distance = 9.9
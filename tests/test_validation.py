"""Tests for the shared input validation helpers."""

import pytest

from src import validation
from src.molecule_builder import MoleculeBuilderError


class TestValidateBond:
    def test_accepts_valid(self):
        assert validation.validate_bond(0.9) == pytest.approx(0.9)
        assert validation.validate_bond(2.5) == pytest.approx(2.5)

    def test_rejects_invalid(self):
        for bad in (0.0, -1, 99.9, float("nan"), "abc", None):
            with pytest.raises(MoleculeBuilderError):
                validation.validate_bond(bad)


class TestValidateScanWindow:
    def test_valid_window(self):
        rmin, rmax, n = validation.validate_scan_window(0.5, 2.0, 9)
        assert (rmin, rmax, n) == (0.5, 2.0, 9)

    def test_reversed_bounds(self):
        with pytest.raises(ValueError):
            validation.validate_scan_window(2.0, 1.0, 5)

    def test_bad_point_count(self):
        with pytest.raises(ValueError):
            validation.validate_scan_window(0.5, 2.0, 2)  # below minimum 3
        with pytest.raises(ValueError):
            validation.validate_scan_window(0.5, 2.0, 99)

    def test_out_of_range_bond(self):
        with pytest.raises(MoleculeBuilderError):
            validation.validate_scan_window(0.0, 2.0, 5)


class TestIntFloatValidation:
    def test_validate_opt_int_coerces_and_clamps(self):
        assert validation.validate_opt_int("n", "7", 1, 10) == 7
        with pytest.raises(ValueError):
            validation.validate_opt_int("n", 11, 1, 10)
        with pytest.raises(ValueError):
            validation.validate_opt_int("n", "x", 1, 10)

    def test_validate_opt_float(self):
        assert validation.validate_opt_float("f", 0.5, 0.0, 1.0) == pytest.approx(0.5)
        with pytest.raises(ValueError):
            validation.validate_opt_float("f", 5.0, 0.0, 1.0)


class TestChoiceValidators:
    def test_backend(self):
        assert validation.validate_backend("STATEVECTOR") == "statevector"
        assert validation.validate_backend("aer") == "aer"
        with pytest.raises(ValueError):
            validation.validate_backend("quantinuum-2000")

    def test_ansatz(self):
        assert validation.validate_ansatz("RY") == "ry"
        assert validation.validate_ansatz("uccsd") == "uccsd"
        with pytest.raises(ValueError):
            validation.validate_ansatz("qaoa")

    def test_optimizer(self):
        assert validation.validate_optimizer("slsqp") == "SLSQP"
        assert validation.validate_optimizer("COBYLA") == "COBYLA"
        with pytest.raises(ValueError):
            validation.validate_optimizer("adam")

    def test_mapping(self):
        assert validation.validate_mapping("parity") == "parity"
        assert validation.validate_mapping("jordan-wigner") == "jordan-wigner"
        with pytest.raises(ValueError):
            validation.validate_mapping("bravyi-kitaev")

    def test_friendly_scan_bounds_message(self):
        msg = validation.friendly_scan_bounds()
        assert "Å" in msg and "0.25" in msg
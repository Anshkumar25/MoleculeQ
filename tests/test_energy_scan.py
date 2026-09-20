"""Tests for the bond-distance energy scan: counts, ordering, failures."""

import numpy as np
import pytest

from src.energy_scan import ScanPoint, ScanResult, run_scan, _scan_distances


class TestDistanceGrid:
    def test_point_count_includes_endpoints(self):
        distances = _scan_distances(0.6, 3.0, 12)
        assert len(distances) == 12
        assert distances[0] == pytest.approx(0.6)
        assert distances[-1] == pytest.approx(3.0)

    def test_strictly_increasing(self):
        distances = _scan_distances(0.5, 2.5, 9)
        assert np.all(np.diff(distances) > 0)

    def test_rejects_invalid_windows(self):
        with pytest.raises(ValueError):
            _scan_distances(2.0, 1.0, 5)
        with pytest.raises(ValueError):
            _scan_distances(0.5, 2.0, 1)  # less than 2 points

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            _scan_distances(0.05, 2.0, 5)  # below STO-3G minimum


class TestScanResult:
    def test_empty_result_lowest_is_none(self):
        scan = ScanResult(points=[], total_points=0)
        assert scan.lowest_energy_point() is None

    def test_lowest_energy_point_picks_minimum_vqe_total(self):
        pts = [
            ScanPoint(R_angstrom=1.0, vqe_total=-1.0),
            ScanPoint(R_angstrom=2.0, vqe_total=-2.5),
            ScanPoint(R_angstrom=3.0, vqe_total=-1.7),
            ScanPoint(R_angstrom=4.0, vqe_total=None, failed=True),
        ]
        scan = ScanResult(points=pts, total_points=4)
        best = scan.lowest_energy_point()
        assert best.R_angstrom == pytest.approx(2.0)

    def test_exact_totals_reconstruct(self):
        pts = [
            ScanPoint(R_angstrom=1.0, exact_electronic=-1.1, nuclear_repulsion=0.2),
            ScanPoint(R_angstrom=2.0, exact_electronic=-1.5, nuclear_repulsion=0.0),
        ]
        scan = ScanResult(points=pts, total_points=2)
        np.testing.assert_allclose(scan.exact_totals, [-0.9, -1.5])


class TestRunScan:
    def test_full_scan_small(self):
        """A small real scan (real pipeline) completes with valid energies."""
        scan = run_scan(0.6, 1.8, 5)
        assert scan.total_points == 5
        assert scan.completed_points == 5
        assert scan.failed_points == 0
        ok = [p for p in scan.points if p.vqe_total is not None]
        assert len(ok) == 5
        for p in ok:
            assert np.isfinite(p.vqe_electronic)

    def test_partial_failure_tolerance(self):
        """A point failure is recorded, not fatal — no fake data inserted."""

        def failing_run_vqe_fn(qubit_op, num_particles, nuclear_repulsion_energy):
            raise RuntimeError("driver crashed")

        def run_vqe_fn(qubit_op, num_particles, nuclear_repulsion_energy):
            # Fail for every other distance by interrogating caller state.
            return failing_run_vqe_fn(
                qubit_op, num_particles, nuclear_repulsion_energy
            )

        scan = run_scan(0.6, 1.2, 4, run_vqe_fn=run_vqe_fn)
        assert scan.total_points == 4
        assert scan.failed_points == 4
        assert len(scan.failure_reasons) == 4
        # every failed point must carry no fabricated energy
        for p in scan.points:
            assert p.failed
            assert p.vqe_total is None
            assert p.vqe_electronic is None
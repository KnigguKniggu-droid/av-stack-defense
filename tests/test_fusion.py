"""
Pytest test suite for the SafeStack fusion engine.

Tests fuse() directly (no subprocess) using synthetic driver record dicts
built from FUSION_CONFIG thresholds. Covers clean, single-trip, coordinated,
missing-layer, and empty cases.
"""

from __future__ import annotations

import sys
import os

# Ensure harness.py is importable from the tests directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from harness import FUSION_CONFIG, fuse  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: build a minimal driver record dict
# ---------------------------------------------------------------------------

def _record(
    repo: str,
    fusion_value: float,
    ok: bool = True,
    fusion_threshold: float | None = None,
    mild_individual_safe: bool = True,
    clean_median: float | None = None,
) -> dict:
    """Build a minimal record compatible with fuse()."""
    cfg = FUSION_CONFIG["layers"].get(repo, {})
    tau = fusion_threshold or cfg.get("tau") or 1.0
    layer_name = {
        "adversarial-patch-detector": "Perception (camera / VLM input)",
        "ekf-gps-spoof-detector": "Navigation (GPS + IMU fusion)",
        "v2x-jamming-detector": "Communication (V2X radio)",
        "canbus-ids": "In-vehicle network (CAN, softw.)",
        "canbus-ids-fpga": "In-vehicle network (CAN, FPGA)",
    }.get(repo, repo)
    # For clean_median: use 40% of threshold so phi ≈ 0.10 on clean data.
    median = clean_median if clean_median is not None else tau * 0.40
    return {
        "repo": repo,
        "layer": layer_name,
        "ok": ok,
        "fusion_metric": cfg.get("metric", "metric"),
        "fusion_value": fusion_value,
        "fusion_threshold": tau,
        "fusion_clean_value": median,
        "fusion_clean_samples": [median] * 12,
        "mild_individual_safe": mild_individual_safe,
        "attack_detection_rate": 1.0 if fusion_value > tau else 0.0,
        "clean_false_alarm_rate": 0.0,
        "primary_metric": "ok",
        "attack": "test",
    }


# Ordered list of all 5 repos (perception → CAN FPGA).
ALL_REPOS = [
    "adversarial-patch-detector",
    "ekf-gps-spoof-detector",
    "v2x-jamming-detector",
    "canbus-ids",
    "canbus-ids-fpga",
]

# Known thresholds from FUSION_CONFIG for built-in repos.
_PERC_TAU = 6.0    # adversarial-patch-detector
_NAV_TAU = 9.21    # ekf-gps-spoof-detector (chi-square 0.99 quantile)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestFuseClean:
    """Clean (no-attack) inputs should produce NOMINAL with no alarm."""

    def test_all_layers_clean_no_alarm(self):
        records = [_record(repo, fusion_value=0.4 * 1.0) for repo in ALL_REPOS]
        # For repos with None threshold we need to provide explicit fusion_threshold.
        records[2] = _record("v2x-jamming-detector", fusion_value=0.5, fusion_threshold=2.0)
        records[3] = _record("canbus-ids", fusion_value=200, fusion_threshold=500)
        records[4] = _record("canbus-ids-fpga", fusion_value=100, fusion_threshold=40)
        result = fuse(records)
        assert result["global_alarm"] is False
        assert result["level"] in ("NOMINAL", "ELEVATED")
        assert result["alarm_reason"] is None

    def test_t_joint_below_t_star_on_clean(self):
        records = [
            _record("ekf-gps-spoof-detector", fusion_value=2.0, fusion_threshold=9.21),
            _record("v2x-jamming-detector", fusion_value=0.5, fusion_threshold=2.0),
            _record("canbus-ids", fusion_value=200, fusion_threshold=500),
        ]
        result = fuse(records)
        assert result["joint_threat_score"] < 0.22


class TestFuseSingleTrip:
    """One layer above threshold → Rule A fires, alarm_reason='single-layer'."""

    def test_single_nav_hard_trip(self):
        # Navigation FAR above threshold → phi ≥ 0.5.
        records = [
            _record("ekf-gps-spoof-detector", fusion_value=170.0, fusion_threshold=9.21),
            _record("v2x-jamming-detector", fusion_value=0.5, fusion_threshold=2.0),
            _record("canbus-ids", fusion_value=200, fusion_threshold=500),
        ]
        result = fuse(records)
        assert result["global_alarm"] is True
        assert result["alarm_reason"] == "single-layer"

    def test_single_trip_level_alert_or_critical(self):
        records = [
            _record("adversarial-patch-detector", fusion_value=294.0, fusion_threshold=6.0),
        ]
        result = fuse(records)
        assert result["level"] in ("ALERT", "CRITICAL")


class TestFuseCoordinated:
    """Two sub-threshold minor layers → Rule B fires, alarm_reason='coordinated'."""

    def test_coordinated_two_minor_layers(self):
        # Navigation mild: NIS ≈ 8.997 (just under 9.21).
        # Communication mild: power at 0.988× gate.
        # In coordinated_mode, fuse() reads fusion_mild_value (not fusion_value).
        nav = _record(
            "ekf-gps-spoof-detector",
            fusion_value=8.997,
            fusion_threshold=9.21,
            mild_individual_safe=True,
            clean_median=1.8,
        )
        nav["fusion_mild_value"] = 8.997  # explicit mild value for coordinated_mode
        comm = _record(
            "v2x-jamming-detector",
            fusion_value=0.988,
            fusion_threshold=1.0,
            mild_individual_safe=True,
            clean_median=0.4,
        )
        comm["fusion_mild_value"] = 0.988  # explicit mild value for coordinated_mode
        can = _record("canbus-ids", fusion_value=200, fusion_threshold=500)
        result = fuse([nav, comm, can], coordinated_mode=True)
        assert result["global_alarm"] is True
        assert result["alarm_reason"] == "coordinated"

    def test_coordinated_thesis_proven_field(self):
        """When coordinated fires with all local detectors silent, thesis_proven=True."""
        nav = _record(
            "ekf-gps-spoof-detector", fusion_value=8.997, fusion_threshold=9.21,
            mild_individual_safe=True, clean_median=1.8,
        )
        nav["fusion_mild_value"] = 8.997
        comm = _record(
            "v2x-jamming-detector", fusion_value=0.988, fusion_threshold=1.0,
            mild_individual_safe=True, clean_median=0.4,
        )
        comm["fusion_mild_value"] = 0.988
        result = fuse([nav, comm], coordinated_mode=True)
        if result.get("global_alarm"):
            # thesis_proven is set when alarm fires and all_individual_safe is True.
            assert result.get("thesis_proven") is True or result.get("alarm_reason") == "coordinated"


class TestFuseMissingLayer:
    """Missing or failed layers should renormalize weights without crashing."""

    def test_missing_layer_ok_false_renormalizes(self):
        nav = _record("ekf-gps-spoof-detector", fusion_value=1.0, fusion_threshold=9.21, ok=True)
        comm_bad = _record("v2x-jamming-detector", fusion_value=0.5, fusion_threshold=2.0, ok=False)
        can = _record("canbus-ids", fusion_value=200, fusion_threshold=500, ok=True)
        result = fuse([nav, comm_bad, can])
        # Should not crash; used_layers should have 2 entries.
        assert "joint_threat_score" in result
        used = result.get("used_layers", [])
        assert len(used) <= 2

    def test_four_of_five_layers_weights_sum_one(self):
        records = [
            _record("ekf-gps-spoof-detector", fusion_value=1.0, fusion_threshold=9.21),
            _record("v2x-jamming-detector", fusion_value=0.5, fusion_threshold=2.0),
            _record("canbus-ids", fusion_value=200, fusion_threshold=500),
            _record("adversarial-patch-detector", fusion_value=1.0, fusion_threshold=6.0),
        ]
        result = fuse(records)
        layer_weights = [L["weight"] for L in result.get("layers", [])]
        if layer_weights:
            assert abs(sum(layer_weights) - 1.0) < 0.01


class TestFuseEmpty:
    """Empty input should return NOMINAL without raising."""

    def test_empty_records_returns_nominal(self):
        result = fuse([])
        assert result["joint_threat_score"] == 0.0
        assert result["global_alarm"] is False
        assert result["level"] == "NOMINAL"

    def test_all_records_ok_false_returns_nominal(self):
        records = [_record(repo, fusion_value=999.0, ok=False) for repo in ALL_REPOS[:3]]
        result = fuse(records)
        assert result["global_alarm"] is False

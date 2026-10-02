"""
coordinated_attack.py — Coordinated cross-layer attack demo driver.

Runs the proven sub-threshold GPS + V2X coordinated attack:
  - Navigation: GPS spike at NIS ≈ 8.997 (under 9.21 gate) → local detector SILENT
  - Communication: tone at 0.988× power gate → local detector SILENT
  - Fusion fires: T_joint ≥ 0.22, alarm_reason="coordinated", thesis_proven=True
  - POSTs FusionResult to SafeStack API → EHD EMERGENCY_STOP

Usage:
    python drivers/coordinated_attack.py
    python harness.py --coordinated
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # one level above av-stack-defense

# Add parent so harness imports work.
sys.path.insert(0, os.path.dirname(HERE))

from harness import FUSION_CONFIG, run_layer, fuse  # noqa: E402


def run_coordinated_attack(api_url: str = "http://localhost:8000") -> dict:
    """Run the coordinated attack and POST the result to the SafeStack API.

    Returns a dict with keys: fusion, ehd_status.
    Raises AssertionError if the coordinated alarm does not fire.
    """
    nav_repo = os.path.join(ROOT, "ekf-gps-spoof-detector")
    comm_repo = os.path.join(ROOT, "v2x-jamming-detector")
    perc_repo = os.path.join(ROOT, "adversarial-patch-detector")
    can_repo = os.path.join(ROOT, "canbus-ids")

    print("[SafeStack] Running coordinated attack — mild GPS spoof + mild V2X jamming")
    print("[SafeStack] Both local detectors should stay SILENT...")

    # Run mild (sub-threshold) versions of navigation and communication.
    nav_result = run_layer(
        os.path.join(HERE, "navigation.py"), nav_repo, mode="mild"
    )
    comm_result = run_layer(
        os.path.join(HERE, "communication.py"), comm_repo, mode="mild"
    )

    # Run clean (no-attack) versions of the remaining layers for weight renormalization.
    perc_result = run_layer(os.path.join(HERE, "perception.py"), perc_repo)
    can_result = run_layer(os.path.join(HERE, "invehicle_can.py"), can_repo)

    # Validate: local detectors must be silent on mild inputs.
    for result in (nav_result, comm_result):
        if result.get("ok"):
            mild_safe = result.get("mild_individual_safe", True)
            if not mild_safe:
                print(
                    f"[SafeStack] WARNING: {result.get('repo')} local detector fired on mild input"
                )

    # Fuse all layers in coordinated mode.
    records = [nav_result, comm_result, perc_result, can_result]
    fusion = fuse(records, coordinated_mode=True)

    t_joint = fusion.get("joint_threat_score", 0.0)
    level = fusion.get("level", "NOMINAL")
    alarm_reason = fusion.get("alarm_reason")
    thesis_proven = fusion.get("thesis_proven", False)

    print(f"\n[SafeStack] Fusion result:")
    print(f"  T_joint       = {t_joint:.3f}")
    print(f"  Level         = {level}")
    print(f"  Alarm reason  = {alarm_reason}")
    print(f"  Thesis proven = {thesis_proven}")

    if alarm_reason != "coordinated":
        print(
            f"[SafeStack] WARNING: expected alarm_reason='coordinated', got '{alarm_reason}'. "
            "The fusion may not have fired. Check that mild attack values are sub-threshold."
        )

    # POST to SafeStack API.
    ehd_status: dict = {}
    try:
        import requests  # type: ignore[import-untyped]

        # Build a minimal FusionResult JSON compatible with safestack_api.models.FusionResult.
        layers_payload = []
        for L in fusion.get("layers", []):
            layers_payload.append(
                {
                    "layer": L.get("layer", ""),
                    "repo": L.get("repo", ""),
                    "metric": L.get("metric", ""),
                    "value": float(L.get("value", 0.0)),
                    "threshold": L.get("threshold"),
                    "phi": float(L.get("phi", 0.0)),
                    "weight": float(L.get("weight", 0.0)),
                    "contribution": float(L.get("contribution", 0.0)),
                    "trip": bool(L.get("trip", False)),
                    "minor": bool(L.get("minor", False)),
                    "in_score": bool(L.get("in_score", True)),
                    "coordination_eligible": bool(L.get("coordination_eligible", True)),
                }
            )

        payload = {
            "joint_threat_score": t_joint,
            "level": level,
            "global_alarm": bool(fusion.get("global_alarm", False)),
            "alarm_reason": alarm_reason,
            "n_minor": int(fusion.get("n_minor", 0)),
            "hard_trips": fusion.get("hard_trips", []),
            "weights_mode": fusion.get("weights_mode", "equal"),
            "layers": layers_payload,
        }

        response = requests.post(
            f"{api_url}/api/threat-event",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        ehd_status = response.json()

        print(f"\n[SafeStack] EHD safety supervisor response:")
        print(f"  safety_state = {ehd_status.get('safety_state')}")
        print(f"  fault_code   = {ehd_status.get('fault_code')}")

        if ehd_status.get("safety_state") == "EMERGENCY_STOP":
            print("\n[SafeStack] ✓ COORDINATED ATTACK DETECTED → EHD EMERGENCY STOP")
        else:
            print(
                f"\n[SafeStack] EHD state is {ehd_status.get('safety_state')} "
                "(expected EMERGENCY_STOP on ALERT level)"
            )

    except ImportError:
        print("[SafeStack] requests not installed — skipping API POST. Install with: pip install requests")
    except Exception as exc:  # noqa: BLE001
        print(f"[SafeStack] API POST failed: {exc}")
        print("[SafeStack] Is the SafeStack API running at", api_url, "?")

    return {"fusion": fusion, "ehd_status": ehd_status}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the coordinated AV attack demo.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="SafeStack API URL")
    args = parser.parse_args()
    result = run_coordinated_attack(api_url=args.api_url)
    print("\n[SafeStack] Done.")

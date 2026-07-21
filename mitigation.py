"""
Resilient control / mitigation state machine for the AV defense stack.

Reads the fusion output (T_joint, per-layer phi values, hard trips) and
determines what defensive action the vehicle should take.  Maps threat
levels to specific containment strategies: sensor isolation, degraded
mode operation, or emergency safe-stop.

The state machine is hierarchical: higher threat levels subsume lower ones.
Transitions are deterministic (no hysteresis in this version) so the
dashboard can show exactly why a given actuation was chosen.

Usage:
    from mitigation import MitigationEngine
    engine = MitigationEngine()
    state = engine.evaluate(fusion_result)
    # state has: level, actions, isolated_layers, degraded_sensors, safe_stop
"""
from typing import Dict, List, Optional
from enum import Enum


class MitigationState(Enum):
    """Possible vehicle-level mitigation states, from least to most severe."""
    NORMAL = "NORMAL"
    MONITORING = "MONITORING"
    GPS_ISOLATED = "GPS_ISOLATED"
    V2X_DEGRADED = "V2X_DEGRADED"
    PERCEPTION_MASKED = "PERCEPTION_MASKED"
    CAN_SAFE_STOP = "CAN_SAFE_STOP"
    FULL_FALLBACK = "FULL_FALLBACK"


# Transition thresholds
T_ELEVATED = 0.15
T_ALERT = 0.22
T_CRITICAL = 0.45


class MitigationEngine:
    """Evaluates fusion output and determines the mitigation response."""

    def __init__(self):
        self.state = MitigationState.NORMAL
        self.actions_taken: List[Dict] = []

    def evaluate(self, fusion: Dict) -> Dict:
        """Given a fusion result dict (from harness.py fuse()), return the
        mitigation state and a list of specific actions.

        Returns a dict with:
            state: MitigationState value
            state_label: human-readable label
            actions: list of {target, action, reason}
            isolated_layers: list of layer repo names to deprioritize
            degraded_sensors: list of sensor types to reduce trust in
            safe_stop: bool, whether the vehicle should pull over
            register_overrides: dict of AXI register writes to apply
        """
        self.actions_taken = []
        t_joint = fusion.get("joint_threat_score", 0.0)
        level = fusion.get("level", "NOMINAL")
        hard_trips = fusion.get("hard_trips_eligible", fusion.get("hard_trips", []))
        layers = {L["repo"]: L for L in fusion.get("layers", [])}

        # Determine which layers are anomalous
        nav_phi = layers.get("ekf-gps-spoof-detector", {}).get("phi", 0.10)
        comm_phi = layers.get("v2x-jamming-detector", {}).get("phi", 0.10)
        perc_phi = layers.get("adversarial-patch-detector", {}).get("phi", 0.10)
        can_sw_phi = layers.get("canbus-ids", {}).get("phi", 0.10)
        can_hw_phi = layers.get("canbus-ids-fpga", {}).get("phi", 0.10)

        isolated_layers = []
        degraded_sensors = []
        register_overrides = {}
        safe_stop = False

        # --- State determination (checked in severity order) ---

        if t_joint >= T_CRITICAL or len(hard_trips) >= 2:
            self.state = MitigationState.FULL_FALLBACK
            safe_stop = True
            isolated_layers = list(layers.keys())
            degraded_sensors = ["gps", "v2x", "camera", "can_bus"]
            self._action("vehicle", "INITIATE_SAFE_PULL_OVER",
                         f"T_joint={t_joint:.3f} >= {T_CRITICAL} or {len(hard_trips)} hard trips")
            self._action("can_bus", "EMERGENCY_HW_OVERRIDE",
                         "Trigger hardware-level safe-stop state on CAN gateway")
            self._action("ekf", "SWITCH_TO_IMU_ONLY",
                         "Isolate GPS completely, dead-reckon from IMU")
            self._action("v2x", "REDUCE_GAIN_TO_ZERO",
                         "Silence V2X radio to prevent poisoned inputs")
            self._action("perception", "MASK_LOW_CONFIDENCE",
                         "Ignore low-confidence vision detections")
            register_overrides = {
                "GPS_ENABLE": 0,
                "V2X_GAIN": 0,
                "SAFE_STOP_CMD": 1,
                "CAN_MIN_PERIOD_OVERRIDE": 1,
            }

        elif level == "ALERT" or (level == "ELEVATED" and len(hard_trips) >= 1):
            # Check which specific layer is driving the alert
            if "ekf-gps-spoof-detector" in hard_trips or nav_phi >= 0.5:
                self.state = MitigationState.GPS_ISOLATED
                isolated_layers.append("ekf-gps-spoof-detector")
                degraded_sensors.append("gps")
                self._action("ekf", "SWITCH_TO_IMU_ONLY",
                             f"Navigation phi={nav_phi:.3f} >= 0.5, GPS unreliable")
                register_overrides["GPS_ENABLE"] = 0

            if "v2x-jamming-detector" in hard_trips or comm_phi >= 0.5:
                self.state = MitigationState.V2X_DEGRADED
                isolated_layers.append("v2x-jamming-detector")
                degraded_sensors.append("v2x")
                self._action("v2x", "REDUCE_GAIN_TO_MINIMAL",
                             f"Communication phi={comm_phi:.3f} >= 0.5, channel jammed")
                register_overrides["V2X_GAIN"] = 1

            if "adversarial-patch-detector" in hard_trips or perc_phi >= 0.5:
                self.state = MitigationState.PERCEPTION_MASKED
                isolated_layers.append("adversarial-patch-detector")
                degraded_sensors.append("camera")
                self._action("perception", "MASK_AFFECTED_REGIONS",
                             f"Perception phi={perc_phi:.3f} >= 0.5, patch detected")
                register_overrides["PERCEPTION_MASK"] = 1

            if "canbus-ids" in hard_trips or can_sw_phi >= 0.5:
                self.state = MitigationState.CAN_SAFE_STOP
                safe_stop = True
                isolated_layers.append("canbus-ids")
                degraded_sensors.append("can_bus")
                self._action("can_bus", "INITIATE_SAFE_STOP",
                             f"CAN SW phi={can_sw_phi:.3f} >= 0.5, bus compromised")
                register_overrides["SAFE_STOP_CMD"] = 1
                register_overrides["CAN_MIN_PERIOD_OVERRIDE"] = 1

            # If multiple layers tripped simultaneously, escalate to full fallback
            if len(hard_trips) >= 2:
                self.state = MitigationState.FULL_FALLBACK
                safe_stop = True
                isolated_layers = list(layers.keys())
                degraded_sensors = ["gps", "v2x", "camera", "can_bus"]
                self._action("vehicle", "ESCALATE_TO_FULL_FALLBACK",
                             f"{len(hard_trips)} simultaneous hard trips")

        elif t_joint >= T_ALERT:
            self.state = MitigationState.MONITORING
            self._action("system", "ELEVATE_MONITORING",
                         f"T_joint={t_joint:.3f} in ALERT band, increasing sensor cross-checks")

        elif t_joint >= T_ELEVATED:
            self.state = MitigationState.MONITORING
            self._action("system", "BEGIN_MONITORING",
                         f"T_joint={t_joint:.3f} in ELEVATED band, watch for escalation")

        else:
            self.state = MitigationState.NORMAL

        return {
            "state": self.state.value,
            "state_label": self._label(),
            "actions": self.actions_taken,
            "isolated_layers": isolated_layers,
            "degraded_sensors": degraded_sensors,
            "safe_stop": safe_stop,
            "register_overrides": register_overrides,
            "trigger": {
                "t_joint": round(t_joint, 3),
                "level": level,
                "hard_trips": hard_trips,
                "nav_phi": round(nav_phi, 3),
                "comm_phi": round(comm_phi, 3),
                "perc_phi": round(perc_phi, 3),
                "can_sw_phi": round(can_sw_phi, 3),
                "can_hw_phi": round(can_hw_phi, 3),
            },
        }

    def _action(self, target: str, action: str, reason: str):
        self.actions_taken.append({
            "target": target,
            "action": action,
            "reason": reason,
        })

    def _label(self) -> str:
        labels = {
            MitigationState.NORMAL: "Normal Operation",
            MitigationState.MONITORING: "Elevated Monitoring",
            MitigationState.GPS_ISOLATED: "GPS Isolated (IMU Dead Reckoning)",
            MitigationState.V2X_DEGRADED: "V2X Degraded (Reduced Gain)",
            MitigationState.PERCEPTION_MASKED: "Perception Masked (Region Filtering)",
            MitigationState.CAN_SAFE_STOP: "CAN Safe Stop (Pull Over)",
            MitigationState.FULL_FALLBACK: "Full Fallback (Emergency Stop)",
        }
        return labels.get(self.state, "Unknown")

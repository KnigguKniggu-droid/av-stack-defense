"""
Cross-layer dependency graph for AV stack fusion.

Models attack propagation vectors between the five detector layers as a directed
weighted graph.  When one layer's anomaly rises, it propagates a fractional boost
to dependent layers, reflecting real-world attack chains (e.g. V2X jamming
degrades navigation, which amplifies CAN anomalies).

The graph is static (domain-knowledge initialized) but the propagation is
computed dynamically from per-layer phi values each fusion cycle.

Usage:
    from dependency_graph import DependencyGraph
    g = DependencyGraph()
    adjusted = g.propagate(layer_phis)
    # adjusted has the same shape as layer_phis but with cross-layer effects
"""
import math
from typing import Dict, List, Tuple

# Layer keys (must match the repo names in harness.py)
LAYERS = [
    "adversarial-patch-detector",
    "ekf-gps-spoof-detector",
    "v2x-jamming-detector",
    "canbus-ids",
    "canbus-ids-fpga",
]

# Short display names
LAYER_NAMES = {
    "adversarial-patch-detector": "Perception",
    "ekf-gps-spoof-detector": "Navigation",
    "v2x-jamming-detector": "Communication",
    "canbus-ids": "CAN (SW)",
    "canbus-ids-fpga": "CAN (FPGA)",
}

# Directed propagation weights: ADJ[sender][receiver] = weight in [0, 1)
# Weight 0 means no influence.  Higher means stronger cross-layer effect.
# Initialized from domain knowledge of real AV attack propagation chains.
#
# Key propagation paths modeled:
#   Communication jamming -> Navigation degradation (GPS relies on V2X辅助 corrections)
#   Navigation spoof -> CAN anomaly (wrong position -> unexpected control commands)
#   Perception patch -> Navigation misread (wrong sign -> wrong path -> position drift)
#   Communication jam -> CAN flooding (attacker jams V2X then floods bus)
#   CAN flood -> FPGA alert (software flood triggers hardware detector)
#   Navigation spoof -> Perception (wrong location -> different scene context)
ADJ: Dict[str, Dict[str, float]] = {
    "adversarial-patch-detector": {
        "ekf-gps-spoof-detector": 0.08,     # misread sign -> wrong path -> GPS drift
        "v2x-jamming-detector": 0.0,        # no direct propagation
        "canbus-ids": 0.05,                  # wrong control action -> CAN anomaly
        "canbus-ids-fpga": 0.03,            # same, through hardware
    },
    "ekf-gps-spoof-detector": {
        "adversarial-patch-detector": 0.05,  # wrong position -> different scene context
        "v2x-jamming-detector": 0.0,        # GPS spoof doesn't cause jamming
        "canbus-ids": 0.12,                 # wrong position -> unexpected actuator commands
        "canbus-ids-fpga": 0.10,            # same, through hardware
    },
    "v2x-jamming-detector": {
        "adversarial-patch-detector": 0.0,  # jamming doesn't affect camera
        "ekf-gps-spoof-detector": 0.10,    # V2X辅助 GPS corrections lost
        "canbus-ids": 0.08,                 # attacker jams then floods bus
        "canbus-ids-fpga": 0.06,            # same, through hardware
    },
    "canbus-ids": {
        "adversarial-patch-detector": 0.0,  # CAN flood doesn't affect camera
        "ekf-gps-spoof-detector": 0.0,     # CAN flood doesn't affect GPS
        "v2x-jamming-detector": 0.0,       # CAN flood doesn't cause jamming
        "canbus-ids-fpga": 0.15,           # SW flood triggers HW detector
    },
    "canbus-ids-fpga": {
        "adversarial-patch-detector": 0.0,
        "ekf-gps-spoof-detector": 0.0,
        "v2x-jamming-detector": 0.0,
        "canbus-ids": 0.05,                # HW alert can cascade to SW
    },
}

# Self-influence: how much a layer's own anomaly amplifies through recursion.
# Set to 0 so we only model cross-layer effects (self-effects are already in phi).
SELF_WEIGHT = 0.0

# Propagation iterations.  More iterations model longer attack chains
# (A -> B -> C).  2 is sufficient for the current 5-layer graph.
ITERS = 2

# Damping factor to prevent runaway amplification across iterations.
DAMPING = 0.85


class DependencyGraph:
    """Cross-layer attack propagation model."""

    def __init__(self, adj: Dict[str, Dict[str, float]] = None,
                 iters: int = ITERS, damping: float = DAMPING):
        self.adj = adj or ADJ
        self.iters = iters
        self.damping = damping
        self.layers = LAYERS

    def propagate(self, layer_phis: Dict[str, float]) -> Dict[str, float]:
        """Given raw per-layer phi values, return adjusted phi values that
        account for cross-layer attack propagation.

        The algorithm:
          1. Start with the raw phi values.
          2. For each iteration, each layer receives a fractional boost from
             every other layer proportional to the sender's current anomaly
             and the edge weight.
          3. Apply damping to prevent runaway amplification.
          4. Clamp all values to [0, 1].

        Returns a dict with the same keys as layer_phis, plus a
        'propagation_deltas' key showing how much each layer shifted.
        """
        raw = {k: layer_phis.get(k, 0.10) for k in self.layers}
        adjusted = dict(raw)

        deltas = {k: 0.0 for k in self.layers}
        propagation_log = []

        for iteration in range(self.iters):
            new_vals = dict(adjusted)
            iter_deltas = {k: 0.0 for k in self.layers}

            for receiver in self.layers:
                boost = 0.0
                contributors = []
                for sender in self.layers:
                    if sender == receiver:
                        continue
                    w = self.adj.get(sender, {}).get(receiver, 0.0)
                    if w <= 0:
                        continue
                    contribution = adjusted[sender] * w
                    boost += contribution
                    if contribution > 0.001:
                        contributors.append({
                            "from": LAYER_NAMES.get(sender, sender),
                            "weight": round(w, 3),
                            "sender_phi": round(adjusted[sender], 3),
                            "contribution": round(contribution, 4),
                        })

                boost *= self.damping
                new_val = min(1.0, adjusted[receiver] + boost)
                iter_deltas[receiver] = new_val - adjusted[receiver]
                new_vals[receiver] = new_val

            adjusted = new_vals
            for k in self.layers:
                deltas[k] += iter_deltas[k]

            propagation_log.append({
                "iteration": iteration + 1,
                "deltas": {LAYER_NAMES.get(k, k): round(v, 4)
                           for k, v in iter_deltas.items() if abs(v) > 0.0001},
            })

        # Final output
        result = dict(adjusted)
        result["propagation_deltas"] = {
            LAYER_NAMES.get(k, k): round(v, 4)
            for k, v in deltas.items()
        }
        result["propagation_log"] = propagation_log
        result["raw_phis"] = {
            LAYER_NAMES.get(k, k): round(v, 3)
            for k, v in raw.items()
        }
        return result

    def get_graph_data(self) -> Dict:
        """Return the graph structure for dashboard visualization."""
        nodes = []
        for layer in self.layers:
            nodes.append({
                "id": layer,
                "name": LAYER_NAMES.get(layer, layer),
            })

        edges = []
        for sender in self.layers:
            for receiver in self.layers:
                w = self.adj.get(sender, {}).get(receiver, 0.0)
                if w > 0:
                    edges.append({
                        "source": sender,
                        "target": receiver,
                        "weight": w,
                        "source_name": LAYER_NAMES.get(sender, sender),
                        "target_name": LAYER_NAMES.get(receiver, receiver),
                    })

        return {"nodes": nodes, "edges": edges}

    def get_critical_paths(self, layer_phis: Dict[str, float],
                           threshold: float = 0.30) -> List[Dict]:
        """Identify propagation paths where the cross-layer boost is significant.
        Returns paths sorted by total propagated impact."""
        paths = []
        for receiver in self.layers:
            total_boost = 0.0
            contributors = []
            for sender in self.layers:
                if sender == receiver:
                    continue
                w = self.adj.get(sender, {}).get(receiver, 0.0)
                if w <= 0:
                    continue
                sender_phi = layer_phis.get(sender, 0.10)
                contribution = sender_phi * w * self.damping
                if contribution > 0.001:
                    total_boost += contribution
                    contributors.append({
                        "layer": LAYER_NAMES.get(sender, sender),
                        "phi": round(sender_phi, 3),
                        "weight": w,
                        "boost": round(contribution, 4),
                    })
            if total_boost >= threshold:
                paths.append({
                    "target": LAYER_NAMES.get(receiver, receiver),
                    "total_boost": round(total_boost, 4),
                    "contributors": sorted(contributors,
                                           key=lambda c: c["boost"],
                                           reverse=True),
                })
        return sorted(paths, key=lambda p: p["total_boost"], reverse=True)

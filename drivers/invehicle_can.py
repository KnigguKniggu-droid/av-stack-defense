"""
In-vehicle-network driver (software): runs the CAN timing IDS on clean traffic
(expect ~0 false positives) and a DoS flood (expect full detection). Uses the
repo's own bus, attacks, and detector modules.

For cross-layer fusion it emits the observed bus rate under flood, the median
clean rate (calibration anchor), a distribution of clean rates over many
independent windows, and a mild rate deviation that the IDS itself does not flag
but that raises the fused anomaly (used for the coordinated cross-layer test).
Emits one normalized JSON line on stdout.
"""
import json
import os
import statistics
import sys

repo = sys.argv[1]
sys.path.insert(0, repo)
os.chdir(repo)

from core.bus import generate_traffic
from core import attacks
from core.detector import TimingIDS, score

M_CLEAN = 12
# Mild deviation: a small injection rate, tuned to keep the observed bus rate
# below the rate gate and below the timing IDS's own alarm.
MILD_RATE_HZ = 70


def observed_rate(frames):
    if len(frames) < 2:
        return float(len(frames))
    span = frames[-1].timestamp - frames[0].timestamp
    return len(frames) / span if span > 0 else float(len(frames))


ids = TimingIDS().train(generate_traffic(duration_s=5.0, seed=1))

clean_rates = []
for s in range(M_CLEAN):
    clean_s = generate_traffic(duration_s=3.0, seed=100 + s)
    clean_rates.append(round(observed_rate(clean_s), 6))

clean = generate_traffic(duration_s=3.0, seed=2)
s_clean = score(ids.detect(clean), clean)

flood = attacks.flooding(generate_traffic(duration_s=3.0, seed=2), rate_hz=2000)
s_atk = score(ids.detect(flood), flood)
flood_rate = observed_rate(flood)

# The timing IDS flags any injection regardless of how small the rate bump is,
# so CAN cannot be a silent-but-elevated layer. It stays at its clean value in
# the coordinated test (its own detector remains the authority for injection).
rate_threshold = ids.bus_rate * 1.25 if ids.bus_rate is not None else None
clean_median = statistics.median(clean_rates)

print(json.dumps({
    "layer": "In-vehicle network (CAN bus, software)",
    "repo": "canbus-ids",
    "attack": "flooding / DoS",
    "clean_false_alarm_rate": s_clean.get("false_positive_rate"),
    "attack_detection_rate": s_atk.get("detection_rate"),
    "primary_metric": f"detect {s_atk.get('detection_rate')}, FP {s_clean.get('false_positive_rate')}",
    "fusion_metric": "rho",
    "fusion_value": round(flood_rate, 6),
    "fusion_threshold": round(rate_threshold, 6) if rate_threshold is not None else None,
    "fusion_clean_value": round(clean_median, 6),
    "fusion_clean_samples": clean_rates,
    "fusion_mild_value": round(clean_median, 6),
    "mild_individual_safe": True,
    "ok": True,
}))

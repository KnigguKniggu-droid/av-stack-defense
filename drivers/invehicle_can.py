"""
In-vehicle-network driver (software): runs the CAN timing IDS on clean traffic
(expect ~0 false positives) and a DoS flood (expect full detection). Uses the
repo's own bus, attacks, and detector modules.
Emits one normalized JSON line on stdout.
"""
import json
import os
import sys

repo = sys.argv[1]
sys.path.insert(0, repo)
os.chdir(repo)

from core.bus import generate_traffic
from core import attacks
from core.detector import TimingIDS, score

ids = TimingIDS().train(generate_traffic(duration_s=5.0, seed=1))

clean = generate_traffic(duration_s=3.0, seed=2)
s_clean = score(ids.detect(clean), clean)

flood = attacks.flooding(generate_traffic(duration_s=3.0, seed=2), rate_hz=2000)
s_atk = score(ids.detect(flood), flood)

print(json.dumps({
    "layer": "In-vehicle network (CAN bus, software)",
    "repo": "canbus-ids",
    "attack": "flooding / DoS",
    "clean_false_alarm_rate": s_clean.get("false_positive_rate"),
    "attack_detection_rate": s_atk.get("detection_rate"),
    "primary_metric": f"detect {s_atk.get('detection_rate')}, FP {s_clean.get('false_positive_rate')}",
    "ok": True,
}))

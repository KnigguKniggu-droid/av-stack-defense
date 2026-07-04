"""Collect CAN timing: clean periodic traffic vs a DoS flood, with the flood marked."""
import os
import sys

import numpy as np

repo, datadir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
os.chdir(repo)

from core.bus import generate_traffic
from core import attacks
from core.detector import TimingIDS

clean = generate_traffic(3.0, seed=2)
flood = attacks.flooding(generate_traffic(3.0, seed=2), rate_hz=2000)

# The IDS learns the per-ID period and the normal bus rate from clean traffic.
ids_model = TimingIDS().train(clean)
baseline = set(ids_model.baseline)                       # known arbitration IDs
# Pick the fastest known ID to display its learned period.
fastest = min(ids_model.baseline.items(), key=lambda kv: kv[1]["period"])
learned_period_ms = fastest[1]["period"] * 1000.0
normal_bus_rate = float(ids_model.bus_rate or 0.0)

# Flood id and its observed rate (msgs/s) in a 50 ms window.
flood_ids = [f.arb_id for f in flood if f.arb_id not in baseline]
flood_id = flood_ids[0] if flood_ids else 0
ft = sorted(f.timestamp for f in flood if f.arb_id == flood_id)
flood_rate = (len(ft) / (ft[-1] - ft[0])) if len(ft) > 1 and ft[-1] > ft[0] else 0.0

t = np.array([f.timestamp for f in flood])
ids = np.array([f.arb_id for f in flood])
is_attack = np.array([0 if f.arb_id in baseline else 1 for f in flood])

np.savez(os.path.join(datadir, "can.npz"),
         t=t, ids=ids, is_attack=is_attack,
         clean_t=np.array([f.timestamp for f in clean]),
         clean_ids=np.array([f.arb_id for f in clean]),
         learned_period_ms=learned_period_ms, timing_tolerance=0.5,
         normal_bus_rate=normal_bus_rate, flood_rate=float(flood_rate),
         flood_id=int(flood_id), rate_factor=4.0)
print("can saved")

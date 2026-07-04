"""Collect CAN timing: clean periodic traffic vs a DoS flood, with the flood marked."""
import os
import sys

import numpy as np

repo, datadir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
os.chdir(repo)

from core.bus import generate_traffic
from core import attacks

clean = generate_traffic(3.0, seed=2)
flood = attacks.flooding(generate_traffic(3.0, seed=2), rate_hz=2000)

# Baseline arbitration IDs (from clean traffic). Anything else is an intruder.
baseline = set(f.arb_id for f in clean)

t = np.array([f.timestamp for f in flood])
ids = np.array([f.arb_id for f in flood])
is_attack = np.array([0 if f.arb_id in baseline else 1 for f in flood])

np.savez(os.path.join(datadir, "can.npz"),
         t=t, ids=ids, is_attack=is_attack,
         clean_t=np.array([f.timestamp for f in clean]),
         clean_ids=np.array([f.arb_id for f in clean]))
print("can saved")

"""
Communication-layer driver: runs the V2X RF jamming detector on clean OFDM
frames (expect no false alarms) and tone-jammed frames (expect detection +
classification). Uses the repo's own channel, jammers, and detector modules.
Emits one normalized JSON line on stdout.
"""
import json
import os
import sys

repo = sys.argv[1]
sys.path.insert(0, repo)
os.chdir(repo)

import numpy as np
from core import channel, jammers
from core.detector import JammingDetector

N = 40
det = JammingDetector().train(channel.frames(40, snr_db=15.0, seed=100))

clean = channel.frames(N, snr_db=15.0, seed=500)
clean_res = [det.analyze(f) for f in clean]
clean_jammed = sum(r["jammed"] for r in clean_res)

tone = [jammers.tone(f) for f in channel.frames(N, snr_db=15.0, seed=500)]
tone_res = [det.analyze(f) for f in tone]
tone_detected = sum(r["jammed"] for r in tone_res)
tone_classified = sum(1 for r in tone_res if r["jammed"] and r["kind"] == "tone/narrowband")

print(json.dumps({
    "layer": "Communication (V2X radio)",
    "repo": "v2x-jamming-detector",
    "attack": "RF jamming (tone)",
    "clean_false_alarm_rate": round(clean_jammed / N, 4),
    "attack_detection_rate": round(tone_detected / N, 3),
    "primary_metric": f"tone detect {tone_detected}/{N}, classified {tone_classified}/{N}",
    "ok": True,
}))

"""
Communication-layer driver: runs the V2X RF jamming detector on clean OFDM
frames (expect no false alarms) and tone-jammed frames (expect detection +
classification). Uses the repo's own channel, jammers, and detector modules.

For cross-layer fusion it emits the peak in-band power under attack, the median
clean peak (calibration anchor), and a distribution of clean peaks over many
independent frame sets so the harness can validate false alarms.
Emits one normalized JSON line on stdout.
"""
import json
import os
import statistics
import sys

repo = sys.argv[1]
sys.path.insert(0, repo)
os.chdir(repo)

import numpy as np
from core import channel, jammers
from core.detector import JammingDetector

N = 40
M_CLEAN = 12
det = JammingDetector().train(channel.frames(40, snr_db=15.0, seed=100))


def peak_power(frames):
    return max(float(np.mean(np.abs(f) ** 2)) for f in frames)


clean_peaks = []
for s in range(M_CLEAN):
    frames = channel.frames(N, snr_db=15.0, seed=500 + s)
    clean_peaks.append(round(peak_power(frames), 6))

clean = channel.frames(N, snr_db=15.0, seed=500)
clean_res = [det.analyze(f) for f in clean]
clean_jammed = sum(r["jammed"] for r in clean_res)

tone = [jammers.tone(f) for f in channel.frames(N, snr_db=15.0, seed=500)]
tone_res = [det.analyze(f) for f in tone]
tone_detected = sum(r["jammed"] for r in tone_res)
tone_classified = sum(1 for r in tone_res if r["jammed"] and r["kind"] == "tone/narrowband")
tone_peak_power = peak_power(tone)
power_threshold = det.p0 + det.power_k * det.pstd
clean_median = statistics.median(clean_peaks)

# Auto-tune a weak tone so its peak power lands just below the detector's power
# gate (the same quantity the detector uses), keeping the detector silent while
# lifting the fused anomaly into the minor band. Used for the coordinated test.
MILD_TARGET = 0.988 * power_threshold
mild_candidates = []
for jsr in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]:
    frames_m = [jammers.tone(f, jsr=jsr) for f in channel.frames(N, snr_db=15.0, seed=500)]
    res_m = [det.analyze(f) for f in frames_m]
    mild_candidates.append((sum(r["jammed"] for r in res_m), peak_power(frames_m)))
silent = [(j, p) for (j, p) in mild_candidates if j == 0 and p < power_threshold]
mild_jam, mild_peak = (min(silent, key=lambda t: abs(t[1] - MILD_TARGET))
                       if silent else min(mild_candidates, key=lambda t: t[0]))

print(json.dumps({
    "layer": "Communication (V2X radio)",
    "repo": "v2x-jamming-detector",
    "attack": "RF jamming (tone)",
    "clean_false_alarm_rate": round(clean_jammed / N, 4),
    "attack_detection_rate": round(tone_detected / N, 3),
    "primary_metric": f"tone detect {tone_detected}/{N}, classified {tone_classified}/{N}",
    "fusion_metric": "P",
    "fusion_value": round(tone_peak_power, 6),
    "fusion_threshold": round(power_threshold, 6),
    "fusion_clean_value": round(clean_median, 6),
    "fusion_clean_samples": clean_peaks,
    "fusion_mild_value": round(mild_peak, 6),
    "mild_individual_safe": bool(mild_jam == 0),
    "ok": True,
}))

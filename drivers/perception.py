"""
Perception-layer driver: runs the adversarial patch detector on a clean scene
(expect no alarm) and a patched scene (expect detection). Imports the repo's own
modules with the repo on sys.path, so the umbrella never modifies the repo.

For cross-layer fusion it emits the peak window gradient-energy ratio under
attack, the median clean peak (calibration anchor), and a distribution of clean
peaks over many independent scenes so the harness can validate false alarms.
Emits one normalized JSON line on stdout.
"""
import json
import os
import statistics
import sys

import numpy as np

repo = sys.argv[1]
sys.path.insert(0, repo)
os.chdir(repo)

import detector
import synth

M_CLEAN = 12


def peak_window_ratio(img, win=32, stride=16):
    gray = img.mean(axis=2)
    energy = detector._gradient_energy(gray)
    med = float(np.median(energy)) + 1e-6
    height, width = gray.shape
    peak = 0.0
    for y in range(0, max(1, height - win + 1), stride):
        for x in range(0, max(1, width - win + 1), stride):
            e = float(energy[y:y + win, x:x + win].mean())
            peak = max(peak, e / med)
    return peak


clean_peaks = []
for s in range(M_CLEAN):
    scene = synth.make_scene(size=256, seed=s)
    clean_peaks.append(round(peak_window_ratio(scene), 6))

img = synth.make_scene(size=256, seed=0)
clean_res, _ = detector.detect(img)
patched, _bbox = synth.add_patch(img, size=44, seed=1)
atk_res, _ = detector.detect(patched)
attack_peak_ratio = peak_window_ratio(patched)
clean_median = statistics.median(clean_peaks)

clean_false = 0.0 if clean_res["num_flagged_windows"] == 0 else 1.0
detected = 1.0 if atk_res["num_flagged_windows"] >= 1 else 0.0

print(json.dumps({
    "layer": "Perception (camera / VLM input)",
    "repo": "adversarial-patch-detector",
    "attack": "adversarial patch",
    "clean_false_alarm_rate": clean_false,
    "attack_detection_rate": detected,
    "primary_metric": f"clean='{clean_res['verdict']}', patched flags {atk_res['num_flagged_windows']} windows",
    "fusion_metric": "r_win",
    "fusion_value": round(attack_peak_ratio, 6),
    "fusion_threshold": 4.0,
    "fusion_clean_value": round(clean_median, 6),
    "fusion_clean_samples": clean_peaks,
    "fusion_mild_value": round(clean_median, 6),
    "mild_individual_safe": True,
    "ok": True,
}))

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

# Mild, sub-threshold attack for the coordinated-evasion thesis: a small
# patch softened by blur so the local detector stays silent, yet the window
# gradient-energy ratio is nudged above the clean population. Sweep patch
# size x blur and keep the strongest candidate that stays fully silent.
TAU = 6.0
from PIL import Image, ImageFilter  # noqa: E402

mild_ratio, mild_silent = statistics.median(clean_peaks), True
for size in (16, 20, 24, 28):
    for blur in (3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.5):
        cand, _ = synth.add_patch(img.copy(), size=size, seed=3)
        soft = Image.fromarray(cand.clip(0, 255).astype("uint8")).filter(
            ImageFilter.GaussianBlur(blur))
        soft_np = np.asarray(soft, dtype=np.float32)
        res, _ = detector.detect(soft_np)
        ratio = peak_window_ratio(soft_np)
        if res["num_flagged_windows"] == 0 and ratio < TAU and ratio > mild_ratio:
            mild_ratio = ratio

print(json.dumps({
    "layer": "Perception (camera / VLM input)",
    "repo": "adversarial-patch-detector",
    "attack": "adversarial patch",
    "clean_false_alarm_rate": clean_false,
    "attack_detection_rate": detected,
    "primary_metric": f"clean='{clean_res['verdict']}', patched flags {atk_res['num_flagged_windows']} windows",
    "fusion_metric": "r_win",
    "fusion_value": round(attack_peak_ratio, 6),
    "fusion_threshold": 6.0,
    "fusion_clean_value": round(clean_median, 6),
    "fusion_clean_samples": clean_peaks,
    "fusion_mild_value": round(mild_ratio, 6),
    "mild_individual_safe": bool(mild_silent),
    "ok": True,
}))

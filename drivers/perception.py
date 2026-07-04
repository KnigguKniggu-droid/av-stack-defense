"""
Perception-layer driver: runs the adversarial patch detector on a clean scene
(expect no alarm) and a patched scene (expect detection). Imports the repo's own
modules with the repo on sys.path, so the umbrella never modifies the repo.
Emits one normalized JSON line on stdout.
"""
import json
import os
import sys

repo = sys.argv[1]
sys.path.insert(0, repo)
os.chdir(repo)

import detector
import synth

img = synth.make_scene(size=256, seed=0)
clean_res, _ = detector.detect(img)
patched, _bbox = synth.add_patch(img, size=44, seed=1)
atk_res, _ = detector.detect(patched)

clean_false = 0.0 if clean_res["num_flagged_windows"] == 0 else 1.0
detected = 1.0 if atk_res["num_flagged_windows"] >= 1 else 0.0

print(json.dumps({
    "layer": "Perception (camera / VLM input)",
    "repo": "adversarial-patch-detector",
    "attack": "adversarial patch",
    "clean_false_alarm_rate": clean_false,
    "attack_detection_rate": detected,
    "primary_metric": f"clean='{clean_res['verdict']}', patched flags {atk_res['num_flagged_windows']} windows",
    "ok": True,
}))

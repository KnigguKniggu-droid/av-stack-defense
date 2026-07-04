"""Collect the perception scene: clean image, patched image, and detected boxes."""
import os
import sys

import numpy as np

repo, datadir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
os.chdir(repo)

import synth
import detector

img = synth.make_scene(256, 0)
patched, tb = synth.add_patch(img, 44, seed=1)
res, _ = detector.detect(patched)

regions = res.get("top_regions", []) or []
boxes = np.array([[r["x"], r["y"], r["w"], r["h"]] for r in regions], dtype=float) \
    if regions else np.zeros((0, 4))
top = regions[0] if regions else {"energy_ratio": 0.0, "saturation": 0.0, "score": 0.0}

np.savez(os.path.join(datadir, "perc.npz"),
         clean=img.astype(np.uint8), patched=patched.astype(np.uint8),
         true_bbox=np.array([tb["x"], tb["y"], tb["w"], tb["h"]], dtype=float),
         det_boxes=boxes,
         verdict=str(res.get("verdict", "")),
         flagged=int(res.get("num_flagged_windows", 0)),
         median_energy=float(res.get("global_median_energy", 0.0)),
         top_ratio=float(top["energy_ratio"]), top_sat=float(top["saturation"]),
         top_score=float(top["score"]), ratio_thresh=4.0, sat_weight=0.5)
print("perc saved")

"""Collect the perception scene: clean image, patched image, and detected boxes."""
import os
import sys
import numpy as np
from pathlib import Path

# Get the adversarial-patch-detector repo path
repo = r"C:\Users\ganes\adversarial-patch-detector"
datadir = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\ganes\av-stack-defense\viz\data"

# Add our detector to Python path
sys.path.insert(0, repo)
os.chdir(repo)

# Import our actual modules
from synth import make_scene, add_patch
from detector import detect

# Generate test scene (matches original parameters)
img = make_scene(256, 0)  # Clean 256x256 image
patched, tb = add_patch(img, 44, seed=1)  # Add 44px patch at seed=1

# Run our actual detector
res, heatmap = detect(patched)

# Extract detection results in expected format
regions = res.get("top_regions", [])
boxes = np.array([[r["x"], r["y"], r["w"], r["h"]] for r in regions], dtype=float) \
    if len(regions) > 0 else np.zeros((0, 4))

# Get top detection or default
top = regions[0] if regions else {
    "energy_ratio": 0.0,
    "saturation": 0.0,
    "score": 0.0
}

# Save results in the format the dashboard expects
np.savez(os.path.join(datadir, "perc.npz"),
         clean=img.astype(np.uint8),
         patched=patched.astype(np.uint8),
         true_bbox=np.array([tb["x"], tb["y"], tb["w"], tb["h"]], dtype=float),
         det_boxes=boxes,
         verdict=str(res.get("verdict", "")),
         flagged=int(res.get("num_flagged_windows", 0)),
         median_energy=float(res.get("global_median_energy", 0.0)),
         top_ratio=float(top["energy_ratio"]),
         top_sat=float(top["saturation"]),
         top_score=float(top["score"]),
         ratio_thresh=4.0,  # Default threshold from detector
         sat_weight=0.5)    # Default weight from detector

print("PERC: Adversarial patch detector data saved")
print(f"  Verdict: {res.get('verdict', 'N/A')}")
print(f"  Flagged windows: {res.get('num_flagged_windows', 0)}")
print(f"  Detection boxes: {len(boxes)} found")
if len(boxes) > 0:
    print(f"  First box: {boxes[0]}")

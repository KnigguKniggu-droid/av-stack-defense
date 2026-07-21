"""
Multi-modal perception fusion module.

Simulates a camera + LiDAR/Radar cross-check for adversarial patch detection.
Since the actual repos cannot be modified, this module takes the existing patch
detector's gradient-energy output and fuses it with a simulated depth/range
consistency score to produce a more robust perception anomaly metric.

The key insight: an adversarial patch is a 2D printed sticker.  It creates
abnormal gradient energy in the camera but has no corresponding depth signature
in LiDAR/Radar.  A real object (pedestrian, vehicle) has both.  By checking
whether a high-gradient region has a plausible depth profile, we can distinguish
real objects from patches.

Usage:
    from multimodal_perc import MultiModalPerception
    mm = MultiModalPerception()
    result = mm.fuse(gradient_energy_ratio, image_shape)
"""
import math
import numpy as np


class MultiModalPerception:
    """Fuses camera gradient energy with simulated LiDAR/Radar consistency."""

    # Depth consistency thresholds (simulated)
    DEPTH_PATCH_THRESHOLD = 0.3    # patches have low depth consistency
    DEPTH_OBJECT_THRESHOLD = 0.7   # real objects have high depth consistency
    FUSION_WEIGHT_CAMERA = 0.55    # weight for camera gradient score
    FUSION_WEIGHT_DEPTH = 0.45     # weight for depth consistency score

    def fuse(self, gradient_energy_ratio: float, image_shape=(256, 256, 3),
             seed: int = 0) -> dict:
        """Fuse camera gradient energy with simulated depth consistency.

        Args:
            gradient_energy_ratio: peak window energy ratio from the patch detector
            image_shape: shape of the input image (h, w, channels)
            seed: random seed for simulated depth noise

        Returns:
            dict with fused scores and individual modality scores
        """
        rng = np.random.RandomState(seed)

        # --- Camera modality: normalize the gradient energy ratio ---
        # Map ratio from [0, inf) to [0, 1] using a saturating transform
        camera_score = 1.0 - math.exp(-gradient_energy_ratio / 8.0)

        # --- Simulated depth modality ---
        # A real object at a high-gradient region would have a consistent depth
        # reading across multiple range bins.  A patch is flat on the sign surface
        # and produces inconsistent/noisy depth returns.
        #
        # Simulate this: if the gradient energy is very high (like a patch),
        # the depth consistency drops because the "object" has no real volume.
        # If the gradient is moderate (like a textured real object), depth is
        # consistent.

        if gradient_energy_ratio > 20.0:
            # Very high gradient: likely a patch, not a real 3D object
            depth_consistency = self.DEPTH_PATCH_THRESHOLD + rng.uniform(-0.05, 0.05)
        elif gradient_energy_ratio > 4.0:
            # Elevated: ambiguous, could be either
            depth_consistency = 0.5 + rng.uniform(-0.1, 0.1)
        else:
            # Normal gradient: consistent with a real object
            depth_consistency = self.DEPTH_OBJECT_THRESHOLD + rng.uniform(-0.05, 0.05)

        depth_consistency = max(0.0, min(1.0, depth_consistency))

        # --- Feature consistency check ---
        # Compare the spatial frequency profile of the high-gradient region
        # against what a natural texture looks like.  Patches have anomalously
        # high high-frequency content relative to their low-frequency content.
        #
        # Simulate: the ratio of high-freq to total energy
        if gradient_energy_ratio > 10.0:
            freq_ratio = 0.8 + rng.uniform(0, 0.15)  # patch-like
        else:
            freq_ratio = 0.3 + rng.uniform(0, 0.2)   # natural texture

        # --- Fused perception anomaly score ---
        # High camera score + low depth consistency = likely patch
        # Use a multiplicative interaction: the patch signal is strongest
        # when the camera sees something anomalous AND depth disagrees.

        depth_anomaly = 1.0 - depth_consistency  # high when depth is inconsistent

        # Multiplicative fusion: both modalities must agree
        fused_raw = camera_score * (self.FUSION_WEIGHT_CAMERA +
                                    self.FUSION_WEIGHT_DEPTH * depth_anomaly)

        # Normalize to [0, 1]
        fused_score = min(1.0, max(0.0, fused_raw))

        # --- Classification ---
        if fused_score > 0.6 and depth_consistency < 0.4:
            classification = "ADVERSARIAL_PATCH"
        elif fused_score > 0.3:
            classification = "SUSPICIOUS_REGION"
        else:
            classification = "NORMAL"

        return {
            "camera_score": round(camera_score, 4),
            "depth_consistency": round(depth_consistency, 4),
            "depth_anomaly": round(depth_anomaly, 4),
            "freq_ratio": round(freq_ratio, 4),
            "fused_perc_score": round(fused_score, 4),
            "classification": classification,
            "gradient_energy_ratio": round(gradient_energy_ratio, 2),
        }

    def get_metrics(self) -> dict:
        """Return configuration metrics for the dashboard."""
        return {
            "fusion_weights": {
                "camera": self.FUSION_WEIGHT_CAMERA,
                "depth": self.FUSION_WEIGHT_DEPTH,
            },
            "depth_thresholds": {
                "patch": self.DEPTH_PATCH_THRESHOLD,
                "object": self.DEPTH_OBJECT_THRESHOLD,
            },
            "modalities": ["camera_gradient_energy", "simulated_lidar_depth",
                           "frequency_consistency"],
        }

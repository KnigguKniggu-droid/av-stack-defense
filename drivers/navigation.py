"""
Navigation-layer driver: runs the EKF GPS-spoof detector on a clean drive
(expect ~no false alarms) and a jump-spoof drive (expect detection). Reuses the
repo's own sim, EKF, and detector modules; the detection loop mirrors the repo's
own cli.run so no repo file is modified.

For cross-layer fusion it also emits:
  * fusion_value        : peak post-event NIS under the full jump spoof
  * fusion_clean_value  : the median clean peak NIS (the calibration anchor)
  * fusion_clean_samples: peak NIS over many independent clean drives
  * fusion_mild_value   : peak NIS under a mild, sub-threshold spoof
  * mild_individual_safe: whether the layer's own detector stays silent on it
The detector trips on a 5-sample NIS average or a CUSUM, while the fusion metric
is the peak NIS, so a brief single-sample spike can lift the peak into the minor
band while the detector's average and CUSUM stay silent. The mild magnitude is
auto-tuned here so one harness run converges.
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
from core.sim import true_trajectory, imu_measurements, gps_measurements, DT
from core import attacks
from core.ekf import EKF
from core.detector import SpoofDetector

M_CLEAN = 12
GATE = SpoofDetector().threshold          # 9.21 chi-square(2) 99% gate
MILD_TARGET = 9.05                        # target peak NIS just below the gate
MILD_SWEEP = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]


def run(mode, seed=None, mild_mag=0.0):
    if seed is not None:
        np.random.seed(seed)
    states, accel, yaw = true_trajectory(60.0)
    a_meas, w_meas = imu_measurements(accel, yaw)
    gps0 = gps_measurements(states)
    start_k = int(len(states) * 0.5)

    if mode == "jump":
        gps = attacks.jump(gps0, start_k)
    elif mode == "mild":
        # a single-sample position spike: a transient the peak sees but the
        # 5-sample average and the CUSUM largely do not.
        gps = {k: v.copy() for k, v in gps0.items()}
        spike_k = min(gps, key=lambda k: abs(k - start_k))
        gps[spike_k] = gps[spike_k] + np.array([mild_mag, 0.0])
    else:
        gps = gps0

    ekf, det = EKF(DT), SpoofDetector()
    ekf.x[:2] = gps[min(gps)]
    n_alerts = post_flag = post_gps = total_gps = 0
    peak_post_nis = 0.0
    first = None
    for k in range(len(states)):
        ekf.predict(a_meas[k], w_meas[k])
        if k in gps:
            total_gps += 1
            nis, _ = ekf.update_gps(gps[k])
            spoof, _ = det.update(nis)
            if k >= start_k:
                peak_post_nis = max(peak_post_nis, float(nis))
                post_gps += 1
                post_flag += int(spoof)
            if spoof and first is None and k >= start_k:
                first = k
            n_alerts += int(spoof and k < start_k)
    return {"n_alerts": n_alerts, "total_gps": total_gps, "detected": first is not None,
            "post_flag": post_flag, "post_gps": post_gps, "peak_post_nis": peak_post_nis}


clean_peaks = [round(run("none", seed=2000 + s)["peak_post_nis"], 6) for s in range(M_CLEAN)]
clean = run("none", seed=2000)
jump = run("jump", seed=7000)

# auto-tune the mild spoof to land the peak NIS just below the gate, silent.
candidates = [(mag, run("mild", seed=7000, mild_mag=mag)) for mag in MILD_SWEEP]
silent = [(m, r) for (m, r) in candidates if not r["detected"] and r["peak_post_nis"] < GATE]
pool = silent or candidates
mag, mild = min(pool, key=lambda mr: abs(mr[1]["peak_post_nis"] - MILD_TARGET))

clean_fa = round(clean["n_alerts"] / clean["total_gps"], 4) if clean["total_gps"] else None
det_rate = round(jump["post_flag"] / jump["post_gps"], 3) if jump["post_gps"] else None

print(json.dumps({
    "layer": "Navigation (GPS + IMU fusion)",
    "repo": "ekf-gps-spoof-detector",
    "attack": "GPS spoof (jump)",
    "clean_false_alarm_rate": clean_fa,
    "attack_detection_rate": det_rate,
    "primary_metric": f"jump detected={jump['detected']}, post-attack flagged {jump['post_flag']}/{jump['post_gps']}",
    "fusion_metric": "d_nis",
    "fusion_value": round(jump["peak_post_nis"], 6),
    "fusion_threshold": GATE,
    "fusion_clean_value": round(statistics.median(clean_peaks), 6),
    "fusion_clean_samples": clean_peaks,
    "fusion_mild_value": round(mild["peak_post_nis"], 6),
    "mild_individual_safe": not mild["detected"],
    "ok": True,
}))

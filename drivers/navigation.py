"""
Navigation-layer driver: runs the EKF GPS-spoof detector on a clean drive
(expect ~no false alarms) and a jump-spoof drive (expect detection). Reuses the
repo's own sim, EKF, and detector modules; the detection loop mirrors the repo's
own cli.run so no repo file is modified.
Emits one normalized JSON line on stdout.
"""
import json
import os
import sys

repo = sys.argv[1]
sys.path.insert(0, repo)
os.chdir(repo)

import numpy as np
from core.sim import true_trajectory, imu_measurements, gps_measurements, DT
from core import attacks
from core.ekf import EKF
from core.detector import SpoofDetector

ATTACKS = {
    "none": lambda g, sk: g,
    "jump": lambda g, sk: attacks.jump(g, sk),
}


def run(attack):
    states, accel, yaw = true_trajectory(60.0)
    a_meas, w_meas = imu_measurements(accel, yaw)
    gps = ATTACKS[attack](gps_measurements(states), int(len(states) * 0.5))
    start_k = int(len(states) * 0.5)

    ekf, det = EKF(DT), SpoofDetector()
    ekf.x[:2] = gps[min(gps)]
    n_alerts = pre = post_flag = post_gps = total_gps = 0
    first = None
    for k in range(len(states)):
        ekf.predict(a_meas[k], w_meas[k])
        if k in gps:
            total_gps += 1
            nis, _ = ekf.update_gps(gps[k])
            spoof, _ = det.update(nis)
            if spoof:
                n_alerts += 1
                if k < start_k:
                    pre += 1
                elif first is None:
                    first = k
            if k >= start_k:
                post_gps += 1
                post_flag += int(spoof)
    return {"n_alerts": n_alerts, "total_gps": total_gps, "detected": first is not None,
            "post_flag": post_flag, "post_gps": post_gps}


clean = run("none")
jump = run("jump")

clean_fa = round(clean["n_alerts"] / clean["total_gps"], 4) if clean["total_gps"] else None
det_rate = round(jump["post_flag"] / jump["post_gps"], 3) if jump["post_gps"] else None

print(json.dumps({
    "layer": "Navigation (GPS + IMU fusion)",
    "repo": "ekf-gps-spoof-detector",
    "attack": "GPS spoof (jump)",
    "clean_false_alarm_rate": clean_fa,
    "attack_detection_rate": det_rate,
    "primary_metric": f"jump detected={jump['detected']}, post-attack flagged {jump['post_flag']}/{jump['post_gps']}",
    "ok": True,
}))

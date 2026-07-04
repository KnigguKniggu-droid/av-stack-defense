"""Collect the GPS spoofing run: true path, spoofed GPS, EKF estimate, detection."""
import os
import sys

import numpy as np

repo, datadir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
os.chdir(repo)

from core.sim import true_trajectory, imu_measurements, gps_measurements, DT
from core import attacks
from core.ekf import EKF
from core.detector import SpoofDetector

states, accel, yaw = true_trajectory(60.0)
a_meas, w_meas = imu_measurements(accel, yaw)
start_k = int(len(states) * 0.5)
gps = attacks.jump(gps_measurements(states), start_k)

ekf, det = EKF(DT), SpoofDetector()
ekf.x[:2] = gps[min(gps)]
est = np.zeros((len(states), 2))
gps_k, gps_xy = [], []
detect_k = -1
for k in range(len(states)):
    ekf.predict(a_meas[k], w_meas[k])
    if k in gps:
        nis, _ = ekf.update_gps(gps[k])
        spoof, _ = det.update(nis)
        gps_k.append(k)
        gps_xy.append(list(gps[k]))
        if spoof and detect_k < 0 and k >= start_k:
            detect_k = k
    est[k] = ekf.x[:2]

np.savez(os.path.join(datadir, "nav.npz"),
         true_xy=states[:, :2], est_xy=est,
         gps_k=np.array(gps_k), gps_xy=np.array(gps_xy),
         start_k=start_k, detect_k=detect_k)
print("nav saved")

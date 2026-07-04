"""Collect V2X jamming: per-frame received power over a clean-then-jammed stream,
the detector's real threshold, and the clean/jammed spectra for an inset."""
import os
import sys

import numpy as np

repo, datadir = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
os.chdir(repo)

from core import channel, jammers
from core.detector import JammingDetector

det = JammingDetector().train(channel.frames(40, snr_db=15.0, seed=100))
clean = list(channel.frames(40, snr_db=15.0, seed=500))
jammed = [jammers.tone(f) for f in channel.frames(40, snr_db=15.0, seed=500)]
stream = clean + jammed  # first 40 honest frames, then 40 jammed

powers = np.array([float(np.mean(np.abs(f) ** 2)) for f in stream])
threshold = float(det.p0 + det.power_k * det.pstd)

# Real classification stats on one jammed frame, straight from the detector.
jinfo = det.analyze(jammed[0])


def spec(s):
    return 20.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(s))) + 1e-9)


np.savez(os.path.join(datadir, "comm.npz"),
         powers=powers, threshold=threshold, jam_start=len(clean),
         clean_spec=spec(clean[0]), jammed_spec=spec(jammed[0]),
         p0=float(det.p0), pstd=float(det.pstd), power_k=float(det.power_k),
         clean_power=float(np.mean(powers[:len(clean)])),
         jammed_power=float(np.mean(powers[len(clean):])),
         jam_flatness=float(jinfo["flatness"]), jam_occbw=float(jinfo["occupied_bw"]),
         jam_kind=str(jinfo["kind"]))
print("comm saved")

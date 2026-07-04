"""Collect the FPGA CAN IDS waveform: compile and run the Verilog testbench to
produce can_ids.vcd, then parse the real signal traces for a timing diagram."""
import os
import re
import shutil
import subprocess
import sys

import numpy as np

repo, datadir = sys.argv[1], sys.argv[2]
iverilog = shutil.which("iverilog") or r"C:\iverilog\bin\iverilog.exe"
vvp = shutil.which("vvp") or r"C:\iverilog\bin\vvp.exe"

sim = os.path.join(repo, "can_ids_sim.vvp")
vcd = os.path.join(repo, "can_ids.vcd")
rtl = os.path.join(repo, "rtl", "can_ids.v")
tb = os.path.join(repo, "tb", "can_ids_tb.v")

subprocess.run([iverilog, "-o", sim, rtl, tb], check=True, capture_output=True, text=True)
subprocess.run([vvp, sim], cwd=repo, capture_output=True, text=True)


def parse_vcd(path, wanted):
    """Return {signal_name: [(time, value), ...]} for the wanted signal names."""
    id_to_name, name_to_id = {}, {}
    scale = 1
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    # header
    i = 0
    for i, ln in enumerate(lines):
        m = re.match(r"\$var\s+\w+\s+\d+\s+(\S+)\s+([^\s\[]+)", ln)
        if m:
            sid, name = m.group(1), m.group(2)
            if name in wanted:
                id_to_name[sid] = name
                name_to_id[name] = sid
        if "$enddefinitions" in ln:
            break
    series = {n: [] for n in wanted}
    t = 0
    for ln in lines[i:]:
        ln = ln.strip()
        if not ln:
            continue
        if ln[0] == "#":
            t = int(ln[1:])
        elif ln[0] in "01":
            sid = ln[1:]
            if sid in id_to_name:
                series[id_to_name[sid]].append((t, int(ln[0])))
        elif ln[0] == "b":
            val, sid = ln[1:].split()
            if sid in id_to_name:
                try:
                    series[id_to_name[sid]].append((t, int(val, 2)))
                except ValueError:
                    pass
    return series


wanted = ["frame_valid", "timing_alert", "unknown_alert"]
series = parse_vcd(vcd, wanted)

# Sample each signal at N evenly spaced points across the real timeline. VCD time
# is in ps here (timescale 1ps) and the sim runs for millions of ps, so we sample
# rather than expand every tick.
def value_at(pairs, times):
    out = np.zeros(len(times))
    idx, cur = 0, 0
    for i, t in enumerate(times):
        while idx < len(pairs) and pairs[idx][0] <= t:
            cur = pairs[idx][1]
            idx += 1
        out[i] = 1 if cur else 0
    return out


real_tmax = 0
for n in wanted:
    if series[n]:
        real_tmax = max(real_tmax, series[n][-1][0])
real_tmax = max(real_tmax, 1)

N = 2500
times = np.linspace(0, real_tmax, N)
np.savez(os.path.join(datadir, "fpga.npz"),
         frame_valid=value_at(series["frame_valid"], times),
         timing_alert=value_at(series["timing_alert"], times),
         unknown_alert=value_at(series["unknown_alert"], times),
         times_ns=times / 1000.0, tmax_ns=real_tmax / 1000.0,
         # Real constants from the RTL (rtl/can_ids.v) and testbench:
         min_period_cycles=80,      # 0x0C0 engine RPM, minimum inter-arrival
         inject_gap_cycles=22,      # testbench injects the second 0x0C0 ~22 cycles later
         clk_mhz=100)
print(f"fpga saved (tmax {real_tmax/1000.0:.0f} ns, "
      f"alerts t={int(series['timing_alert'][-1][1]) if series['timing_alert'] else 0})")

"""
In-vehicle-network driver (hardware): compiles and runs the synthesizable Verilog
CAN IDS testbench under Icarus Verilog, and reports whether its self-checking
testbench passes (no false alarms on normal traffic, alerts on attacks). If
Icarus Verilog is not installed, this layer is reported as skipped, not failed.
Emits one normalized JSON line on stdout.
"""
import json
import os
import shutil
import subprocess
import sys

repo = sys.argv[1]
iverilog = shutil.which("iverilog")
vvp = shutil.which("vvp")

base = {
    "layer": "In-vehicle network (CAN bus, FPGA/RTL)",
    "repo": "canbus-ids-fpga",
    "attack": "CAN injection + flood (hardware, 1-cycle latency)",
}

if not iverilog or not vvp:
    base.update({"clean_false_alarm_rate": None, "attack_detection_rate": None,
                 "primary_metric": "skipped: Icarus Verilog not installed (winget install Icarus.Verilog)",
                 "ok": None})
    print(json.dumps(base))
    sys.exit(0)

sim = os.path.join(repo, "can_ids_sim.vvp")
rtl = os.path.join(repo, "rtl", "can_ids.v")
tb = os.path.join(repo, "tb", "can_ids_tb.v")
try:
    subprocess.run([iverilog, "-o", sim, rtl, tb], check=True, capture_output=True, text=True)
    r = subprocess.run([vvp, sim], capture_output=True, text=True)
    passed = "RESULT: PASS" in r.stdout
    base.update({
        "clean_false_alarm_rate": 0.0 if passed else None,
        "attack_detection_rate": 1.0 if passed else 0.0,
        "primary_metric": "self-checking testbench PASS at 1-cycle latency" if passed else "testbench FAIL",
        "ok": bool(passed),
    })
except Exception as exc:
    base.update({"clean_false_alarm_rate": None, "attack_detection_rate": None,
                 "primary_metric": f"error: {exc}", "ok": False})
print(json.dumps(base))

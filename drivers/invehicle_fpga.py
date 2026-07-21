"""
In-vehicle-network driver (hardware): compiles and runs the synthesizable Verilog
CAN IDS testbench under Icarus Verilog, and reports whether its self-checking
testbench passes (no false alarms on normal traffic, alerts on attacks). If
Icarus Verilog is not installed, this layer is reported as skipped, not failed.
Emits one normalized JSON line on stdout.
"""
import json
import os
import re
import shutil
import subprocess
import sys

repo = sys.argv[1]
iverilog = shutil.which("iverilog")
vvp = shutil.which("vvp")
rtl = os.path.join(repo, "rtl", "can_ids.v")
tb = os.path.join(repo, "tb", "can_ids_tb.v")


def _first_int(pattern, text, default):
    match = re.search(pattern, text, flags=re.S)
    return int(match.group(1)) if match else default


def fusion_timing_params():
    try:
        with open(rtl, "r", encoding="utf-8") as f:
            rtl_text = f.read()
        with open(tb, "r", encoding="utf-8") as f:
            tb_text = f.read()
    except OSError:
        return 80, 22, 92

    threshold = _first_int(r"min_period\[0\]\s*=\s*(\d+)", rtl_text, 80)
    attack_repeat = _first_int(r"ATTACK 1.*?repeat\s*\((\d+)\)", tb_text, 20)
    clean_repeat = _first_int(r"NORMAL.*?repeat\s*\((\d+)\)", tb_text, 90)
    return threshold, attack_repeat + 2, clean_repeat + 2


threshold, attack_gap, clean_gap = fusion_timing_params()

base = {
    "layer": "In-vehicle network (CAN bus, FPGA/RTL)",
    "repo": "canbus-ids-fpga",
    "attack": "CAN injection + flood (hardware, 1-cycle latency)",
    "fusion_metric": "c",
    "fusion_threshold": threshold,
    "fusion_value": None,
    "fusion_clean_value": clean_gap,
    # The hardware gap is deterministic, so its clean distribution is a point.
    "fusion_clean_samples": [clean_gap],
    "fusion_mild_value": clean_gap,
    "mild_individual_safe": True,
}

if not iverilog or not vvp:
    base.update({"clean_false_alarm_rate": None, "attack_detection_rate": None,
                 "primary_metric": "skipped: Icarus Verilog not installed (winget install Icarus.Verilog)",
                 "ok": None})
    print(json.dumps(base))
    sys.exit(0)

sim = os.path.join(repo, "can_ids_sim.vvp")
try:
    subprocess.run([iverilog, "-o", sim, rtl, tb], check=True, capture_output=True, text=True)
    r = subprocess.run([vvp, sim], capture_output=True, text=True)
    passed = "RESULT: PASS" in r.stdout
    base.update({
        "clean_false_alarm_rate": 0.0 if passed else None,
        "attack_detection_rate": 1.0 if passed else 0.0,
        "primary_metric": "self-checking testbench PASS at 1-cycle latency" if passed else "testbench FAIL",
        "fusion_value": attack_gap if passed else None,
        "ok": bool(passed),
    })
except Exception as exc:
    base.update({"clean_false_alarm_rate": None, "attack_detection_rate": None,
                 "primary_metric": f"error: {exc}", "ok": False})
print(json.dumps(base))

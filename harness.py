#!/usr/bin/env python3
"""
Cross-layer AV defense harness.

Runs each layer's detector through a common interface and reports one comparable
metrics table across the whole autonomous-vehicle stack: perception, navigation,
communication, and the in-vehicle network (software and hardware).

Each layer runs in its own subprocess with its source repo on sys.path, so the
five original repositories are used unmodified and their local `core` packages
never collide. Every driver emits one normalized JSON record:

    layer, repo, attack, clean_false_alarm_rate, attack_detection_rate,
    primary_metric, ok

Run:
    python harness.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The five repos live one level up, alongside this umbrella folder.
ROOT = os.path.dirname(HERE)
PY = sys.executable

# Ordered top of stack (perception) down to the wire (in-vehicle network).
LAYERS = [
    ("drivers/perception.py",     os.path.join(ROOT, "adversarial-patch-detector")),
    ("drivers/navigation.py",     os.path.join(ROOT, "ekf-gps-spoof-detector")),
    ("drivers/communication.py",  os.path.join(ROOT, "v2x-jamming-detector")),
    ("drivers/invehicle_can.py",  os.path.join(ROOT, "canbus-ids")),
    ("drivers/invehicle_fpga.py", os.path.join(ROOT, "canbus-ids-fpga")),
]


def run_layer(driver, repo):
    path = os.path.join(HERE, driver)
    if not os.path.isdir(repo):
        return {"layer": driver, "repo": os.path.basename(repo), "ok": False,
                "primary_metric": "repo not found", "clean_false_alarm_rate": None,
                "attack_detection_rate": None, "attack": "-"}
    try:
        proc = subprocess.run([PY, path, repo], capture_output=True, text=True, timeout=300)
        line = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
        if not line:
            return {"layer": driver, "repo": os.path.basename(repo), "ok": False,
                    "primary_metric": f"no output ({proc.stderr.strip()[:80]})",
                    "clean_false_alarm_rate": None, "attack_detection_rate": None, "attack": "-"}
        return json.loads(line[-1])
    except Exception as exc:
        return {"layer": driver, "repo": os.path.basename(repo), "ok": False,
                "primary_metric": f"error: {exc}", "clean_false_alarm_rate": None,
                "attack_detection_rate": None, "attack": "-"}


def fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main():
    print("\nCross-layer AV defense harness")
    print("Running each layer's detector through a common clean-vs-attack test.\n")

    results = []
    for driver, repo in LAYERS:
        r = run_layer(driver, repo)
        results.append(r)
        status = "ok" if r.get("ok") else ("skip" if r.get("ok") is None else "FAIL")
        print(f"  [{status:4}] {r.get('layer', driver)}")

    # comparable table
    print("\n" + "=" * 78)
    print(f"{'Layer':<34}{'Attack':<22}{'clean FA':>9}{'detect':>9}")
    print("-" * 78)
    for r in results:
        print(f"{r.get('layer','?')[:33]:<34}{str(r.get('attack','-'))[:21]:<22}"
              f"{fmt(r.get('clean_false_alarm_rate')):>9}{fmt(r.get('attack_detection_rate')):>9}")
    print("=" * 78)

    ok = sum(1 for r in results if r.get("ok"))
    skipped = sum(1 for r in results if r.get("ok") is None)
    print(f"\n{ok} layers detecting, {skipped} skipped, {len(results)} total.\n")
    for r in results:
        print(f"  - {r.get('repo','?')}: {r.get('primary_metric','')}")

    out = os.path.join(HERE, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

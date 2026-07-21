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
import math
import os
import random
import statistics
import subprocess
import sys

from dependency_graph import DependencyGraph
from mitigation import MitigationEngine
from bridge import SoCBridge

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

FUSION_CONFIG = {
    "weights_mode": "equal",
    "z0": 2.2,
    "T_low": 0.15,
    "T_star": 0.22,
    "T_crit": 0.45,
    "phi_minor": 0.30,
    "k_min": 2,
    "eps": 1e-9,
    "e_min": -6.0,
    "e_max": 6.0,
    "layers": {
        "adversarial-patch-detector": {
            "type": "logratio", "direction": "high", "tau": 4.0, "a": 0.42, "w": 0.20,
        },
        "ekf-gps-spoof-detector": {
            "type": "linear", "direction": "high", "tau": 9.21, "a": 4.60, "w": 0.20,
        },
        "v2x-jamming-detector": {
            "type": "logratio", "direction": "high", "tau": None, "a": 0.50, "w": 0.20,
        },
        "canbus-ids": {
            "type": "logratio", "direction": "high", "tau": None, "a": 0.30, "w": 0.20,
        },
        "canbus-ids-fpga": {
            "type": "logratio", "direction": "low", "tau": None, "a": 0.30, "w": 0.20,
        },
    },
}


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


def _fusion_public_config():
    return {
        "T_low": FUSION_CONFIG["T_low"],
        "T_star": FUSION_CONFIG["T_star"],
        "T_crit": FUSION_CONFIG["T_crit"],
        "phi_minor": FUSION_CONFIG["phi_minor"],
        "k_min": FUSION_CONFIG["k_min"],
        "z0": FUSION_CONFIG["z0"],
    }


def _raw_exceedance(kind, direction, value, tau):
    eps = FUSION_CONFIG["eps"]
    value = max(value, eps)
    tau = max(tau, eps)
    if kind == "linear":
        return value - tau
    ratio = math.log(value / tau)
    return ratio if direction == "high" else -ratio


def _scale_for_record(cfg, record, tau):
    clean_value = record.get("fusion_clean_value")
    if clean_value is None:
        return cfg["a"]
    raw_clean = _raw_exceedance(cfg["type"], cfg["direction"], float(clean_value), tau)
    if abs(raw_clean) <= FUSION_CONFIG["eps"]:
        return cfg["a"]
    return abs(raw_clean) / FUSION_CONFIG["z0"]


def _exceedance(kind, direction, value, tau, a):
    e = _raw_exceedance(kind, direction, value, tau) / a
    return max(FUSION_CONFIG["e_min"], min(FUSION_CONFIG["e_max"], e))


def _phi(e):
    return 1.0 / (1.0 + math.exp(-e))


def fuse(records, clean_mode=False, weights_mode="equal", coordinated_mode=False,
         eligible_repos=None):
    value_key = "fusion_clean_value" if clean_mode else "fusion_value"
    layers = []
    skipped_layers = []
    weight_sum = 0.0

    WEIGHT_PROFILES = {
        "equal": {
            "adversarial-patch-detector": 0.20,
            "ekf-gps-spoof-detector": 0.20,
            "v2x-jamming-detector": 0.20,
            "canbus-ids": 0.20,
            "canbus-ids-fpga": 0.20
        },
        "safety_critical": {
            "adversarial-patch-detector": 0.15,
            "ekf-gps-spoof-detector": 0.25,
            "v2x-jamming-detector": 0.15,
            "canbus-ids": 0.25,
            "canbus-ids-fpga": 0.20
        }
    }
    profile_weights = WEIGHT_PROFILES.get(weights_mode, WEIGHT_PROFILES["equal"])

    for record in records:
        repo = record.get("repo")
        cfg = FUSION_CONFIG["layers"].get(repo)
        value = record.get(value_key)
        if not cfg:
            skipped_layers.append(record.get("layer", repo or "?"))
            continue
        if (not record.get("ok") and record.get("ok") is not None) or (value is None and not coordinated_mode):
            # Allow FPGA to proceed in coordinated mode even if skipped, or filter it
            skipped_layers.append(record.get("layer", repo or "?"))
            continue

        tau = record.get("fusion_threshold", cfg["tau"])
        if tau is None:
            skipped_layers.append(record.get("layer", repo or "?"))
            continue

        tau = float(tau)
        a = _scale_for_record(cfg, record, tau)

        if coordinated_mode:
            # Honest thesis test: use each layer's REAL mild, sub-threshold value.
            # Navigation and Communication carry a genuine mild attack; the rest stay clean.
            mild = record.get("fusion_mild_value")
            value = float(mild) if mild is not None else float(record.get("fusion_clean_value") or tau)
        else:
            value = float(value)

        e = _exceedance(cfg["type"], cfg["direction"], value, tau, a)
        phi = _phi(e)
        w = profile_weights.get(repo, cfg["w"])

        layers.append({
            "layer": record.get("layer"),
            "repo": repo,
            "metric": record.get("fusion_metric"),
            "value": round(value, 6),
            "threshold": round(tau, 6),
            "exceedance": round(e, 3),
            "phi": round(phi, 3),
            "_phi": phi,
            "weight": w,
            "minor": FUSION_CONFIG["phi_minor"] <= phi < 0.5,
            "trip": phi >= 0.5,
            "coordination_eligible": (eligible_repos is None or repo in eligible_repos),
            "individual_safe": bool(record.get("mild_individual_safe", True)),
        })
        # Only coordination-eligible layers contribute to the fused score, so a
        # near-threshold layer cannot inflate T_joint on clean data.
        if eligible_repos is None or repo in eligible_repos:
            weight_sum += w

    if not layers:
        return {
            "joint_threat_score": 0.0,
            "level": "NOMINAL",
            "global_alarm": False,
            "alarm_reason": None,
            "n_minor": 0,
            "hard_trips": [],
            "weights_mode": weights_mode,
            "config": _fusion_public_config(),
            "used_layers": [],
            "skipped_layers": skipped_layers,
            "layers": [],
        }

    t_joint = 0.0
    for layer in layers:
        if layer["coordination_eligible"] and weight_sum > 0:
            normalized_weight = layer["weight"] / weight_sum
            contribution = normalized_weight * layer["_phi"]
        else:
            normalized_weight = 0.0
            contribution = 0.0
        t_joint += contribution
        del layer["_phi"]
        layer["weight"] = round(normalized_weight, 3)
        layer["contribution"] = round(contribution, 3)
        layer["in_score"] = layer["coordination_eligible"]

    n_minor = sum(1 for layer in layers if layer["minor"] or layer["trip"])
    # Only layers with clean margin (coordination-eligible) may vote in Rule B,
    # so near-threshold layers cannot manufacture a coordinated false alarm.
    n_minor_elig = sum(1 for layer in layers
                       if (layer["minor"] or layer["trip"]) and layer["coordination_eligible"])
    hard_trips = [layer["layer"] for layer in layers if layer["trip"]]
    # The global fused alarm is only driven by coordination-eligible layers.
    # A near-threshold (ineligible) layer still shows its score, but its own
    # local detector stays authoritative and it cannot raise the fused alarm.
    hard_trips_elig = [layer["layer"] for layer in layers
                       if layer["trip"] and layer["coordination_eligible"]]
    rule_a = len(hard_trips_elig) >= 1
    rule_b = (
        len(layers) >= FUSION_CONFIG["k_min"]
        and n_minor_elig >= FUSION_CONFIG["k_min"]
        and t_joint >= FUSION_CONFIG["T_star"]
    )

    if t_joint >= FUSION_CONFIG["T_crit"] or len(hard_trips_elig) >= 2:
        level = "CRITICAL"
    elif t_joint >= FUSION_CONFIG["T_star"] or rule_a:
        level = "ALERT"
    elif t_joint >= FUSION_CONFIG["T_low"]:
        level = "ELEVATED"
    else:
        level = "NOMINAL"

    reason = "single-layer" if rule_a else ("coordinated" if rule_b else None)
    result = {
        "joint_threat_score": round(t_joint, 3),
        "level": level,
        "global_alarm": rule_a or rule_b,
        "alarm_reason": reason,
        "n_minor": n_minor,
        "n_minor_eligible": n_minor_elig,
        "hard_trips": hard_trips,
        "hard_trips_eligible": hard_trips_elig,
        "weights_mode": weights_mode,
        "config": _fusion_public_config(),
        "used_layers": [layer["layer"] for layer in layers],
        "skipped_layers": skipped_layers,
        "layers": layers,
    }
    if coordinated_mode:
        all_safe = all(layer["individual_safe"] for layer in layers)
        result["all_individual_safe"] = all_safe
        # The core cross-layer thesis: every local detector stays silent, no
        # single layer trips, yet the fused score raises a coordinated alarm.
        result["thesis_proven"] = bool(
            all_safe and not hard_trips and reason == "coordinated"
        )
    return result


def _percentile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def _clean_phi_samples(cfg, samples, tau, a):
    return [_phi(_exceedance(cfg["type"], cfg["direction"], float(v), tau, a)) for v in samples]


def _calibrated_scale(cfg, samples, tau):
    med = statistics.median(samples)
    raw = _raw_exceedance(cfg["type"], cfg["direction"], float(med), tau)
    if abs(raw) <= FUSION_CONFIG["eps"]:
        return cfg["a"]
    return abs(raw) / FUSION_CONFIG["z0"]


def _fusable_clean_layers(records):
    out = []
    for record in records:
        repo = record.get("repo")
        cfg = FUSION_CONFIG["layers"].get(repo)
        samples = record.get("fusion_clean_samples")
        tau = record.get("fusion_threshold", cfg["tau"] if cfg else None)
        if not cfg or record.get("ok") is False or not samples or tau is None:
            continue
        out.append({"layer": record.get("layer"), "repo": repo, "cfg": cfg,
                    "tau": float(tau), "samples": [float(x) for x in samples]})
    return out


def coordination_eligibility(records):
    """A layer may vote in Rule B only if its clean 95th-percentile phi stays
    below phi_minor, i.e. it has genuine margin between clean and its threshold.
    This stops a near-threshold layer from manufacturing coordinated alarms."""
    eligible, stats = set(), []
    phi_minor = FUSION_CONFIG["phi_minor"]
    for L in _fusable_clean_layers(records):
        a = _calibrated_scale(L["cfg"], L["samples"], L["tau"])
        phis = _clean_phi_samples(L["cfg"], L["samples"], L["tau"], a)
        p95 = _percentile(phis, 0.95)
        is_elig = p95 < phi_minor
        if is_elig:
            eligible.add(L["repo"])
        stats.append({"layer": L["layer"], "repo": L["repo"], "a": round(a, 4),
                      "clean_phi_median": round(statistics.median(phis), 3),
                      "clean_phi_p95": round(p95, 3), "clean_phi_max": round(max(phis), 3),
                      "coordination_eligible": is_elig})
    return eligible, stats


def validate_false_alarm(records, eligible_repos, trials=8000, seed=0):
    """Honest held-out false-alarm estimate. Calibrate each layer's scale on a
    calibration split of its clean peaks, then Monte-Carlo the fused alarm over
    an independent held-out split, drawing one clean peak per layer per trial."""
    rng = random.Random(seed)
    phi_minor = FUSION_CONFIG["phi_minor"]
    T_star, k_min = FUSION_CONFIG["T_star"], FUSION_CONFIG["k_min"]

    bundles = []
    for L in _fusable_clean_layers(records):
        s = L["samples"]
        mid = max(1, len(s) // 2)
        calib, held = s[:mid], (s[mid:] or s[:mid])
        a = _calibrated_scale(L["cfg"], calib, L["tau"])
        bundles.append({"repo": L["repo"], "w": 0.20,
                        "held_phi": _clean_phi_samples(L["cfg"], held, L["tau"], a),
                        "eligible": L["repo"] in eligible_repos,
                        "calib_n": len(calib), "held_n": len(held)})

    if not bundles:
        return {"fused_false_alarm_rate": None, "note": "no fusable clean samples"}

    wsum = sum(b["w"] for b in bundles if b["eligible"]) or 1.0
    n_elig = sum(1 for b in bundles if b["eligible"])
    alarms, tj_list = 0, []
    for _ in range(trials):
        tj, n_minor_elig, hard = 0.0, 0, 0
        for b in bundles:
            phi = rng.choice(b["held_phi"])
            if not b["eligible"]:
                continue
            tj += (b["w"] / wsum) * phi
            if phi >= 0.5:
                hard += 1
            elif phi >= phi_minor:
                n_minor_elig += 1
        tj_list.append(tj)
        rule_a = hard >= 1
        rule_b = n_elig >= k_min and n_minor_elig >= k_min and tj >= T_star
        if rule_a or rule_b:
            alarms += 1

    return {
        "fused_false_alarm_rate": round(alarms / trials, 4),
        "trials": trials,
        "t_joint_median": round(statistics.median(tj_list), 4),
        "t_joint_p95": round(_percentile(tj_list, 0.95), 4),
        "t_joint_max": round(max(tj_list), 4),
        "calibration_split_per_layer": bundles[0]["calib_n"],
        "heldout_split_per_layer": bundles[0]["held_n"],
        "method": "50/50 clean-peak split; calibrate on split A, Monte-Carlo fused alarm on split B",
    }


def apply_cross_layer_intelligence(fusion_result, dep_graph, bridge):
    """Apply dependency graph propagation, mitigation evaluation, and bridge
    register updates to a fusion result.  Returns the enhanced fusion dict
    with new keys: propagation, mitigation, bridge_state."""
    # 1. Extract raw per-layer phi values
    raw_phis = {}
    for L in fusion_result.get("layers", []):
        raw_phis[L["repo"]] = L.get("phi", 0.10)

    # 2. Run dependency graph propagation
    propagation = dep_graph.propagate(raw_phis)
    adjusted_phis = {k: v for k, v in propagation.items()
                     if k in ["adversarial-patch-detector", "ekf-gps-spoof-detector",
                              "v2x-jamming-detector", "canbus-ids", "canbus-ids-fpga"]}

    # 3. Compute propagation-adjusted T_joint
    wsum = sum(L.get("weight", 0.20) for L in fusion_result.get("layers", [])
               if L.get("coordination_eligible", L.get("in_score", True)))
    if wsum > 0:
        t_propagated = sum(
            (L.get("weight", 0.20) / wsum) * adjusted_phis.get(L["repo"], L.get("phi", 0.10))
            for L in fusion_result.get("layers", [])
            if L.get("coordination_eligible", L.get("in_score", True))
        )
    else:
        t_propagated = fusion_result.get("joint_threat_score", 0.0)

    # 4. Run mitigation on the propagated state
    # Build a modified fusion result with propagated values for mitigation
    mit_fusion = dict(fusion_result)
    mit_fusion["joint_threat_score"] = t_propagated
    # Re-evaluate level based on propagated score
    if t_propagated >= 0.45:
        mit_fusion["level"] = "CRITICAL"
    elif t_propagated >= 0.22:
        mit_fusion["level"] = "ALERT"
    elif t_propagated >= 0.15:
        mit_fusion["level"] = "ELEVATED"
    else:
        mit_fusion["level"] = "NOMINAL"

    engine = MitigationEngine()
    mitigation = engine.evaluate(mit_fusion)

    # 5. Apply to bridge
    bridge.reset()
    bridge.apply_mitigation(mitigation)

    # 6. Enhance the fusion result
    fusion_result["propagation"] = {
        "raw_phis": propagation.get("raw_phis", {}),
        "adjusted_phis": {k: round(v, 3) for k, v in adjusted_phis.items()},
        "deltas": propagation.get("propagation_deltas", {}),
        "t_joint_raw": round(fusion_result.get("joint_threat_score", 0.0), 3),
        "t_joint_propagated": round(t_propagated, 3),
        "delta": round(t_propagated - fusion_result.get("joint_threat_score", 0.0), 4),
    }
    fusion_result["mitigation"] = mitigation
    fusion_result["bridge_state"] = bridge.get_state()
    fusion_result["graph_data"] = dep_graph.get_graph_data()

    return fusion_result


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

    # Coordination eligibility from clean margin, shared by every profile.
    eligible, elig_stats = coordination_eligibility(results)

    # 1. Equal Weights Profile
    fusion = fuse(results, weights_mode="equal", eligible_repos=eligible)
    fusion_clean = fuse(results, clean_mode=True, weights_mode="equal", eligible_repos=eligible)
    fusion_coord = fuse(results, coordinated_mode=True, weights_mode="equal", eligible_repos=eligible)

    # 2. Safety Critical Weights Profile
    fusion_safety = fuse(results, weights_mode="safety_critical", eligible_repos=eligible)
    fusion_clean_safety = fuse(results, clean_mode=True, weights_mode="safety_critical", eligible_repos=eligible)
    fusion_coord_safety = fuse(results, coordinated_mode=True, weights_mode="safety_critical", eligible_repos=eligible)

    # 3. Honest held-out false-alarm validation
    validation = validate_false_alarm(results, eligible)
    validation["per_layer"] = elig_stats
    validation["coordination_eligible_layers"] = sorted(eligible)

    print("\n" + "=" * 78)
    print("Cross-Layer Signal Fusion Evaluation (Dynamic Calibration)")
    print("-" * 78)
    print("  WEIGHTS: EQUAL PROFILE (0.20 per layer)")
    print(f"    - Clean state:       {fusion_clean['joint_threat_score']:.3f} ({fusion_clean['level']}, alarm={fusion_clean['global_alarm']})")
    print(f"    - Coordinated evasion: {fusion_coord['joint_threat_score']:.3f} ({fusion_coord['level']}, alarm={fusion_coord['global_alarm']})")
    print(f"    - Saturated attack:    {fusion['joint_threat_score']:.3f} ({fusion['level']}, alarm={fusion['global_alarm']})")
    print("  WEIGHTS: SAFETY-CRITICAL PROFILE")
    print(f"    - Clean state:       {fusion_clean_safety['joint_threat_score']:.3f} ({fusion_clean_safety['level']}, alarm={fusion_clean_safety['global_alarm']})")
    print(f"    - Coordinated evasion: {fusion_coord_safety['joint_threat_score']:.3f} ({fusion_coord_safety['level']}, alarm={fusion_coord_safety['global_alarm']})")
    print(f"    - Saturated attack:    {fusion_safety['joint_threat_score']:.3f} ({fusion_safety['level']}, alarm={fusion_safety['global_alarm']})")
    print("=" * 78)

    print("\nCoordination eligibility (clean margin, p95 phi < phi_minor):")
    for s in elig_stats:
        flag = "eligible " if s["coordination_eligible"] else "INELIGIBLE"
        print(f"  [{flag}] {s['layer'][:34]:<34} clean phi p95={s['clean_phi_p95']:.3f}")

    print("\nCoordinated cross-layer thesis (equal weights):")
    print(f"  every local detector silent : {fusion_coord.get('all_individual_safe')}")
    print(f"  no single layer tripped     : {not fusion_coord.get('hard_trips')}")
    print(f"  fused joint score           : {fusion_coord['joint_threat_score']:.3f} ({fusion_coord['level']})")
    print(f"  coordinated alarm raised    : {fusion_coord['alarm_reason'] == 'coordinated'}")
    print(f"  THESIS PROVEN               : {fusion_coord.get('thesis_proven')}")

    fa = validation.get("fused_false_alarm_rate")
    print(f"\nHeld-out fused false-alarm rate: {fa}  "
          f"(over {validation.get('trials')} clean trials, "
          f"T_joint p95={validation.get('t_joint_p95')})")
    print("=" * 78)

    # 4. Cross-layer intelligence: dependency graph, mitigation, bridge
    dep_graph = DependencyGraph()
    bridge = SoCBridge()

    print("\n" + "=" * 78)
    print("Cross-Layer Intelligence: Propagation + Mitigation + SoC Bridge")
    print("-" * 78)

    for label, fus in [("Saturated attack", fusion),
                       ("Coordinated evasion", fusion_coord),
                       ("Clean drive", fusion_clean)]:
        apply_cross_layer_intelligence(fus, dep_graph, bridge)
        prop = fus.get("propagation", {})
        mit = fus.get("mitigation", {})
        print(f"\n  [{label}]")
        print(f"    T_joint raw={prop.get('t_joint_raw', '?')} -> "
              f"propagated={prop.get('t_joint_propagated', '?')} "
              f"(delta={prop.get('delta', '?')})")
        print(f"    Mitigation: {mit.get('state', '?')} ({mit.get('state_label', '?')})")
        for a in mit.get("actions", []):
            print(f"      -> {a['target']}: {a['action']}")
        print(f"    Bridge writes: {fus.get('bridge_state', {}).get('write_count', 0)}")

    # Apply to safety profiles too
    apply_cross_layer_intelligence(fusion_safety, dep_graph, bridge)
    apply_cross_layer_intelligence(fusion_coord_safety, dep_graph, bridge)
    apply_cross_layer_intelligence(fusion_clean_safety, dep_graph, bridge)

    print("\n" + "=" * 78)

    # Equal Weights Writes
    fusion_out = os.path.join(HERE, "fusion.json")
    with open(fusion_out, "w", encoding="utf-8") as f:
        json.dump(fusion, f, indent=2)
    print(f"Wrote {fusion_out}")

    fusion_clean_out = os.path.join(HERE, "fusion_clean.json")
    with open(fusion_clean_out, "w", encoding="utf-8") as f:
        json.dump(fusion_clean, f, indent=2)
    print(f"Wrote {fusion_clean_out}")

    fusion_coord_out = os.path.join(HERE, "fusion_coordinated.json")
    with open(fusion_coord_out, "w", encoding="utf-8") as f:
        json.dump(fusion_coord, f, indent=2)
    print(f"Wrote {fusion_coord_out}")

    # Safety Weights Writes
    fusion_safety_out = os.path.join(HERE, "fusion_safety.json")
    with open(fusion_safety_out, "w", encoding="utf-8") as f:
        json.dump(fusion_safety, f, indent=2)
    print(f"Wrote {fusion_safety_out}")

    fusion_clean_safety_out = os.path.join(HERE, "fusion_clean_safety.json")
    with open(fusion_clean_safety_out, "w", encoding="utf-8") as f:
        json.dump(fusion_clean_safety, f, indent=2)
    print(f"Wrote {fusion_clean_safety_out}")

    fusion_coord_safety_out = os.path.join(HERE, "fusion_coordinated_safety.json")
    with open(fusion_coord_safety_out, "w", encoding="utf-8") as f:
        json.dump(fusion_coord_safety, f, indent=2)
    print(f"Wrote {fusion_coord_safety_out}")

    validation_out = os.path.join(HERE, "fusion_validation.json")
    with open(validation_out, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"Wrote {validation_out}")

    # Cross-layer intelligence outputs
    graph_data = dep_graph.get_graph_data()
    graph_out = os.path.join(HERE, "dependency_graph.json")
    with open(graph_out, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
    print(f"Wrote {graph_out}")

    # Write the mitigation state from the saturated attack scenario
    mit_out = os.path.join(HERE, "mitigation_state.json")
    with open(mit_out, "w", encoding="utf-8") as f:
        json.dump(fusion.get("mitigation", {}), f, indent=2)
    print(f"Wrote {mit_out}")

    # Write bridge state
    bridge_out = os.path.join(HERE, "bridge_state.json")
    with open(bridge_out, "w", encoding="utf-8") as f:
        json.dump(fusion.get("bridge_state", {}), f, indent=2)
    print(f"Wrote {bridge_out}")

    # Write propagation data for all three scenarios
    propagation_out = os.path.join(HERE, "propagation.json")
    with open(propagation_out, "w", encoding="utf-8") as f:
        json.dump({
            "attack": fusion.get("propagation", {}),
            "coordinated": fusion_coord.get("propagation", {}),
            "clean": fusion_clean.get("propagation", {}),
        }, f, indent=2)
    print(f"Wrote {propagation_out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

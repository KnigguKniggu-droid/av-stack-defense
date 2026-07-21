# FUSION_SPEC.md — Cross-Layer Signal Fusion

Design spec for the Joint Threat Score that fuses the five detector layers into a
single cross-layer alarm. This document defines the math and the data contract.
It is the design deliverable (Step 1). Implementation in `harness.py` (Step 2) and
the dashboard panel (Step 3) consume the contract defined in section 8.

## 1. Goal

Each layer already trips its own local alarm when its metric crosses a hard
threshold. Those local alarms are blind to each other. A capable adversary can
stay just under every individual threshold while nudging several layers at once,
and no single detector fires.

Cross-layer fusion closes that gap. It maps every layer's raw metric onto one
common anomaly scale, combines them into a Joint Threat Score `T_joint`, and
raises a global alarm when several layers are jointly elevated, even if none of
them crosses its own threshold.

## 2. Inputs: the raw per-layer metric

Fusion needs a continuous scalar per layer, not the pass/fail summary in the
current `results.json`. Each driver must surface the scalar below. The name in
the last column is the metric symbol used throughout this spec.

| Layer | Driver | Raw scalar to surface | Symbol | Direction |
|---|---|---|---|---|
| Perception | `drivers/perception.py` | peak window gradient-energy ratio | `r_win` | high = anomalous |
| Navigation | `drivers/navigation.py` | peak Normalized Innovation Squared | `d_nis` | high = anomalous |
| Communication | `drivers/communication.py` | occupancy power in the band | `P` | high = anomalous |
| CAN (software) | `drivers/invehicle_can.py` | observed bus message rate | `rho` | high = anomalous |
| CAN (FPGA/RTL) | `drivers/invehicle_fpga.py` | minimum inter-injection cycle gap | `c` | low = anomalous |

"Direction" states which way is a threat. For the FPGA layer a smaller gap means
frames arriving too close together, which is the injection or flood signature, so
low is anomalous.

## 3. Per-layer normalization phi_i

Every layer is reduced to a bounded anomaly value `phi_i` in the open interval
(0, 1) through two steps: a signed, dimensionless exceedance `e_i`, then a
logistic squash.

### 3.1 Signed exceedance

The exceedance is zero exactly at the layer's own hard threshold `tau_i`,
positive when the metric is more anomalous, negative when it is safer. Two
transform types cover all five layers:

- Log-ratio (scale-free, for ratio and power metrics), direction high:

  $$e_i = \frac{1}{a_i}\,\ln\!\left(\frac{m_i}{\tau_i}\right)$$

- Log-ratio, direction low (FPGA gap):

  $$e_i = \frac{1}{a_i}\,\ln\!\left(\frac{\tau_i}{m_i}\right)$$

- Linear (for the chi-square NIS, which is additive in its own units),
  direction high:

  $$e_i = \frac{m_i - \tau_i}{a_i}$$

`a_i > 0` is a per-layer scale (the spread of `e_i`). Clamp `e_i` to `[-6, 6]`
before the next step so a single saturated layer cannot dominate.

### 3.2 Logistic squash

$$\phi_i = \sigma(e_i) = \frac{1}{1 + e^{-e_i}}$$

By construction `phi_i = 0.5` exactly at the layer's own hard threshold, rises
toward 1 as the metric exceeds it, and decays toward 0 in the safe region. Every
layer therefore contributes exactly 0.5 at its own decision boundary, which is
what makes the layers comparable and the weights meaningful.

### 3.3 Calibrating the scale a_i from clean data

Do not hand-pick `a_i`. Calibrate it so the clean, no-attack median of each metric
maps to a low common anomaly floor `phi_i ≈ 0.10`, i.e. `e_i ≈ -2.2`:

$$a_i = \frac{\left| e_i^{\text{raw}}(\text{clean median}) \right|}{z_0}, \qquad z_0 = 2.2$$

where `e_i^raw` is the exceedance from 3.1 evaluated at the clean-run median with
`a_i = 1`. This ties the scale to the real clean-run spread and gives a uniform
clean floor across layers. The defaults in section 4 assume a clean ratio of
about `0.4 * tau` for the high-direction ratio metrics; replace them with
calibrated values once a clean run is available.

## 4. Default FUSION_CONFIG

Central configuration, one row per layer. Keep this in `harness.py` so all tuning
lives in one place.

| Layer | type | direction | tau_i | a_i (default) | w_i (equal) |
|---|---|---|---|---|---|
| Perception | logratio | high | 4.0 | 0.42 | 0.20 |
| Navigation | linear | high | 9.21 | 4.60 | 0.20 |
| Communication | logratio | high | tau_comm | 0.50 | 0.20 |
| CAN (software) | logratio | high | rho_0 * (1 + 0.25) | 0.30 | 0.20 |
| CAN (FPGA) | logratio | low | c_min | 0.30 | 0.20 |

Notes:

- `tau = 9.21` for Navigation is the 0.99 quantile of the chi-square distribution
  with 2 degrees of freedom, which is the existing NIS gate.
- The CAN software threshold is the normal rate `rho_0` plus a 25 percent margin,
  so a mild rate rise reads as a minor anomaly rather than an instant trip.
- `tau_comm` and `c_min` are read from the communication and FPGA drivers.
- Weights are equal by default and must sum to 1. A safety-weighted alternative
  that raises Navigation and CAN (both directly affect vehicle control) is
  `[0.15, 0.25, 0.15, 0.25, 0.20]`. State which mode is active in the output.

## 5. Joint Threat Score

$$T_{\text{joint}} = \sum_{i=1}^{5} w_i\,\phi_i, \qquad \sum_i w_i = 1$$

`T_joint` lies in (0, 1). It is a convex blend of the per-layer anomalies, so it
is directly comparable across runs and easy to visualize as a single meter. Each
layer's `contribution_i = w_i * phi_i` is reported so the panel can show a stacked
breakdown.

## 6. Global alarm logic

Two independent triggers, combined with OR.

- Rule A, single-layer hard trip: `max_i phi_i >= 0.5`. Some layer crossed its own
  threshold. This reproduces the existing per-layer alarms.
- Rule B, coordinated multi-layer: `N_minor >= k_min` AND `T_joint >= T_star`,
  where `N_minor = #{ i : phi_i >= phi_minor }` counts elevated-but-not-tripped
  layers. This is the new capability: it fires on several simultaneous minor
  anomalies even when no single layer crosses its threshold.

Defaults: `phi_minor = 0.30`, `k_min = 2`, `T_star = 0.22`.

### 6.1 Threat level for the panel

Map `T_joint` to a discrete level for color and glow, independent of the boolean
alarm:

| Level | Condition |
|---|---|
| NOMINAL | `T_joint < 0.15` |
| ELEVATED | `0.15 <= T_joint < 0.22` |
| ALERT | `T_joint >= 0.22` or Rule A fires |
| CRITICAL | `T_joint >= 0.45` or two or more layers satisfy Rule A |

### 6.2 Why these numbers

Worked cases with equal weights and a clean floor `phi ≈ 0.10`:

- All clean: `T ≈ 0.20 * 5 * 0.10 = 0.10`. Below every band. No alarm.
- One layer exactly at its threshold (`phi = 0.5`), rest clean:
  `T = 0.20*0.5 + 0.80*0.10 = 0.18`. Rule A fires on that layer, but the
  coordinated Rule B does not (`0.18 < 0.22`), so a lone layer cannot masquerade
  as a cross-layer event.
- Two minor layers at `phi = 0.40`, rest clean:
  `T = 0.20*(0.40+0.40) + 0.20*3*0.10 = 0.22`. `N_minor = 2 >= k_min`, so Rule B
  fires. This is the coordinated case the fusion exists to catch.
- Three minor layers at `phi = 0.40`: `T ≈ 0.28`, firmly in ALERT.

`T_star = 0.22` sits above the lone-layer case (0.18) and at the two-minor-layer
case (0.22), which is exactly the separation we want.

## 7. Robustness

- Missing or failed layer (`ok` is false or null): drop it from the sum and
  renormalize the remaining weights to sum to 1, so a skipped layer neither
  inflates nor deflates the score. Record which layers were used.
- If fewer than `k_min` layers are available, disable Rule B and fall back to
  Rule A only.
- Clamp `e_i` to `[-6, 6]` (section 3.1). Guard the log transforms against
  non-positive inputs by flooring `m_i` and `tau_i` at a small epsilon.

## 8. Output contract (for Step 2 and Step 3)

`harness.py` writes a `fusion` object. Put it either as a sibling `fusion.json` or
as a top-level key alongside the existing per-layer array. Schema:

```json
{
  "joint_threat_score": 0.27,
  "level": "ALERT",
  "global_alarm": true,
  "alarm_reason": "coordinated",
  "n_minor": 3,
  "hard_trips": [],
  "weights_mode": "equal",
  "config": { "T_low": 0.15, "T_star": 0.22, "T_crit": 0.45,
              "phi_minor": 0.30, "k_min": 2, "z0": 2.2 },
  "layers": [
    { "layer": "Perception (camera / VLM input)", "metric": "r_win",
      "value": 3.4, "threshold": 4.0, "exceedance": -0.39, "phi": 0.40,
      "weight": 0.20, "contribution": 0.080, "minor": true, "trip": false }
  ]
}
```

- `alarm_reason` is `"single-layer"`, `"coordinated"`, or `null`.
- `hard_trips` lists the layers with `phi >= 0.5`.
- The `layers` array holds one entry per fused layer, in stack order.

The dashboard panel (Step 3) renders `joint_threat_score` as the glowing meter,
`level` as its color, and the per-layer `contribution` values as a stacked
breakdown, with `hard_trips` and `alarm_reason` driving the alert text.

## 9. Handoff to Codex (Step 2)

1. In each driver, surface the raw scalar from section 2 by adding two keys to the
   JSON record it already prints:

   ```json
   "fusion_metric": "r_win",
   "fusion_value": 3.4
   ```

   Source of each scalar: Perception = the maximum window gradient-energy ratio
   seen on the tested frame; Navigation = the peak NIS over the post-event window;
   Communication = the occupancy power in the monitored band; CAN software = the
   observed message rate; FPGA = the minimum inter-injection cycle gap in the
   testbench trace.

2. In `harness.py`, add `FUSION_CONFIG` (section 4) and a `fuse(records)` step
   after the layer loop that computes `e_i`, `phi_i`, `T_joint`, the alarm, and the
   level, then writes the section 8 object. Reference implementation below.

3. Keep all constants in `FUSION_CONFIG`. Do not scatter magic numbers.

## 10. Reference implementation

```python
import math

# type, direction, tau, a, w   (tau may be resolved at runtime from the driver)
FUSION_CONFIG = {
    "adversarial-patch-detector": ("logratio", "high", 4.0,   0.42, 0.20),
    "ekf-gps-spoof-detector":     ("linear",   "high", 9.21,  4.60, 0.20),
    "v2x-jamming-detector":       ("logratio", "high", None,  0.50, 0.20),
    "canbus-ids":                 ("logratio", "high", None,  0.30, 0.20),
    "canbus-ids-fpga":            ("logratio", "low",  None,  0.30, 0.20),
}
T_LOW, T_STAR, T_CRIT = 0.15, 0.22, 0.45
PHI_MINOR, K_MIN = 0.30, 2
EPS = 1e-9


def _exceedance(kind, direction, value, tau, a):
    value = max(value, EPS)
    tau = max(tau, EPS)
    if kind == "linear":
        e = (value - tau) / a
    else:  # logratio
        r = math.log(value / tau)
        e = r / a if direction == "high" else -r / a
    return max(-6.0, min(6.0, e))


def _phi(e):
    return 1.0 / (1.0 + math.exp(-e))


def fuse(records):
    layers, wsum = [], 0.0
    for r in records:
        cfg = FUSION_CONFIG.get(r.get("repo"))
        if not cfg or not r.get("ok") or r.get("fusion_value") is None:
            continue
        kind, direction, tau_default, a, w = cfg
        tau = r.get("fusion_threshold", tau_default)
        if tau is None:
            continue
        e = _exceedance(kind, direction, float(r["fusion_value"]), float(tau), a)
        phi = _phi(e)
        layers.append({
            "layer": r.get("layer"), "metric": r.get("fusion_metric"),
            "value": r["fusion_value"], "threshold": tau,
            "exceedance": round(e, 3), "phi": round(phi, 3), "weight": w,
            "trip": phi >= 0.5, "minor": PHI_MINOR <= phi < 0.5,
        })
        wsum += w

    if not layers:
        return {"joint_threat_score": 0.0, "level": "NOMINAL",
                "global_alarm": False, "alarm_reason": None, "layers": []}

    # renormalize weights over available layers
    t_joint = 0.0
    for L in layers:
        L["weight"] = L["weight"] / wsum
        L["contribution"] = round(L["weight"] * L["phi"], 3)
        t_joint += L["weight"] * L["phi"]

    n_minor = sum(1 for L in layers if L["phi"] >= PHI_MINOR)
    hard = [L["layer"] for L in layers if L["trip"]]
    rule_a = len(hard) >= 1
    rule_b = (len(layers) >= K_MIN) and (n_minor >= K_MIN) and (t_joint >= T_STAR)

    if t_joint >= T_CRIT or len(hard) >= 2:
        level = "CRITICAL"
    elif t_joint >= T_STAR or rule_a:
        level = "ALERT"
    elif t_joint >= T_LOW:
        level = "ELEVATED"
    else:
        level = "NOMINAL"

    reason = "single-layer" if rule_a else ("coordinated" if rule_b else None)
    return {
        "joint_threat_score": round(t_joint, 3), "level": level,
        "global_alarm": rule_a or rule_b, "alarm_reason": reason,
        "n_minor": n_minor, "hard_trips": hard, "weights_mode": "equal",
        "config": {"T_low": T_LOW, "T_star": T_STAR, "T_crit": T_CRIT,
                   "phi_minor": PHI_MINOR, "k_min": K_MIN, "z0": 2.2},
        "layers": layers,
    }
```

## 11. Validation

The fusion must not cost the project its clean record. Validate on clean, no-attack
runs and confirm the fused false-alarm rate is at or below the target (aim for
0.00, accept at most 0.01). If clean runs push `T_joint` near `T_star`, raise
`T_star` or recalibrate the `a_i` (section 3.3) until the clean floor sits near
0.10. Then confirm that a synthetic coordinated case, two or more layers held in
the minor band with none crossing its own threshold, produces a global alarm with
`alarm_reason = "coordinated"`.

## 12. Implemented extensions and validated results

The framework below is implemented in `harness.py` and the five drivers, and the
numbers are measured from a real run, not asserted.

### 12.1 Self-calibration from clean data (Option C)

Each driver now runs its clean scenario over many independent seeds and emits the
distribution of clean peaks as `fusion_clean_samples`, with the median as
`fusion_clean_value`. The harness sets each scale `a_i` so the clean median maps
to `phi = 0.10` (section 3.3). Calibration is therefore automatic for any
baseline, with no hand-tuned constants.

### 12.2 Coordination eligibility (false-alarm guard)

A layer may drive the global fused alarm only if it has genuine clean margin,
defined as clean 95th-percentile `phi < phi_minor`. A near-threshold layer is
marked ineligible: it is excluded from the fused score and from both alarm rules,
its own local detector stays authoritative, and it is still reported for context
with `in_score = false`. Measured eligibility:

| Layer | clean phi p95 | eligible |
|---|---|---|
| Perception (peak window ratio) | 0.327 | no |
| Navigation | 0.100 | yes |
| Communication | 0.100 | yes |
| CAN (software) | 0.101 | yes |
| CAN (FPGA) | 0.100 | yes |

Perception's peak-ratio metric sits too close to its own threshold on clean
frames to add cross-layer value, so it is honestly excluded rather than allowed
to manufacture false alarms.

### 12.3 Weight profiles (Option B)

Two selectable profiles, both summing to 1 over the participating layers:

- `equal`: 0.20 each.
- `safety_critical`: Perception 0.15, Navigation 0.25, Communication 0.15,
  CAN software 0.25, CAN FPGA 0.20, prioritizing the layers that command vehicle
  dynamics. The harness writes the fused state under both profiles.

### 12.4 Coordinated cross-layer attack, proven (Option A)

The core thesis is demonstrated through the real detectors, not a synthetic
control. Navigation carries a brief single-sample GPS spike auto-tuned to a peak
NIS of 8.997, just under the 9.21 gate, and Communication carries a weak tone
auto-tuned to a power just under its own power gate. Measured result:

| Layer | phi | state | local detector |
|---|---|---|---|
| Navigation | 0.414 | minor | silent |
| Communication | 0.355 | minor | silent |
| CAN, FPGA | 0.10 | clean | silent |
| Perception | 0.10 | excluded | silent |

Every local detector stays silent, no single layer trips, yet
`T_joint = 0.242` and the fused alarm fires with
`alarm_reason = "coordinated"` and `thesis_proven = true`. This is the
cross-layer thesis made concrete.

### 12.5 Held-out false-alarm validation (honest measurement)

`validate_false_alarm` splits each layer's clean peaks 50/50, calibrates the
scale on one split, and Monte-Carlo evaluates the fused alarm over the
independent held-out split (8000 trials), drawing one clean peak per eligible
layer per trial. Measured fused false-alarm rate: **0.0**, with clean
`T_joint` 95th percentile at 0.100. Because calibration and evaluation use
disjoint clean data, this is a real generalization estimate, not a value true by
construction.

### 12.6 Output files and added fields

`main()` writes: `results.json`, `fusion.json` and `fusion_safety.json`
(saturated attack, both profiles), `fusion_clean.json` and
`fusion_clean_safety.json`, `fusion_coordinated.json` and
`fusion_coordinated_safety.json`, and `fusion_validation.json`.

New fields beyond section 8: `n_minor_eligible`, `hard_trips_eligible`,
`used_layers`, `skipped_layers`; per-layer `coordination_eligible`, `in_score`,
`individual_safe`; and, in coordinated output, `all_individual_safe` and
`thesis_proven`. The panel should read `in_score` to know which layers form the
score and `coordination_eligible` to gray out context-only layers.

### 12.7 Driver contract additions

Each driver emits `fusion_clean_samples` (list of clean peaks),
`fusion_mild_value` (its value under a mild sub-threshold attack, equal to the
clean value for layers with no evadable mild attack), and `mild_individual_safe`
(whether its own detector stays silent on that mild value). CAN keeps its clean
value here because its timing IDS flags any injection regardless of rate, so it
cannot be a silent-but-elevated layer; its own detector remains the authority.

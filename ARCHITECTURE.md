# Architecture: Cross-Layer Defense for the Autonomous Vehicle Stack

This document maps each layer of a connected autonomous vehicle to the attack it
faces, the detector that defends it, and the signal that detector produces. It
then states the cross-layer thesis that motivates unifying these detectors, and
it is explicit about what the current code demonstrates versus what is future work.

## The stack as layers

A connected autonomous vehicle is a layered system. An attack can target any
layer, and the more dangerous attacks cross between layers. From the top of the
stack down to the wire:

```
   ┌──────────────────────────────────────────────────────────────┐
   │  Perception          camera / VLM input                        │
   │     attack: adversarial patch      detector: patch detector     │
   ├──────────────────────────────────────────────────────────────┤
   │  Navigation          GPS + IMU fusion                          │
   │     attack: GPS spoofing           detector: EKF + NIS/CUSUM     │
   ├──────────────────────────────────────────────────────────────┤
   │  Communication       V2X radio link                            │
   │     attack: RF jamming             detector: energy + spectral   │
   ├──────────────────────────────────────────────────────────────┤
   │  In-vehicle network  CAN bus (software and FPGA)               │
   │     attack: injection / flood / Bus-Off   detector: timing IDS   │
   └──────────────────────────────────────────────────────────────┘
```

## Layer-by-layer mapping

Each detector reduces its layer to a common shape: it learns what normal looks
like, then flags a deviation. That shared shape is what makes cross-layer fusion
possible later. The metrics below are produced by `harness.py` on each run.

| Layer | Attack modeled | Detector mechanism | Signal it emits | Measured (clean FA / detect) |
|---|---|---|---|---|
| Perception | Adversarial patch on a camera image | Local gradient-energy and saturation vs the scene median | Flagged high-frequency windows and a patch box | 0.000 / 1.000 |
| Navigation | GPS spoofing (jump) | Extended Kalman Filter fusing GPS and IMU, chi-square test on innovations plus CUSUM | Normalized innovation squared and a cumulative drift statistic | 0.000 / 1.000 |
| Communication | RF jamming (tone) | Energy detection plus spectral-shape classification on an OFDM link | Received power and spectral flatness and occupied bandwidth | 0.000 / 1.000 |
| In-vehicle network (software) | Flooding, injection, replay, Bus-Off | Per-message timing model, flag arrivals off the learned period | Timing residual and unknown-ID and silence flags | 0.000 / 1.000 |
| In-vehicle network (FPGA) | CAN injection and flood, in hardware | Same timing logic in synthesizable Verilog at single-cycle latency | Hardware timing and unknown-ID alerts | skipped without Icarus Verilog |

FA is the false-alarm rate on clean input. Detect is the detection rate on the
attack. Both are captured live by the harness, not asserted by hand.

## The cross-layer thesis

The reason to unify these detectors is that real attacks do not respect layer
boundaries, and a coordinated attacker can exploit that.

Transitive attack propagation is the core pattern. A manipulation at one layer
cascades into a failure at another. A perturbation at the perception layer, such
as a misread sign, can drive an incorrect control action that then appears as
anomalous traffic on the in-vehicle network. In the other direction, a timing
attack on the CAN bus can silence a legitimate ECU and seize a safety-critical
function without ever compromising it directly.

The defensive consequence is that a single-layer detector sees only its own slice
of the stack. An attacker who spreads a small, individually quiet perturbation
across several layers can stay under every single detector's threshold while still
causing a dangerous combined effect. Defending against that requires observing all
layers together and correlating their signals.

## What this code demonstrates today

- **Unification.** All layers run through one harness with one common interface,
  and each detector reports comparable clean-versus-attack metrics. This is the
  substrate a cross-layer defense needs, and it did not exist before: the five
  detectors were five separate programs.
- **Independent correctness.** On each layer's own attack, every software detector
  reports full detection with zero false alarms in the harness run. The hardware
  layer runs when Icarus Verilog is installed.
- **Non-invasive integration.** The five source repositories are used unmodified.
  Each driver adds the repo to its own subprocess path, so their local modules
  never collide and none of the original code changes.

## What is future work, and not yet claimed

- **Coordinated cross-layer attacks.** The harness currently runs each layer's
  attack independently. It does not yet inject a single attack that spans layers.
  Building that shared, time-synchronized attack scenario is the next step.
- **Signal fusion.** The detectors emit comparable signals, but the harness does
  not yet fuse them into a joint decision. The hypothesis that fusion catches
  coordinated attacks that evade every single-layer detector is stated, not proven.
- **Agentic orchestration.** A multi-agent controller that weighs layers and adapts
  to an adaptive attacker is the research direction, not part of this code.

This separation is deliberate. The umbrella is honest about being the integration
layer that makes the cross-layer research possible, without claiming the research
result before it is demonstrated.

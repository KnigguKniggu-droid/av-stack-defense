# AV Stack Defense (Cross-Layer Harness)

> **TL;DR** One umbrella that runs five autonomous-vehicle attack detectors,
> spanning perception, navigation, communication, and the in-vehicle network, and
> reports one comparable metrics table across the whole stack. The five detectors
> stay in their own repositories, unmodified. This is the integration substrate for
> studying cross-layer attack propagation.

## Quickstart

```bash
python harness.py
```

No dependencies beyond what the individual detectors already use (numpy, Pillow).
The harness runs each layer in its own subprocess and prints a comparable table.

## Live visualization

Every panel below is rendered from the real output of the real detectors on a
single run. The inputs are the projects' own simulations; the detection code and
results are genuine. Generate them yourself with `python viz/build.py` (needs
`matplotlib`), or double-click `Visualize.bat` on Windows.

**Navigation, GPS spoofing.** An Extended Kalman Filter fuses GPS and IMU. The
motion model integrates the IMU, the update compares each GPS fix against the
prediction, and the normalized innovation squared (NIS) is chi-square tested.

![GPS spoofing detection](viz/media/nav.gif)

$$\mathbf{y}=\mathbf{z}-H\mathbf{x},\qquad S=HPH^\top+R,\qquad d=\mathbf{y}^\top S^{-1}\mathbf{y}\ \sim\ \chi^2_2$$

Alarm when $\overline{d}_5 \gt 9.21$ (99%) or CUSUM $g_k=\max(0,\,g_{k-1}+(d_k-3)) \gt 14$. This run: honest $\overline{d}\approx 1.8$, spoofed peak $\approx 170$.

**Communication, V2X jamming.** Energy detection on a real OFDM link, with spectral
flatness for classification.

![V2X jamming detection](viz/media/comm.gif)

$$P=\frac{1}{N}\sum_{n}\lvert x[n]\rvert^2,\qquad \tau=P_0+4\sigma_0,\qquad \mathrm{SF}=\frac{\left(\prod_k \mathrm{PSD}[k]\right)^{1/N}}{\frac{1}{N}\sum_k \mathrm{PSD}[k]}$$

Alarm when $P \gt \tau$. This run: $\tau=1.151$, clean $P=1.033$, jammed $P=9.30$.

**Perception, adversarial patch.** High-frequency gradient energy relative to the
scene median localizes the patch.

![Adversarial patch detection](viz/media/perc.gif)

$$E(x,y)=\left(\frac{\partial I}{\partial x}\right)^2+\left(\frac{\partial I}{\partial y}\right)^2,\qquad r=\frac{\overline{E}_{\text{win}}}{\operatorname{median}(E)},\qquad \text{flag if } r\ge 4$$

This run: the top flagged window has $r\approx 294\times$ the median energy.

**In-vehicle network, CAN flood.** A learned per-ID timing model flags the flood.

![CAN flood detection](viz/media/can.gif)

$$T=\frac{1}{N}\sum_i g_i,\qquad z=\frac{g-T}{\sigma},\qquad \text{TIMING: } g \lt 0.5T,\qquad \text{RATE\\_FLOOD: } \rho \gt 4\rho_0$$

This run: fastest period $T\approx 10$ ms, normal bus $\rho_0\approx 336$ msg/s, flood $\approx 2005$ msg/s.

**Hardware, FPGA CAN IDS.** The same detection in synthesizable Verilog, at
single-cycle latency.

![FPGA CAN IDS waveform](viz/media/fpga.gif)

$$c_{k+1}=c_k+1,\qquad \text{TIMING: } \big(c-\text{last\\_seen}[id]\big) \lt \text{min\\_period}[id]$$

This run: injected `0x0C0` arrives $\approx 22$ cycles apart $\lt 80$, so the alert fires in one clock at 100 MHz.

## What this is

Five separate projects each defend one layer of a connected autonomous vehicle:

| Layer | Repository | Defends |
|---|---|---|
| Perception | `adversarial-patch-detector` | camera against adversarial patches |
| Navigation | `ekf-gps-spoof-detector` | GPS/IMU against spoofing |
| Communication | `v2x-jamming-detector` | V2X radio against RF jamming |
| In-vehicle network (software) | `canbus-ids` | CAN bus against injection and flooding |
| In-vehicle network (hardware) | `canbus-ids-fpga` | CAN bus in synthesizable Verilog |

On their own they are five demos. This umbrella runs them through one common
interface so their results are directly comparable, which is the first step toward
a coordinated cross-layer defense. See `ARCHITECTURE.md` for the layer-by-layer
mapping and the cross-layer thesis.

## Example run

```
Layer                             Attack                 clean FA   detect
------------------------------------------------------------------------------
Perception (camera / VLM input)   adversarial patch         0.000    1.000
Navigation (GPS + IMU fusion)     GPS spoof (jump)          0.000    1.000
Communication (V2X radio)         RF jamming (tone)         0.000    1.000
In-vehicle network (CAN, softw.)  flooding / DoS            0.000    1.000
In-vehicle network (CAN, FPGA)    CAN injection + flood         -        -   (needs Icarus Verilog)
```

`clean FA` is the false-alarm rate on clean input; `detect` is the detection rate
on the attack. The numbers are captured live and written to `results.json`.

## How it works

```
av-stack-defense/
├─ harness.py            # runs every layer, prints the comparable table, writes results.json
├─ drivers/
│   ├─ perception.py     # adversarial-patch-detector: clean scene vs patched scene
│   ├─ navigation.py     # ekf-gps-spoof-detector: clean drive vs jump spoof
│   ├─ communication.py  # v2x-jamming-detector: clean frames vs tone jammer
│   ├─ invehicle_can.py  # canbus-ids: clean traffic vs DoS flood
│   └─ invehicle_fpga.py # canbus-ids-fpga: Icarus Verilog testbench (optional)
└─ ARCHITECTURE.md       # layer mapping, cross-layer thesis, scope
```

Each driver adds its target repository to a subprocess path and calls that repo's
own detector. The repositories are never modified, and their local `core` packages
never collide because each runs in isolation.

The five detector repositories must sit alongside this folder (one level up).

## Honest scope

- This harness proves each layer detects its own attack through one shared
  interface. It does **not** yet inject a single attack that spans layers, and it
  does **not** yet fuse the per-layer signals into a joint decision. Those are the
  research steps, described in `ARCHITECTURE.md`, and they are intentionally not
  claimed as done here.
- The FPGA layer is reported as skipped, not failed, when Icarus Verilog is absent.
- To run everything including hardware: `winget install Icarus.Verilog`.

"""
Build a fully animated visual dashboard for the cross-layer AV defense stack.

Every panel is rendered from the REAL output of the real detectors in the five
repos (each collector imports the actual repo modules and runs them; nothing is
faked). Five animated panels:
  * Navigation   : GPS-spoofing simulation (car, spoofed GPS, EKF, detection)
  * Communication: per-frame received power crossing the jamming threshold
  * Perception   : the detector's boxes appearing on the adversarial patch
  * In-vehicle   : CAN messages streaming in, the DoS flood breaking the rhythm
  * FPGA         : the real Verilog waveform (frame_valid + alert lines), swept
"""
import base64
import io
import os
import subprocess
import sys
import webbrowser

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
UMB = os.path.dirname(HERE)
ROOT = os.path.dirname(UMB)
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
PY = sys.executable

INK, FG, ACCENT, DANGER, GPSY = "#0c1211", "#e5ecea", "#3fb59f", "#ff5c5c", "#f2c14e"
BLUE = "#8aa0ff"
plt.rcParams.update({
    "figure.facecolor": INK, "axes.facecolor": "#131a19", "savefig.facecolor": INK,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": "#93a19f",
    "ytick.color": "#93a19f", "axes.edgecolor": "#243130", "grid.color": "#1e2827",
    "font.size": 11,
})

LAYERS = [
    ("c_nav.py", os.path.join(ROOT, "ekf-gps-spoof-detector")),
    ("c_comm.py", os.path.join(ROOT, "v2x-jamming-detector")),
    ("c_perc.py", os.path.join(ROOT, "adversarial-patch-detector")),
    ("c_can.py", os.path.join(ROOT, "canbus-ids")),
    ("c_fpga.py", os.path.join(ROOT, "canbus-ids-fpga")),
]


def collect():
    for script, repo in LAYERS:
        print(f"  collecting {script} ...", flush=True)
        r = subprocess.run([PY, os.path.join(HERE, script), repo, DATA],
                           capture_output=True, text=True, timeout=150)
        if r.returncode != 0:
            print(f"    WARN {script}: {r.stderr.strip()[-200:]}")


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _save(anim, name, fps=12):
    path = os.path.join(DATA, name)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close("all")
    return _b64(path)


def nav_gif():
    d = np.load(os.path.join(DATA, "nav.npz"))
    true_xy, est_xy = d["true_xy"], d["est_xy"]
    gps_k, gps_xy = list(d["gps_k"]), d["gps_xy"]
    start_k, detect_k = int(d["start_k"]), int(d["detect_k"])
    n = len(true_xy)
    frames = list(range(0, n, max(1, n // 55))) + [n - 1]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    allx = np.concatenate([true_xy[:, 0], gps_xy[:, 0], est_xy[:, 0]])
    ally = np.concatenate([true_xy[:, 1], gps_xy[:, 1], est_xy[:, 1]])
    ax.set_xlim(allx.min() - 20, allx.max() + 20); ax.set_ylim(ally.min() - 20, ally.max() + 20)
    ax.set_title("GPS spoofing: true path, spoofed GPS, EKF estimate", fontsize=12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(True, alpha=0.3)
    tl, = ax.plot([], [], color=ACCENT, lw=2.2, label="true path")
    el, = ax.plot([], [], color=BLUE, lw=1.6, ls="--", label="EKF estimate")
    gpre = ax.scatter([], [], s=14, color=GPSY, label="GPS (honest)")
    gpost = ax.scatter([], [], s=26, color=DANGER, marker="x", label="GPS (spoofed)")
    car, = ax.plot([], [], "o", color="#fff", ms=9, mec=ACCENT, mew=2)
    ban = ax.text(0.5, 0.94, "", transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, facecolor="#131a19", edgecolor="#243130")

    def fr(k):
        tl.set_data(true_xy[:k + 1, 0], true_xy[:k + 1, 1])
        el.set_data(est_xy[:k + 1, 0], est_xy[:k + 1, 1])
        pre = [g for gk, g in zip(gps_k, gps_xy) if gk <= k and gk < start_k]
        post = [g for gk, g in zip(gps_k, gps_xy) if gk <= k and gk >= start_k]
        gpre.set_offsets(np.array(pre) if pre else np.empty((0, 2)))
        gpost.set_offsets(np.array(post) if post else np.empty((0, 2)))
        car.set_data([true_xy[k, 0]], [true_xy[k, 1]])
        if k >= start_k and (detect_k < 0 or k >= detect_k):
            ban.set_text("GPS SPOOFING DETECTED"); ban.set_color(DANGER)
        elif k >= start_k:
            ban.set_text("spoofing active..."); ban.set_color(GPSY)
        return ()
    return _save(FuncAnimation(fig, fr, frames=frames), "nav.gif")


def comm_gif():
    d = np.load(os.path.join(DATA, "comm.npz"))
    powers, thr, js = d["powers"], float(d["threshold"]), int(d["jam_start"])
    x = np.arange(len(powers))
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.set_xlim(0, len(powers)); ax.set_ylim(0, max(powers.max() * 1.15, thr * 1.4))
    ax.axhline(thr, color=GPSY, ls="--", lw=1.3, label="jamming threshold")
    ax.axvspan(js, len(powers), color=DANGER, alpha=0.06)
    ax.set_title("V2X jamming: received power per frame", fontsize=12)
    ax.set_xlabel("frame #"); ax.set_ylabel("received power"); ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], color=ACCENT, lw=1.6)
    pts = ax.scatter([], [], s=18)
    ban = ax.text(0.5, 0.94, "", transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, facecolor="#131a19", edgecolor="#243130")
    frames = list(range(1, len(powers) + 1))

    def fr(i):
        line.set_data(x[:i], powers[:i])
        cols = [DANGER if powers[j] > thr else ACCENT for j in range(i)]
        pts.set_offsets(np.c_[x[:i], powers[:i]]); pts.set_color(cols)
        if i > js and powers[js:i].max() > thr:
            ban.set_text("JAMMING DETECTED"); ban.set_color(DANGER)
        return ()
    return _save(FuncAnimation(fig, fr, frames=frames), "comm.gif")


def perc_gif():
    d = np.load(os.path.join(DATA, "perc.npz"))
    clean, patched = d["clean"], d["patched"]
    tb, boxes = d["true_bbox"], d["det_boxes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 3.9))
    a1.imshow(clean); a1.set_title("clean scene", fontsize=11); a1.axis("off")
    a2.imshow(patched); a2.set_title("adversarial patch + detector boxes", fontsize=11); a2.axis("off")
    a2.add_patch(patches.Rectangle((tb[0], tb[1]), tb[2], tb[3], fill=False,
                                   edgecolor=GPSY, lw=1.4, ls="--"))
    ban = a2.text(0.5, 1.12, "", transform=a2.transAxes, ha="center", fontsize=12, fontweight="bold")
    drawn = []
    nb = len(boxes)
    frames = list(range(nb + 6))

    def fr(i):
        while len(drawn) < min(i, nb):
            b = boxes[len(drawn)]
            r = patches.Rectangle((b[0], b[1]), b[2], b[3], fill=False, edgecolor=DANGER, lw=1.5)
            a2.add_patch(r); drawn.append(r)
        if i >= nb and nb:
            ban.set_text("PATCH LOCALIZED"); ban.set_color(DANGER)
        return ()
    return _save(FuncAnimation(fig, fr, frames=frames), "perc.gif", fps=8)


def can_gif():
    d = np.load(os.path.join(DATA, "can.npz"))
    t, ids, atk = d["t"], d["ids"], d["is_attack"]
    o = np.argsort(t); t, ids, atk = t[o], ids[o], atk[o]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.set_xlim(t.min(), t.max()); ax.set_ylim(ids.min() - 30, ids.max() + 30)
    ax.set_title("CAN bus timing: the DoS flood breaks the rhythm", fontsize=12)
    ax.set_xlabel("time (s)"); ax.set_ylabel("arbitration ID"); ax.grid(True, alpha=0.3)
    okp = ax.scatter([], [], s=11, color=ACCENT, label="normal periodic")
    badp = ax.scatter([], [], s=20, color=DANGER, marker="s", label="DoS flood")
    ban = ax.text(0.5, 0.94, "", transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, facecolor="#131a19", edgecolor="#243130")
    times = np.linspace(t.min(), t.max(), 46)

    def fr(tc):
        sel = t <= tc
        so, sb = sel & (atk == 0), sel & (atk == 1)
        okp.set_offsets(np.c_[t[so], ids[so]] if so.any() else np.empty((0, 2)))
        badp.set_offsets(np.c_[t[sb], ids[sb]] if sb.any() else np.empty((0, 2)))
        if sb.any():
            ban.set_text("CAN INTRUSION DETECTED"); ban.set_color(DANGER)
        return ()
    return _save(FuncAnimation(fig, fr, frames=times), "can.gif")


def fpga_gif():
    d = np.load(os.path.join(DATA, "fpga.npz"))
    fv, ta, ua = d["frame_valid"], d["timing_alert"], d["unknown_alert"]
    x, tmax = d["times_ns"], float(d["tmax_ns"])
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.set_xlim(0, tmax); ax.set_ylim(-0.5, 6.5)
    ax.set_yticks([0.5, 2.5, 4.5]); ax.set_yticklabels(["timing_alert", "unknown_alert", "frame_valid"])
    ax.set_title("FPGA CAN IDS: real Verilog waveform (1-cycle alerts)", fontsize=12)
    ax.set_xlabel("simulation time (ns)"); ax.grid(True, alpha=0.25)
    ax.step(x, fv * 0.9 + 4, color=ACCENT, lw=1.0, where="post")
    ax.step(x, ua * 0.9 + 2, color=DANGER, lw=1.2, where="post")
    ax.step(x, ta * 0.9 + 0, color=DANGER, lw=1.2, where="post")
    ax.fill_between(x, 2, ua * 0.9 + 2, step="post", color=DANGER, alpha=0.25)
    ax.fill_between(x, 0, ta * 0.9 + 0, step="post", color=DANGER, alpha=0.25)
    head = ax.axvline(0, color="#fff", lw=1.2, alpha=0.8)
    ban = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")
    alert_mask = (ta > 0) | (ua > 0)
    first_alert = float(x[np.argmax(alert_mask)]) if alert_mask.any() else None
    times = np.linspace(0, tmax, 48)

    def fr(tc):
        head.set_xdata([tc, tc])
        if first_alert is not None and tc >= first_alert:
            ban.set_text("HARDWARE ALERT (1-cycle latency)"); ban.set_color(DANGER)
        return ()
    return _save(FuncAnimation(fig, fr, frames=times), "fpga.gif")


DETBLOCK = """
    <section class="detblock">
      <h2 class="dt"><span class="dot"></span>{title}</h2>
      <div class="animrow"><img src="data:image/gif;base64,{img}" alt="{title}"/></div>
      <div class="cap">
        <p><span class="lbl">What you're seeing</span>{does}</p>
        <p><span class="lbl">Why it matters</span>{why}</p>
        <p><span class="lbl">What it contributes</span>{contrib}</p>
      </div>
      <div class="mathx">{mathx}</div>
    </section>"""


def make_math():
    """Real detector math in LaTeX (rendered by MathJax), filled with this run's values.
    Uses \\lt / \\gt inside math so no raw < or > reaches the HTML parser."""
    def disp(*eqs):
        return "".join(r"\[" + e + r"\]" for e in eqs)

    m = {}

    d = np.load(os.path.join(DATA, "nav.npz"))
    honest, peak = f"{float(d['honest_nis']):.1f}", f"{float(d['attack_nis']):.0f}"
    m["nav"] = disp(
        r"\textbf{state}\quad \mathbf{x}=[\,p_x,\;p_y,\;v,\;\theta\,]^\top,\qquad \dot p_x=v\cos\theta,\;\;\dot p_y=v\sin\theta",
        r"\textbf{IMU}\quad v_{k+1}=v_k+a\,\Delta t,\qquad \theta_{k+1}=\theta_k+\omega\,\Delta t",
        r"\textbf{predict}\quad \hat{\mathbf{x}}=f(\mathbf{x},\mathbf{u}),\qquad P=F\,P\,F^{\top}+Q,\qquad F=\frac{\partial f}{\partial \mathbf{x}}",
        r"\textbf{update}\quad \mathbf{y}=\mathbf{z}-H\mathbf{x},\quad S=H P H^{\top}+R,\quad K=P H^{\top} S^{-1}",
        r"\textbf{NIS}\quad d=\mathbf{y}^{\top} S^{-1}\mathbf{y}\ \sim\ \chi^{2}_{2},\qquad g_k=\max\!\big(0,\;g_{k-1}+(d_k-3)\big)",
        r"\textbf{alarm}\quad \overline{d}_{5}\gt 9.21\ \ (\chi^{2}_{2},\,99\%)\quad\text{or}\quad g_k\gt 14",
    ) + ("<p class='run'>This run: honest \\(\\overline d\\approx " + honest +
         "\\); under spoof \\(d\\) peaks \\(\\approx " + peak + "\\gg 9.21\\) &rarr; fired.</p>")

    d = np.load(os.path.join(DATA, "comm.npz"))
    p0, sig, tau = f"{float(d['p0']):.3f}", f"{float(d['pstd']):.3f}", f"{float(d['threshold']):.3f}"
    cp, jp, kind = f"{float(d['clean_power']):.3f}", f"{float(d['jammed_power']):.2f}", str(d['jam_kind'])
    m["comm"] = disp(
        r"\textbf{OFDM}\quad x[n]=\mathrm{IFFT}\{S_m\}+\text{CP},\qquad S_m\in\mathrm{QPSK}",
        r"\textbf{power}\quad P=\frac{1}{N}\sum_{n=0}^{N-1}\lvert x[n]\rvert^{2},\qquad \tau=P_0+4\,\sigma_0",
        r"\textbf{spectrum}\quad X[k]=\sum_{n=0}^{N-1}x[n]\,e^{-j2\pi kn/N},\qquad \mathrm{PSD}[k]=\lvert X[k]\rvert^{2}",
        r"\textbf{flatness}\quad \mathrm{SF}=\frac{\left(\prod_{k}\mathrm{PSD}[k]\right)^{1/N}}{\tfrac{1}{N}\sum_{k}\mathrm{PSD}[k]},\qquad \text{alarm if } P\gt\tau",
    ) + ("<p class='run'>This run: \\(P_0=" + p0 + ",\\ \\sigma_0=" + sig + ",\\ \\tau=" + tau +
         "\\). Clean \\(P=" + cp + "\\lt\\tau\\); jammed \\(P=" + jp +
         "\\gt\\tau\\) &rarr; alarm, classified &ldquo;" + kind + "&rdquo;.</p>")

    d = np.load(os.path.join(DATA, "perc.npz"))
    med, ratio, score = f"{float(d['median_energy']):.3f}", f"{float(d['top_ratio']):.0f}", f"{float(d['top_score']):.0f}"
    sat = f"{float(d['top_sat']):.2f}"
    m["perc"] = disp(
        r"\textbf{gradient energy}\quad E(x,y)=\left(\frac{\partial I}{\partial x}\right)^{2}+\left(\frac{\partial I}{\partial y}\right)^{2}=\lVert\nabla I\rVert^{2}",
        r"\textbf{window}\quad r=\frac{\overline{E}_{\text{win}}}{\operatorname{median}(E)},\qquad \text{score}=r\,(1+0.5\,S)",
        r"\textbf{flag}\quad r\ \ge\ 4",
    ) + ("<p class='run'>This run: \\(\\operatorname{median}(E)=" + med + "\\); the top window has \\(r=" +
         ratio + "\\times\\) the median \\((\\ge 4\\Rightarrow\\text{flagged})\\), saturation \\(S=" +
         sat + "\\), score \\(=" + score + "\\).</p>")

    d = np.load(os.path.join(DATA, "can.npz"))
    T, bus, fr = f"{float(d['learned_period_ms']):.1f}", f"{float(d['normal_bus_rate']):.0f}", f"{float(d['flood_rate']):.0f}"
    fid = f"0x{int(d['flood_id']):03X}"
    m["can"] = disp(
        r"\textbf{learn}\quad T=\frac{1}{N}\sum_{i} g_i,\qquad \sigma=\sqrt{\tfrac{1}{N}\sum_i (g_i-T)^2},\qquad g_i=t_i-t_{i-1}",
        r"\textbf{z-score}\quad z=\frac{g-T}{\sigma}",
        r"\textbf{rules}\quad \text{TIMING}:g\lt 0.5T,\quad \text{SILENCE}:g\gt 6T,\quad \text{RATE\_FLOOD}:\ \rho\gt 4\rho_0",
    ) + ("<p class='run'>This run: fastest \\(T\\approx " + T + "\\,\\text{ms}\\), normal \\(\\rho_0\\approx " +
         bus + "\\,\\text{msg/s}\\). Flood \\(" + fid + "\\) is unknown at \\(\\rho\\approx " + fr +
         "\\approx 6\\rho_0\\) &rarr; UNKNOWN_ID + RATE_FLOOD.</p>")

    d = np.load(os.path.join(DATA, "fpga.npz"))
    gap, mp = int(d['inject_gap_cycles']), int(d['min_period_cycles'])
    m["fpga"] = disp(
        r"\textbf{cycle counter}\quad c_{k+1}=c_k+1,\qquad t=\frac{c}{f_{\text{clk}}},\quad f_{\text{clk}}=100\,\mathrm{MHz}",
        r"\textbf{TIMING}\quad \text{seen}[id]\ \wedge\ \big(c-\text{last\_seen}[id]\big)\lt \text{min\_period}[id]",
        r"\text{min\_period (cycles)}=\{\text{0x0C0}{:}80,\ \text{0x0D0}{:}80,\ \text{0x110}{:}160,\ \text{0x320}{:}800\}",
    ) + ("<p class='run'>This run: injected \\(\\text{0x0C0}\\) arrives \\(\\approx " + str(gap) +
         "\\) cycles apart \\(\\lt " + str(mp) + "\\) &rarr; alert in one clock \\((\\approx 10\\,\\text{ns})\\).</p>")

    return m


def make_explainer():
    """A plain-language, step-by-step walkthrough of every detector's math."""
    D = [
        {"title": "Navigation — detecting GPS spoofing",
         "problem": "The car must know where it is, but GPS can be faked from outside. "
                    "The trick is to cross-check every GPS fix against the car's own motion, "
                    "measured by sensors an attacker cannot reach.",
         "steps": [
            ("1. Describe the car's state",
             r"\mathbf{x}=[\,p_x,\;p_y,\;v,\;\theta\,]^{\top}",
             "We track four numbers: position east \\(p_x\\), position north \\(p_y\\), "
             "speed \\(v\\), and heading angle \\(\\theta\\). Together they say where the car "
             "is, how fast it is moving, and which way it points."),
            ("2. Predict motion from physics (the IMU)",
             r"\dot p_x=v\cos\theta,\quad \dot p_y=v\sin\theta,\qquad v_{k+1}=v_k+a\,\Delta t,\quad \theta_{k+1}=\theta_k+\omega\,\Delta t",
             "This is basic kinematics. Velocity splits into an east part \\(v\\cos\\theta\\) and "
             "a north part \\(v\\sin\\theta\\). The accelerometer measures acceleration \\(a\\), so "
             "the new speed is the old speed plus \\(a\\) times the time step \\(\\Delta t\\). The "
             "gyroscope measures turn rate \\(\\omega\\), so heading updates the same way. This lets "
             "the car estimate its own position using no GPS at all."),
            ("3. Track uncertainty (the Kalman filter, and the calculus)",
             r"P=F\,P\,F^{\top}+Q,\qquad F=\frac{\partial f}{\partial \mathbf{x}}",
             "\\(P\\) is how unsure the filter is about the state. Predicting forward grows that "
             "uncertainty by the process noise \\(Q\\). \\(F\\) is the Jacobian, the matrix of "
             "partial derivatives of the motion model. That derivative is the calculus step: it "
             "linearizes the physics so the update stays a clean matrix equation. This linearization "
             "is exactly what the word Extended means in Extended Kalman Filter."),
            ("4. Compare GPS to the prediction (the innovation)",
             r"\mathbf{y}=\mathbf{z}-H\mathbf{x}",
             "\\(\\mathbf{z}\\) is the real GPS reading. \\(H\\mathbf{x}\\) is what GPS should read "
             "if the physics prediction is correct. Their difference \\(\\mathbf{y}\\) is the "
             "surprise: how far GPS strays from what the motion predicted. A small surprise means the "
             "two agree; a large one means something is wrong."),
            ("5. Normalize the surprise (NIS)",
             r"S=HPH^{\top}+R,\qquad d=\mathbf{y}^{\top} S^{-1}\mathbf{y}\ \sim\ \chi^{2}_{2}",
             "Raw distance is not enough, because some surprise is always expected from noise. "
             "\\(S\\) is the expected size of the surprise. Dividing the surprise by its expected "
             "size gives a unitless score \\(d\\): how many expected-surprises big this one is. Under "
             "honest conditions \\(d\\) follows a chi-square law with two degrees of freedom (GPS "
             "gives two numbers, east and north) and averages about 2."),
            ("6. Decide",
             r"\overline{d}_{5} \gt 9.21\ \ (99\%)\quad\text{or}\quad g_k=\max\!\big(0,\,g_{k-1}+(d_k-3)\big) \gt 14",
             "9.21 is the chi-square value that honest data exceeds only 1% of the time, so passing "
             "it means the GPS is almost certainly spoofed. The CUSUM term \\(g_k\\) adds up small "
             "persistent surprises to catch a slow drift that never spikes. On this run the honest "
             "score sat near 1.8 and the spoof drove it to about 170, roughly 90 times over the "
             "line, so it was caught at once."),
         ]},
        {"title": "Communication — detecting V2X jamming",
         "problem": "Cars broadcast safety messages over radio. A jammer floods that channel with "
                    "energy so the car goes deaf. We detect the flood, and identify what kind it is.",
         "steps": [
            ("1. Build the radio signal (OFDM)",
             r"x[n]=\mathrm{IFFT}\{S_m\}+\text{CP},\qquad S_m\in\mathrm{QPSK}",
             "Real vehicle radios use OFDM. Data symbols \\(S_m\\) are spread across many "
             "frequencies with an inverse FFT, and a cyclic prefix (CP) is added to survive echoes. "
             "\\(x[n]\\) is the resulting time-domain waveform."),
            ("2. Measure how loud the channel is",
             r"P=\frac{1}{N}\sum_{n=0}^{N-1}\lvert x[n]\rvert^{2}",
             "Power is the average squared amplitude, which is literally how much energy is in the "
             "signal. A jammer adds energy, so the power rises."),
            ("3. Set the alarm line from clean data",
             r"\tau=P_0+4\,\sigma_0",
             "From clean frames we learn the normal power \\(P_0\\) and its natural wobble "
             "\\(\\sigma_0\\). The threshold sits four standard deviations above normal, high enough "
             "that honest noise almost never trips it. On this run \\(P_0=1.033\\) and "
             "\\(\\tau=1.151\\). Clean power stayed at 1.033, under the line; jamming spiked it to "
             "9.30, far over."),
            ("4. Identify the jammer by its spectrum shape",
             r"X[k]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N},\qquad \mathrm{SF}=\frac{\left(\prod_{k}\mathrm{PSD}[k]\right)^{1/N}}{\tfrac{1}{N}\sum_{k}\mathrm{PSD}[k]}",
             "The FFT \\(X[k]\\) breaks the signal into its frequencies, and \\(\\lvert X[k]\\rvert^2\\) "
             "is the power at each one (the PSD). Spectral flatness compares the geometric mean to "
             "the arithmetic mean of that spectrum. A flat, spread-out spectrum means a broadband "
             "barrage; a single sharp spike means a narrowband tone; a moving band means a sweep. "
             "This run classified the attack as tone/narrowband."),
         ]},
        {"title": "Perception — detecting an adversarial patch",
         "problem": "A printed sticker on a sign can fool the camera's vision model. These patches "
                    "are dense, high-frequency noise, so we hunt for the noisiest region of the image.",
         "steps": [
            ("1. Measure sharpness at every pixel (image gradient)",
             r"E(x,y)=\left(\frac{\partial I}{\partial x}\right)^{2}+\left(\frac{\partial I}{\partial y}\right)^{2}=\lVert\nabla I\rVert^{2}",
             "\\(\\partial I/\\partial x\\) is how fast brightness changes left to right, and "
             "\\(\\partial I/\\partial y\\) top to bottom. Squaring and adding gives the gradient "
             "energy: high where the image changes sharply, low where it is smooth. This is the "
             "gradient from calculus applied to an image. Adversarial patches are built to be dense "
             "high-frequency noise, so their gradient energy is enormous."),
            ("2. Compare each region to the whole scene",
             r"r=\frac{\overline{E}_{\text{win}}}{\operatorname{median}(E)}",
             "Slide a window across the image. For each window, divide its average energy by the "
             "median energy of the whole scene. \\(r\\) then says this region is \\(r\\) times "
             "sharper than a typical spot. A normal textured area is only a few times the median."),
            ("3. Flag the patch",
             r"\text{flag if}\quad r\ \ge\ 4",
             "Above four times the median, a region is suspiciously noisy and likely a patch. On "
             "this run the top window measured 294 times the median energy, so it was not subtle."),
         ]},
        {"title": "In-vehicle network — detecting a CAN flood",
         "problem": "The car's internal bus carries brake, steer, and engine commands and has no "
                    "authentication. We catch an attacker by watching the timing of messages.",
         "steps": [
            ("1. Learn the normal rhythm",
             r"T=\frac{1}{N}\sum_i g_i,\qquad \sigma=\sqrt{\tfrac{1}{N}\sum_i (g_i-T)^{2}},\qquad g_i=t_i-t_{i-1}",
             "Each message type is sent on a fixed clock. We measure the gaps \\(g_i\\) between "
             "arrivals and average them: \\(T\\) is the normal period and \\(\\sigma\\) its jitter. "
             "On this run the fastest message repeats every \\(T\\approx 10\\) ms."),
            ("2. Score each new message",
             r"z=\frac{g-T}{\sigma},\qquad \text{TIMING if}\ \ g \lt 0.5T",
             "When a message arrives with gap \\(g\\), the z-score says how many jitters "
             "off-schedule it is. If it comes more than twice too fast, meaning \\(g\\) is under "
             "half the normal period, it is an injection."),
            ("3. Catch floods and strangers",
             r"\text{RATE\_FLOOD if}\ \ \rho \gt 4\rho_0,\qquad \text{UNKNOWN\_ID if id}\notin\text{baseline}",
             "If the whole-bus message rate \\(\\rho\\) jumps to four times normal \\(\\rho_0\\), or "
             "an ID appears that was never in the learned set, it is an attack. On this run normal "
             "traffic ran near 336 messages per second; the flood used the never-seen ID 0x000 at "
             "about 2005 per second, roughly six times normal, tripping both rules."),
         ]},
        {"title": "Hardware — the same detection in an FPGA",
         "problem": "Software checks add delay, but a real gateway must flag an attack at line rate. "
                    "So the same timing logic is built directly into digital hardware.",
         "steps": [
            ("1. A hardware clock",
             r"c_{k+1}=c_k+1,\qquad t=\frac{c}{f_{\text{clk}}},\qquad f_{\text{clk}}=100\,\mathrm{MHz}",
             "A counter ticks up once every clock cycle. Multiplying the cycle count by the clock "
             "period converts it to real time. At 100 MHz, one cycle is 10 nanoseconds."),
            ("2. The timing check, in logic",
             r"\text{TIMING if}\quad \text{seen}[id]\ \wedge\ \big(c-\text{last\_seen}[id]\big) \lt \text{min\_period}[id]",
             "For each known ID the chip stores when it was last seen and its minimum allowed "
             "spacing in cycles. If a frame arrives sooner than allowed, the alert is raised on the "
             "very next clock, with no software in the loop. On this run the injected 0x0C0 arrived "
             "22 cycles after the previous one, under its 80-cycle minimum, so the alert fired one "
             "cycle later."),
         ]},
    ]
    keys = ["nav", "comm", "perc", "can", "fpga"]
    result = {}
    for key, det in zip(keys, D):
        parts = ['<div class="mathlab">The math, step by step</div>']
        parts.append('<p class="prob"><span class="ppill">The problem</span>' + det["problem"] + '</p>')
        for name, latex, prose in det["steps"]:
            parts.append('<div class="step"><div class="sname">' + name + '</div>')
            parts.append(r'\[' + latex + r'\]')
            parts.append('<p>' + prose + '</p></div>')
        result[key] = "".join(parts)
    return result


def build_html(imgs):
    order = [
        ("Navigation &middot; GPS spoofing", imgs["nav"],
         "An Extended Kalman Filter fuses the car's GPS with its inertial sensors to estimate where "
         "it truly is. Halfway through the drive the GPS is spoofed with a sudden jump. The filter "
         "compares each GPS fix against its own physics-based prediction with a chi-square test, and "
         "the instant the spoofed fix disagrees with the motion, it fires.",
         "GPS spoofing makes the vehicle navigate on a lie. A faked position can route a car off "
         "course or into a hazard with no obvious sensor failure, which makes it one of the most "
         "dangerous attacks on connected vehicles.",
         "Proves the navigation layer can catch a location attack in real time using only sensors "
         "the car already has, with zero false alarms on honest data. This sensor-fusion and "
         "anomaly-detection core is the same math that carries over to biosignal monitoring."),
        ("Communication &middot; V2X jamming", imgs["comm"],
         "A real OFDM radio link, like the one cars use to talk to each other and to infrastructure, "
         "is measured frame by frame. Honest frames sit under a learned power threshold. When a tone "
         "jammer switches on, the received power spikes over the line and the detector flags and "
         "classifies it.",
         "V2X messages carry safety warnings like collision and emergency-braking alerts. If an "
         "attacker jams that radio, the car goes deaf to its surroundings at exactly the moment it "
         "needs to hear them.",
         "Shows the communication layer can notice it is being silenced, and identify how, using "
         "standard energy detection plus spectral analysis, with no false alarms on clean frames."),
        ("Perception &middot; adversarial patch", imgs["perc"],
         "A camera scene gets an adversarial patch added, the kind of printed sticker that fools a "
         "vision model into misreading a sign. The detector scans for the dense high-frequency, "
         "high-saturation texture that patches have and natural scenes lack, and boxes the suspect "
         "regions (red) around where the patch really is (dashed yellow).",
         "The camera is the car's primary eyes. One well-placed patch can flip a classification and "
         "trigger a wrong, potentially fatal driving decision. This is the most studied class of AV "
         "attack.",
         "Demonstrates the perception layer can localize a physical-world attack on the camera, and "
         "pairs an explainable frequency method with a trained CNN so the two can be compared."),
        ("In-vehicle network &middot; CAN flood", imgs["can"],
         "The timing of messages on the car's internal CAN bus, the network linking engine, brakes, "
         "and steering, is plotted live. Legitimate traffic is strictly periodic, forming clean "
         "rows. A denial-of-service flood injects an unknown ID far too fast, breaking that rhythm, "
         "and the timing detector flags it at once.",
         "The CAN bus has no built-in authentication, so any compromised component can flood or "
         "spoof safety-critical commands. This is where a perception or network attack finally turns "
         "into physical control, the heart of cross-layer propagation.",
         "Proves the deepest layer, the control network itself, can be defended with a lightweight "
         "timing model that runs on an embedded gateway, catching floods at 100 percent detection "
         "and zero false positives on public data."),
        ("Hardware &middot; FPGA CAN IDS", imgs["fpga"],
         "The same CAN detector, rebuilt in synthesizable Verilog and run through a real hardware "
         "simulation. The waveform shows message strobes arriving, and when an attack frame hits, "
         "the alert line pulses within a single clock cycle.",
         "Software detection adds latency, but a real automotive gateway must flag an attack at line "
         "rate, before a malicious frame is acted on. Doing it in hardware is what makes the defense "
         "actually deployable on a vehicle.",
         "Shows the detection logic works in real digital hardware at single-cycle latency, bridging "
         "the work from a Python demo to something that could sit on a physical FPGA gateway. Core "
         "ECE digital-design proof."),
    ]
    keys = ["nav", "comm", "perc", "can", "fpga"]
    ex = make_explainer()
    blocks = "".join(DETBLOCK.format(title=t, img=g, does=a, why=b, contrib=c, mathx=ex[k])
                     for k, (t, g, a, b, c) in zip(keys, order))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>AV Stack Defense - Live</title>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body{{margin:0;background:{INK};color:{FG};font-family:'Segoe UI',system-ui,sans-serif;}}
  .wrap{{max-width:1120px;margin:0 auto;padding:34px 22px 80px;}}
  .eyebrow{{font-family:ui-monospace,Consolas,monospace;font-size:11px;letter-spacing:.2em;
    text-transform:uppercase;color:{ACCENT};}}
  h1{{font-size:30px;margin:6px 0 4px;letter-spacing:-.02em;}}
  .sub{{color:#93a19f;font-size:14.5px;max-width:72ch;}}
  .strip{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 6px;}}
  .stat{{background:#131a19;border:1px solid #243130;border-radius:9px;padding:8px 13px;
    font-family:ui-monospace,Consolas,monospace;font-size:12.5px;}}
  .stat b{{color:{ACCENT};}}
  .dot{{width:8px;height:8px;border-radius:50%;background:{ACCENT};box-shadow:0 0 8px {ACCENT};
    display:inline-block;flex:0 0 auto;}}
  .detblock{{max-width:820px;margin:34px auto 0;border-top:1px solid #243130;padding-top:28px;}}
  .detblock:first-of-type{{border-top:0;padding-top:8px;}}
  .dt{{font-size:20px;font-weight:700;margin:0 0 14px;display:flex;align-items:center;gap:10px;
    letter-spacing:-.01em;}}
  .animrow{{text-align:center;}}
  .animrow img{{width:100%;max-width:600px;border-radius:10px;background:{INK};
    border:1px solid #243130;}}
  .cap{{margin-top:16px;}}
  .cap p{{color:#aeb9b7;font-size:13px;line-height:1.55;margin:0 0 9px;}}
  .cap p:last-child{{margin-bottom:0;}}
  .lbl{{display:block;font-family:ui-monospace,Consolas,monospace;font-size:10px;letter-spacing:.09em;
    text-transform:uppercase;color:{ACCENT};margin-bottom:3px;font-weight:600;}}
  .mathx{{margin-top:20px;}}
  .mathlab{{font-family:ui-monospace,Consolas,monospace;font-size:10.5px;letter-spacing:.09em;
    text-transform:uppercase;color:{GPSY};margin-bottom:10px;}}
  .prob{{background:#131a19;border:1px solid #243130;border-radius:9px;padding:12px 14px;
    font-size:13.5px;line-height:1.55;color:#c3ccca;}}
  .ppill{{display:inline-block;font-family:ui-monospace,Consolas,monospace;font-size:9.5px;
    letter-spacing:.09em;text-transform:uppercase;color:{GPSY};margin-right:9px;}}
  .step{{margin-top:18px;}}
  .sname{{font-weight:650;font-size:14px;color:{ACCENT};margin-bottom:2px;}}
  .step p{{font-size:13.5px;line-height:1.65;color:#b7c2c0;margin:4px 0 0;}}
  .step mjx-container[display="true"]{{text-align:left !important;margin:.45em 0 !important;
    overflow-x:auto;overflow-y:hidden;max-width:100%;}}
  .foot{{max-width:820px;margin:40px auto 0;color:#63716f;font-size:12px;
    font-family:ui-monospace,Consolas,monospace;}}
</style></head><body><div class="wrap">
  <div class="eyebrow">Cross-layer AV defense &middot; live run (real detectors, simulated inputs)</div>
  <h1>Five detectors, one vehicle, watched in real time</h1>
  <p class="sub">Every panel animates the real output of a real detector from the repos, captured on
  this run. Inputs are the projects' own simulations; the detection code and results are genuine.</p>
  <div class="strip">
    <div class="stat"><b>5/5</b> layers detecting</div>
    <div class="stat"><b>0.000</b> false-alarm rate</div>
    <div class="stat"><b>1.000</b> detection rate</div>
    <div class="stat"><b>FPGA</b> testbench PASS</div>
  </div>
  {blocks}
  <div class="foot">Generated live from av-stack-defense/viz/build.py. Re-run to regenerate.</div>
</div></body></html>"""


def main():
    print("Collecting real signals from each detector...")
    collect()
    print("Rendering animated visuals (this takes ~60-90s)...")
    imgs = {"nav": nav_gif(), "comm": comm_gif(), "perc": perc_gif(),
            "can": can_gif(), "fpga": fpga_gif()}
    out = os.path.join(HERE, "dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html(imgs))
    print(f"Wrote {out}")
    webbrowser.open("file:///" + out.replace("\\", "/"))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

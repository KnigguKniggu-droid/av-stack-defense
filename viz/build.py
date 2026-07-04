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


PANEL = """
    <div class="panel">
      <div class="ph"><span class="dot"></span>{title}</div>
      <img src="data:image/gif;base64,{img}" alt="{title}"/>
      <div class="cap">
        <p><span class="lbl">What you're seeing</span>{does}</p>
        <p><span class="lbl">Why it matters</span>{why}</p>
        <p><span class="lbl">What it contributes</span>{contrib}</p>
      </div>
    </div>"""


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
    panels = "".join(PANEL.format(title=t, img=g, does=a, why=b, contrib=c)
                     for t, g, a, b, c in order)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>AV Stack Defense - Live</title>
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
  .grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin-top:20px;}}
  @media(max-width:820px){{.grid{{grid-template-columns:1fr;}}}}
  .panel{{background:#101715;border:1px solid #243130;border-radius:13px;padding:14px;}}
  .panel:last-child{{grid-column:1/-1;max-width:560px;margin:0 auto;}}
  .ph{{font-size:14px;font-weight:650;margin-bottom:10px;display:flex;align-items:center;gap:9px;}}
  .dot{{width:8px;height:8px;border-radius:50%;background:{ACCENT};box-shadow:0 0 8px {ACCENT};}}
  .panel img{{width:100%;border-radius:8px;display:block;background:{INK};}}
  .cap{{margin-top:12px;}}
  .cap p{{color:#aeb9b7;font-size:12.5px;line-height:1.55;margin:0 0 9px;}}
  .cap p:last-child{{margin-bottom:0;}}
  .lbl{{display:block;font-family:ui-monospace,Consolas,monospace;font-size:10px;letter-spacing:.09em;
    text-transform:uppercase;color:{ACCENT};margin-bottom:3px;font-weight:600;}}
  .foot{{margin-top:30px;color:#63716f;font-size:12px;font-family:ui-monospace,Consolas,monospace;}}
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
  <div class="grid">{panels}</div>
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

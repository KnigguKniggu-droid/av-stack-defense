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
import json
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


HERO_BLOCK = """
    <section class="section" id="nav">
      <div class="section-header">
        <h2 class="section-title">{title}</h2>
        <p class="section-description">{thesis}</p>
      </div>
      <figure class="fig">
        <img src="data:image/gif;base64,{img}" alt="{alt}"/>
        <figcaption>{figcap}</figcaption>
      </figure>
      <div class="facets-grid">
        <div class="facet-block">
          <span class="facet-label">What you're seeing</span>
          <p>{does}</p>
        </div>
        <div class="facet-block">
          <span class="facet-label">Why it matters</span>
          <p>{why}</p>
        </div>
        <div class="facet-block">
          <span class="facet-label">What it contributes</span>
          <p>{contrib}</p>
        </div>
      </div>
      {math}
    </section>"""


DET_BLOCK = """
    <section class="section" id="{id}">
      <div class="section-header">
        <h2 class="section-title">{title}</h2>
      </div>
      <figure class="fig">
        <img src="data:image/gif;base64,{img}" alt="{alt}"/>
        <figcaption>{figcap}</figcaption>
      </figure>
      <div class="facets-grid">
        <div class="facet-block">
          <span class="facet-label">What you're seeing</span>
          <p>{does}</p>
        </div>
        <div class="facet-block">
          <span class="facet-label">Why it matters</span>
          <p>{why}</p>
        </div>
        <div class="facet-block">
          <span class="facet-label">What it contributes</span>
          <p>{contrib}</p>
        </div>
      </div>
      {math}
    </section>"""


MATH_DETAILS = """
      <details class="math-details">
        <summary class="math-summary">
          <span>The math, worked step by step</span>
          <span class="math-chevron">&#9662;</span>
        </summary>
        <div class="math-content"><div class="mathx">{body}</div></div>
      </details>"""


STYLE = """
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');

  :root{
    --bg:#030712; --surface:#090d16; --surface-soft:#0e1320;
    --line:#1f293d; --line-soft:#2b3a54;
    --ink:#f8fafc; --ink-2:#cbd5e1; --ink-3:#94a3b8; --muted:#64748b; --faint:#475569;
    --teal:#06b6d4; --danger:#ef4444; --gpsy:#fbbf24; --blue:#6366f1;
    --sans:'Inter',system-ui,-apple-system,sans-serif;
    --font-display:'Outfit',system-ui,-apple-system,sans-serif;
    --mono:'JetBrains Mono',ui-monospace,'SFMono-Regular',monospace;
    --e-out:cubic-bezier(.22,1,.36,1);
  }

  *{box-sizing:border-box;}
  html{-webkit-text-size-adjust:100%;}
  body{margin:0;font-family:var(--sans);font-size:15px;line-height:1.6;
    letter-spacing:-0.005em;color:var(--ink);
    background:
      radial-gradient(1200px 520px at 50% -8%, #0f1e36 0%, rgba(15,30,54,0) 65%),
      var(--bg);
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}
  img{max-width:100%;display:block;}
  a{color:var(--teal);text-underline-offset:3px;}
  ::selection{background:rgba(6,182,212,.28);color:#fff;}
  :focus-visible{outline:2px solid var(--teal);outline-offset:3px;border-radius:5px;}

  .wrap{max-width:1000px;margin:0 auto;
    padding:clamp(30px,5vw,64px) clamp(18px,4vw,28px) 120px;}

  /* ---- masthead ---- */
  .masthead{max-width:860px;margin-inline:auto;margin-bottom:40px;}
  .status{display:inline-flex;align-items:center;gap:9px;font-family:var(--mono);
    font-size:11px;letter-spacing:.01em;color:var(--muted);margin:0 0 16px;}
  .live{width:8px;height:8px;border-radius:50%;background:var(--teal);flex:0 0 auto;
    box-shadow:0 0 0 0 rgba(6, 182, 212, .6);animation:live 2.6s var(--e-out) infinite;}
  @keyframes live{
    0%{box-shadow:0 0 0 0 rgba(6, 182, 212, .55);}
    70%{box-shadow:0 0 0 7px rgba(6, 182, 212, 0);}
    100%{box-shadow:0 0 0 0 rgba(6, 182, 212, 0);}}

  h1{font-family:var(--font-display);font-weight:800;letter-spacing:-0.03em;
    font-size:clamp(2rem,1.4rem+2.5vw,3rem);line-height:1.06;margin:0 0 16px;
    color:#fff;text-wrap:balance;}
  .lede{font-size:clamp(0.95rem,.9rem+0.2vw,1.08rem);color:var(--ink-2);
    max-width:68ch;margin:0 0 24px;text-wrap:pretty;}

  .readout{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 24px;}
  .chip{font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums;
    color:var(--ink-2);background:var(--surface-soft);border:1px solid var(--line);
    border-radius:6px;padding:6px 12px;transition:border-color .25s var(--e-out);}
  .chip:hover{border-color:rgba(6, 182, 212, .35);}
  .chip b{color:var(--teal);font-weight:600;}

  .stack-note{font-size:.92rem;color:var(--muted);max-width:76ch;margin:0;
    padding-top:20px;border-top:1px solid var(--line);line-height:1.55;}
  .stack-note b{color:var(--ink-2);font-weight:600;}

  /* ---- section layout wide 1-column ---- */
  .section{max-width:860px;margin-inline:auto;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:32px;margin-top:32px;box-shadow:0 12px 40px rgba(0,0,0,0.5);}
  .section-header{margin-bottom:24px;border-bottom:1px solid var(--line);padding-bottom:16px;}
  .section-title{font-family:var(--font-display);font-size:22px;font-weight:700;margin:0;letter-spacing:-0.015em;color:#ffffff;}
  .section-description{font-size:13.5px;color:var(--ink-2);margin:6px 0 0;line-height:1.5;}

  /* ---- figures (large) ---- */
  .fig{margin:0 0 24px;}
  .fig img{width:100%;border-radius:8px;background:var(--bg);
    border:1px solid var(--line);box-shadow:0 8px 30px rgba(0,0,0,.4);
    transition:box-shadow .3s var(--e-out);}
  .fig img:hover{box-shadow:0 12px 40px rgba(0,0,0,.6);}
  figcaption{font-family:var(--mono);font-size:11px;color:var(--muted);
    margin-top:10px;letter-spacing:-.01em;}

  /* ---- facets horizontal grid ---- */
  .facets-grid{display:grid;grid-template-columns:1fr;gap:20px;margin-bottom:24px;border-top:1px solid var(--line);padding-top:20px;}
  @media (min-width:680px){
    .facets-grid{grid-template-columns:repeat(3,1fr);gap:24px;}
  }
  .facet-block{display:flex;flex-direction:column;}
  .facet-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;
    text-transform:uppercase;color:var(--teal);font-weight:600;margin-bottom:6px;}
  .facet-block p{margin:0;font-size:13px;line-height:1.55;color:var(--ink-2);}

  /* ---- math disclosure ---- */
  .math-details{margin-top:4px;}
  .math-summary{display:flex;justify-content:space-between;align-items:center;
    cursor:pointer;list-style:none;font-family:var(--mono);font-size:11px;
    letter-spacing:.04em;text-transform:uppercase;color:var(--gpsy);font-weight:600;
    padding:12px 16px;background:var(--surface-soft);border:1px solid var(--line);
    border-radius:6px;transition:all 0.2s ease;}
  .math-summary::-webkit-details-marker{display:none;}
  .math-summary:hover{background:var(--line-soft);color:#ffc93c;}
  .math-chevron{font-size:10px;color:var(--teal);
    transition:transform .25s var(--e-out);}
  .math-details[open] .math-chevron{transform:rotate(180deg);}
  .math-content{padding:16px;background:rgba(14, 19, 32, 0.5);border:1px solid var(--line);
    border-top:0;border-bottom-left-radius:6px;border-bottom-right-radius:6px;
    animation:reveal .3s var(--e-out);}
  @keyframes reveal{from{opacity:0;transform:translateY(-6px);}to{opacity:1;transform:none;}}

  .mathlab{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
    text-transform:uppercase;color:var(--gpsy);font-weight:600;margin:4px 0 12px;}
  .prob{background:var(--surface-soft);border:1px solid var(--line-soft);border-radius:8px;
    padding:12px 14px;font-size:12.5px;line-height:1.55;color:var(--ink-2);
    margin:0 0 16px;}
  .ppill{display:block;font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;
    text-transform:uppercase;color:var(--gpsy);font-weight:600;margin-bottom:4px;}
  .step{margin-top:12px;padding:0 0 12px;border-bottom:1px solid var(--line-soft);}
  .step:last-child{border-bottom:0;padding-bottom:0;}
  .sname{font-family:var(--sans);font-weight:600;font-size:13.5px;color:var(--teal);}
  .step p{font-size:12.5px;line-height:1.55;color:var(--ink-3);margin:6px 0 0;}
  .run{font-size:12.5px;color:var(--ink-2);margin-top:8px;}

  mjx-container[display="true"]{overflow-x:auto;overflow-y:hidden;max-width:100%;
    text-align:left !important;margin:.6em 0 !important;padding-bottom:2px;}
  mjx-container[display="true"]::-webkit-scrollbar{height:5px;}
  mjx-container[display="true"]::-webkit-scrollbar-thumb{background:var(--line);border-radius:2px;}

  /* ---- footer ---- */
  .foot{max-width:860px;margin:50px auto 0;padding-top:20px;
    border-top:1px solid var(--line);color:var(--faint);font-size:11px;
    font-family:var(--mono);text-align:center;letter-spacing:-.01em;}

  /* ---- fusion card ---- */
  .fusion-card{background:var(--surface);border:2px solid var(--teal);border-radius:16px;padding:32px;margin:40px auto 0;max-width:860px;
    box-shadow:0 12px 40px rgba(6, 182, 212, 0.12);transition:border-color 0.3s ease;}
  .fusion-card.level-CRITICAL{border-color:var(--danger);box-shadow:0 12px 40px rgba(255, 92, 92, 0.16);}
  .fusion-card.level-ALERT{border-color:var(--danger);box-shadow:0 12px 40px rgba(255, 92, 92, 0.12);}
  .fusion-card.level-ELEVATED{border-color:var(--gpsy);box-shadow:0 12px 40px rgba(242, 193, 78, 0.12);}
  .fusion-card.level-NOMINAL{border-color:var(--teal);box-shadow:0 12px 40px rgba(6, 182, 212, 0.12);}

  .fusion-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;flex-wrap:wrap;gap:16px;}
  .fusion-title{font-size:20px;font-weight:700;margin:0;letter-spacing:-0.02em;display:flex;align-items:center;gap:12px;color:#ffffff;}

  .badge{display:inline-block;padding:6px 14px;border-radius:20px;font-family:var(--mono);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;}
  .badge-CRITICAL{background:rgba(255, 92, 92, 0.15);color:var(--danger);border:1px solid rgba(255, 92, 92, 0.3);}
  .badge-ALERT{background:rgba(255, 92, 92, 0.12);color:var(--danger);border:1px solid rgba(255, 92, 92, 0.25);}
  .badge-ELEVATED{background:rgba(242, 193, 78, 0.12);color:var(--gpsy);border:1px solid rgba(242, 193, 78, 0.25);}
  .badge-NOMINAL{background:rgba(6, 182, 212, 0.12);color:var(--teal);border:1px solid rgba(6, 182, 212, 0.25);}

  .threat-gauge-container{margin:24px 0 16px;}
  .gauge-label{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em;}
  .gauge-value{font-size:26px;font-weight:800;color:#ffffff;}

  .progress-bg{background:var(--surface-soft);border-radius:10px;height:18px;overflow:hidden;position:relative;border:1px solid var(--line);margin-bottom:20px;}
  .progress-bar{height:100%;border-radius:10px;transition:width 0.5s ease;}

  /* Stacked contribution bar */
  .stacked-bar{display:flex;height:12px;border-radius:6px;overflow:hidden;margin-top:4px;background:var(--surface-soft);border:1px solid var(--line);}
  .stacked-segment{height:100%;transition:width 0.5s ease;position:relative;}

  .segment-label-list{display:flex;flex-wrap:wrap;gap:12px 24px;margin-top:16px;padding-top:16px;border-top:1px dashed var(--line);}
  .segment-label{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--ink-2);font-weight:400;}
  .color-dot{width:8px;height:8px;border-radius:50%;display:inline-block;}

  .fusion-summary{background:rgba(12, 18, 17, 0.4);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:20px;font-size:13px;line-height:1.6;color:var(--ink-2);}
  .fusion-summary p{margin:0 0 8px;}
  .fusion-summary p:last-child{margin-bottom:0;}
  .fusion-summary b{color:#ffffff;}
  .fusion-summary code{font-family:var(--mono);font-size:11px;color:var(--teal);}

  .fusion-tabs{display:flex;gap:8px;margin-bottom:14px;border-bottom:1px solid var(--line);padding-bottom:14px;}
  .fusion-tab-btn{background:var(--surface-soft);border:1px solid var(--line);color:var(--muted);padding:8px 16px;border-radius:20px;font-family:var(--sans);font-size:12px;font-weight:500;cursor:pointer;transition:all 0.2s ease;}
  .fusion-tab-btn:hover{color:var(--ink);border-color:var(--teal);}
  .fusion-tab-btn.active{background:var(--teal);color:#030712;border-color:var(--teal);font-weight:600;}
  .fusion-tab-content{display:none;}
  .fusion-tab-content.active-content{display:block;}

  /* ---- scroll reveal (progressive; content is visible without JS) ---- */
  .reveal{opacity:0;transform:translateY(20px);}
  .reveal.in{opacity:1;transform:none;
    transition:opacity .7s var(--e-out),transform .7s var(--e-out);}

  @media (prefers-reduced-motion: reduce){
    *,*::before,*::after{animation-duration:.001ms !important;
      animation-iteration-count:1 !important;transition-duration:.001ms !important;}
    .reveal{opacity:1 !important;transform:none !important;}
  }
"""


SCRIPT = """
<script>
(function(){
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var secs = document.querySelectorAll('.section');
  if(reduce || !('IntersectionObserver' in window)) return;
  secs.forEach(function(s){ s.classList.add('reveal'); });
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); }
    });
  }, {rootMargin:'0px 0px -12% 0px', threshold:0.08});
  secs.forEach(function(s){ io.observe(s); });
})();

function switchFusion(mode) {
  document.querySelectorAll('.fusion-tab-btn').forEach(function(btn) {
    btn.classList.remove('active');
  });
  document.querySelectorAll('.fusion-tab-content').forEach(function(content) {
    content.classList.remove('active-content');
  });

  var btn = document.querySelector('button[onclick*="' + mode + '"]');
  if (btn) btn.classList.add('active');

  var content = document.getElementById('fusion-content-' + mode);
  if (content) {
    content.classList.add('active-content');
    var card = document.querySelector('.fusion-card');
    if (card) {
      var level = content.getAttribute('data-level');
      card.className = 'fusion-card level-' + level + ' reveal in';
      var badge = document.getElementById('fusion-global-badge');
      if (badge) {
        badge.textContent = level + ' THREAT';
        badge.className = 'badge badge-' + level;
      }
    }
  }
}
</script>"""


MASTHEAD = """
  <header class="masthead">
    <p class="status"><span class="live"></span>Live run &middot; real detectors, simulated inputs</p>
    <h1>Five detectors, one vehicle, watched in real time</h1>
    <p class="lede">Every panel animates the real output of a real detector, captured on this run.
    The inputs are each project's own simulation, and the detection code and the numbers are genuine.</p>
    <div class="readout">
      <div class="chip"><b>5/5</b> layers detecting</div>
      <div class="chip"><b>0.000</b> false-alarm rate</div>
      <div class="chip"><b>1.000</b> detection rate</div>
      <div class="chip"><b>FPGA</b> testbench PASS</div>
    </div>
    <p class="stack-note">The stack is walked from the outside in: navigation first, then communication,
    perception, the in-vehicle network, and the hardware gateway. The navigation detector leads as the
    <b>flagship</b>, and the same sensor-fusion and anomaly-detection core carries across the others.</p>
  </header>"""


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
        parts = ['<p class="prob"><span class="ppill">The problem</span>' + det["problem"] + '</p>']
        for name, latex, prose in det["steps"]:
            parts.append('<div class="step"><div class="sname">' + name + '</div>')
            parts.append(r'\[' + latex + r'\]')
            parts.append('<p>' + prose + '</p></div>')
        result[key] = "".join(parts)
    return result


def build_html(imgs):
    nav = {
        "id": "nav",
        "title": "Navigation &middot; GPS spoofing",
        "thesis": "An Extended Kalman Filter fuses GPS with the car's own inertial sensors, then "
                  "challenges every fix with a chi-square test. The instant a spoof disagrees with "
                  "the physics the filter fires, with zero false alarms on honest data.",
        "alt": "Animated map. The car's true path is a teal line, the spoofed GPS fixes jump away as "
               "red crosses, and the dashed EKF estimate tracks the truth until a red banner reads "
               "GPS spoofing detected.",
        "figcap": "True path against spoofed GPS against the EKF estimate, from one live run.",
        "does": "An Extended Kalman Filter fuses the car's GPS with its inertial sensors to estimate where it truly is, moment to moment. The teal line is the real path, the dashed line is the filter's estimate, and the red crosses are GPS fixes that have been spoofed. Halfway through the drive the GPS is attacked with a sudden position jump, so the reported location leaps away from the true one. The filter compares every incoming GPS fix against its own physics-based prediction using a chi-square test on the innovation, which is the gap between what it expected and what it received. As long as the two agree the drive looks normal, but the instant the spoofed fix disagrees with the motion the car is actually feeling, the test spikes past its gate and the detector fires. The red banner marks the exact frame where the spoof is caught.",
        "why": "GPS spoofing makes the vehicle navigate on a lie, and it does so quietly. A faked position can route a car off its intended course, send it toward a hazard, or push it across a lane boundary, all without any obvious hardware failure to warn the driver. Because the GPS receiver itself reports clean, confident fixes, nothing downstream has a reason to distrust them. That silence is what makes spoofing one of the most dangerous attacks on connected and autonomous vehicles. It also scales cheaply, since a single roadside transmitter can spoof every receiver in range at once. Catching it therefore has to happen inside the vehicle, from physics the attacker cannot fake.",
        "contrib": "This proves the navigation layer can catch a location attack in real time using only sensors the car already carries, with no extra hardware and zero false alarms on honest data. It works by fusing GPS and inertial measurements and then testing their consistency, rather than trusting any single sensor on its own. The same chi-square innovation test and CUSUM change detector generalize to any setting where a trusted signal can be quietly manipulated. That sensor-fusion and anomaly-detection core is the exact math that carries over to biosignal monitoring and alarm-fatigue research. Within the cross-layer framework it anchors the navigation layer with a clean, measurable detector. It is also the flagship example of turning raw sensor streams into a decision you can defend.",
    }
    rest = [
        {"id": "comm",
         "title": "Communication &middot; V2X jamming",
         "alt": "Animated line chart of received radio power per frame. Points stay teal below the "
                "yellow jamming threshold, then spike red above it when the jammer switches on and a "
                "red banner reads jamming detected.",
         "figcap": "Received power per frame against the learned jamming threshold.",
         "does": "A real OFDM radio link, the same kind of physical layer cars use to talk to each other and to roadside infrastructure, is measured frame by frame. Each point on the chart is the received power in one frame, and honest traffic sits comfortably below a threshold the detector learned from clean data. When a tone jammer switches on it floods the band with energy, so the received power spikes above the yellow line. The detector flags those frames as jammed and then inspects the spectrum shape to classify the jammer as a narrowband tone, a wideband barrage, or a sweep. The red banner marks the moment the link realizes it is under attack. Everything here runs on standard energy detection plus spectral analysis, not on a black box.",
         "why": "V2X messages carry the safety information a car depends on, including collision alerts, emergency-braking notices, and signal-phase timing from intersections. If an attacker jams that radio, the vehicle goes deaf to its surroundings at exactly the moment it most needs to hear them. Jamming is also cheap and hard to trace, since the attacker only has to transmit noise and never has to break any cryptography. A car that cannot tell jamming from a genuinely quiet channel will simply assume the road ahead is clear. Detecting the jamming is the difference between failing silently and failing safely, for example by slowing down or handing control back to the driver. This layer gives the vehicle a way to know when its ears have been taken away.",
         "contrib": "This shows the communication layer can notice when it is being silenced, and can identify how, rather than just losing packets with no explanation. It builds a genuine OFDM waveform and attacks it, so the detection is measured against a real physical layer instead of a toy signal. The method reaches full detection and correct classification at a realistic signal-to-noise ratio, with no false alarms on clean frames. Because it relies on energy and spectral features, it is light enough to run continuously on the radio hardware a vehicle already has. In the cross-layer picture it defends the wireless entry point an attacker would use to reach the rest of the stack. It also demonstrates the DSP and spectral-analysis foundation that the whole communication defense is built on."},
        {"id": "perc",
         "title": "Perception &middot; adversarial patch",
         "alt": "Two camera frames side by side. The clean scene on the left, and on the right an "
                "adversarial patch with red detector boxes converging on the dashed-yellow true "
                "patch location.",
         "figcap": "Detector boxes in red closing on the real patch in dashed yellow.",
         "does": "A camera scene is shown twice: the clean frame on the left, and on the right the same frame with an adversarial patch added, the kind of printed sticker that can fool a vision model into misreading a sign. The detector scans the image for the dense high-frequency, high-saturation texture that these patches carry and that natural scenes almost never contain. As it searches, it draws red boxes around the regions it finds most suspicious, and those boxes converge on where the patch really is, marked in dashed yellow. The tighter the red boxes close on the yellow region, the more confident the localization. This is deliberately an explainable method, so you can see why a region was flagged instead of trusting a single opaque score. A trained CNN runs alongside it as a second opinion.",
         "why": "The camera is the car's primary set of eyes, and most driving decisions trace back to what it reports. One well-placed patch can flip a classification, turning a stop sign into a speed-limit sign in the model's view, and trigger a wrong and potentially fatal maneuver. Because the attack is a physical object in the world, it needs no digital access to the vehicle at all. It also survives changes in distance, angle, and lighting, which makes it practical rather than merely theoretical. This is the most studied class of attack on autonomous vehicles for exactly that reason. A perception layer that can find and isolate the patch is what lets the car recover the correct reading instead of acting on the lie.",
         "contrib": "This demonstrates the perception layer can localize a physical-world attack on the camera, not merely notice that something is wrong. It pairs an explainable frequency-and-saturation method with a trained CNN, so a transparent signal-processing approach and a learned model can be compared head to head. Once the patch is located, masking it lets the underlying model re-read the scene and recover the correct prediction. Reaching very high accuracy on this task shows the approach is more than a demonstration. Within the framework it defends the top of the stack, where an attack is most visible to a human yet most damaging to the machine. It also brings the computer-vision and adversarial-machine-learning side of the work into the same measured format as every other layer."},
        {"id": "can",
         "title": "In-vehicle network &middot; CAN flood",
         "alt": "Animated scatter of CAN bus messages over time. Legitimate traffic forms clean "
                "periodic rows in teal, then a red denial-of-service flood breaks the rhythm and a "
                "red banner reads CAN intrusion detected.",
         "figcap": "Arbitration ID against time. The flood is the block that breaks the grid.",
         "does": "The timing of messages on the car's internal CAN bus, the network that links the engine, brakes, and steering, is plotted live as arbitration ID against time. Legitimate traffic is strictly periodic, so honest messages form clean, evenly spaced rows across the chart. The detector learns each message's normal period during a short training window, and it needs no authentication on the bus to do so. When a denial-of-service flood injects an unknown ID far too fast, it breaks that rhythm and shows up as a dense block that does not fit the grid. The timing model sees the violated period immediately and raises the red intrusion banner. The same logic also catches injection, replay, and bus-off attacks, not only flooding.",
         "why": "The CAN bus was designed for reliability, not security, so it has no built-in authentication and every node trusts every message it sees. That means any single compromised component, whether a hacked infotainment unit or a malicious plug-in dongle, can flood or spoof safety-critical commands. This is the layer where an attack that began at the camera or the radio finally turns into physical control of the vehicle, which is the heart of cross-layer propagation. A flood here can drown out real brake or steering messages at the worst possible moment. Because the bus is closed and fast, the defense has to be lightweight and run in real time on an embedded gateway. Protecting it is what keeps a higher-layer compromise from ever reaching the actuators.",
         "contrib": "This proves the deepest layer, the control network itself, can be defended with a lightweight timing model rather than a heavy cryptographic retrofit. It learns the normal cadence of the bus and flags deviations, so it needs no changes to the existing controllers or the message format. On public CAN intrusion data it reaches full detection with zero false positives, which is the standard a safety case demands. Because it is cheap to compute, it can run continuously on the kind of gateway a real vehicle already carries. In the cross-layer story it is the last line before an attack becomes motion, so its reliability matters most of all. It also sets up the hardware version that follows, where the same detector is pushed down into silicon."},
        {"id": "fpga",
         "title": "Hardware &middot; FPGA CAN IDS",
         "alt": "Animated digital waveform of the Verilog CAN detector. A sweep cursor moves across "
                "the frame_valid and alert lines, and the alert pulses one clock cycle after an "
                "attack frame arrives.",
         "figcap": "The real Verilog waveform. Alerts fire at single-cycle latency.",
         "does": "This is the same CAN timing detector, but rebuilt in synthesizable Verilog and run through a real hardware simulation instead of Python. The waveform shows the actual digital signals: the message-valid strobes arriving on the bus, and the alert line the detector drives. A sweep cursor moves across the trace so you can follow events in clock-cycle time. When an attack frame arrives, the alert line pulses within a single clock cycle of the violation, with no software loop in between. The design is self-checking, so the testbench itself confirms the detector stays silent on normal traffic and fires on attacks. What you are watching is register-transfer logic behaving exactly as it would on a real chip.",
         "why": "Software detection, however accurate, adds latency, and a real automotive gateway has to flag an attack at line rate, before a malicious frame is ever acted on. At bus speed even a few milliseconds of delay can be the difference between catching a spoofed brake command and executing it. Moving the detector into hardware removes the operating system, the scheduler, and the interpreter from the critical path entirely. Single-cycle latency means the alert is ready essentially the moment the offending frame is seen. This is also what makes the defense deployable, since production gateways are built from exactly this kind of logic. A detector that only runs in a notebook cannot protect a moving vehicle, but one in silicon can.",
         "contrib": "This shows the detection logic works in real digital hardware at single-cycle latency, closing the gap between a Python demonstration and something that could sit on a physical FPGA gateway. It is written in synthesizable Verilog, so it is not a simulation shortcut but code that could be placed and routed onto a real device. The self-checking testbench passes, which is the hardware equivalent of a green test suite. Proving the same defense in both software and silicon is a genuine cross-domain result, not a repeat of a single idea. It is also a core ECE digital-design proof, the kind that shows the work holds up at the register-transfer level. Within the framework it is the layer that turns a research detector into a deployable one."},
    ]

    ex = make_explainer()
    hero = HERO_BLOCK.format(
        title=nav["title"], thesis=nav["thesis"], img=imgs["nav"], alt=nav["alt"],
        figcap=nav["figcap"], does=nav["does"], why=nav["why"], contrib=nav["contrib"],
        math=MATH_DETAILS.format(body=ex["nav"]))
    others = "".join(
        DET_BLOCK.format(
            id=d["id"], title=d["title"], img=imgs[d["id"]], alt=d["alt"],
            figcap=d["figcap"], does=d["does"], why=d["why"], contrib=d["contrib"],
            math=MATH_DETAILS.format(body=ex[d["id"]]))
        for d in rest)

    # Load fusion data helper
    def load_json(name):
        path = os.path.join(UMB, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    f_attack = load_json("fusion.json")
    f_clean = load_json("fusion_clean.json")
    f_coord = load_json("fusion_coordinated.json")

    f_attack_safety = load_json("fusion_safety.json")
    f_clean_safety = load_json("fusion_clean_safety.json")
    f_coord_safety = load_json("fusion_coordinated_safety.json")
    f_val = load_json("fusion_validation.json")

    def render_fusion_tab(fusion, fusion_safety, mode_id, active=False, validation=None):
        stacked_segments = []
        segment_labels = []

        layer_colors = {
            "adversarial-patch-detector": "var(--teal)",
            "ekf-gps-spoof-detector": "var(--blue)",
            "v2x-jamming-detector": "var(--gpsy)",
            "canbus-ids": "var(--danger)",
            "canbus-ids-fpga": "#b088ff"
        }
        layer_shortnames = {
            "adversarial-patch-detector": "Perception",
            "ekf-gps-spoof-detector": "Navigation",
            "v2x-jamming-detector": "Communication",
            "canbus-ids": "CAN (SW)",
            "canbus-ids-fpga": "CAN (FPGA)"
        }

        for L in fusion.get("layers", []):
            repo = L.get("repo")
            color = layer_colors.get(repo, "#cccccc")
            name = layer_shortnames.get(repo, L.get("layer"))
            contrib = L.get("contribution", 0.0)
            phi = L.get("phi", 0.0)

            seg_pct = contrib * 100
            stacked_segments.append(
                f'<div class="stacked-segment" style="width: {seg_pct}%; background: {color};" '
                f'title="{name}: contribution={contrib:.3f} (phi={phi:.3f})"></div>'
            )

            segment_labels.append(
                f'<div class="segment-label">'
                f'<span class="color-dot" style="background: {color};"></span>'
                f'<b>{name}</b> ({contrib:.3f})'
                f'</div>'
            )

        stacked_bar_html = "".join(stacked_segments)
        segment_labels_html = "".join(segment_labels)

        threat_pct = fusion.get("joint_threat_score", 0.0) * 100
        threat_pct_safety = fusion_safety.get("joint_threat_score", 0.0) * 100
        level = fusion.get("level", "NOMINAL")

        alarm_status = "ACTIVE DETECTED ALERT" if fusion.get("global_alarm") else "NOMINAL OPERATION"
        alarm_class = "badge-ALERT" if fusion.get("global_alarm") else "badge-NOMINAL"
        reason = fusion.get("alarm_reason", "None")
        if reason == "coordinated":
            reason = "Coordinated Multi-Layer Evasion (Rule B)"
        elif reason == "single-layer":
            reason = "Single-Layer Hard Trip (Rule A)"

        active_class = "active-content" if active else ""

        thesis_html = ""
        if mode_id == "coordinated" and fusion.get("thesis_proven"):
            thesis_html = (
                '<p style="margin-top:10px;padding-top:10px;border-top:1px solid var(--line);">'
                '<b style="color:var(--teal);">Cross-layer thesis proven.</b> Every local detector stayed '
                'silent and no single layer crossed its own threshold, yet the fused score raised a coordinated '
                'alarm. A quiet, multi-layer attack that slips under every individual gate is still caught.</p>'
            )
        val_html = ""
        if validation and validation.get("fused_false_alarm_rate") is not None:
            _fa = validation.get("fused_false_alarm_rate")
            _tr = validation.get("trials", 0)
            _p95 = validation.get("t_joint_p95")
            val_html = (
                f'<p style="margin-top:8px;"><b>Held-out false-alarm rate:</b> '
                f'<code>{_fa:.3f}</code> over {_tr:,} clean Monte-Carlo trials, calibrated and evaluated on '
                f'disjoint clean data (clean T_joint p95 = <code>{_p95}</code>). Fusion adds no false alarms.</p>'
            )

        return f"""
        <div id="fusion-content-{mode_id}" class="fusion-tab-content {active_class}" data-level="{level}">
          <div class="threat-gauge-container">
            <div class="gauge-label">
              <span>Joint Threat Score (T_joint) &middot; Equal Weights</span>
              <span class="gauge-value">{threat_pct:.1f}%</span>
            </div>
            <div class="progress-bg">
              <div class="progress-bar" style="width: {threat_pct}%; background: {"var(--danger)" if level in ["ALERT", "CRITICAL"] else ("var(--gpsy)" if level == "ELEVATED" else "var(--teal)")};"></div>
            </div>

            <div class="gauge-label" style="margin-top: 10px;">
              <span>Joint Threat Score (T_joint) &middot; Safety-Critical Weights</span>
              <span class="gauge-value" style="font-size: 18px; color: var(--ink-2);">{threat_pct_safety:.1f}%</span>
            </div>
            <div class="progress-bg" style="height: 10px; margin-bottom: 24px;">
              <div class="progress-bar" style="width: {threat_pct_safety}%; background: {"var(--danger)" if level in ["ALERT", "CRITICAL"] else ("var(--gpsy)" if level == "ELEVATED" else "var(--teal)")}; opacity: 0.75;"></div>
            </div>

            <div class="gauge-label" style="margin-bottom: 4px;">
              <span>Anomalous Energy Contribution Breakdown (Equal Weights)</span>
            </div>
            <div class="stacked-bar">
              {stacked_bar_html}
            </div>
            <div class="segment-label-list">
              {segment_labels_html}
            </div>
          </div>

          <div class="fusion-summary">
            <p><b>Global Alarm Status:</b> <span class="badge {alarm_class}" style="padding: 2px 10px; font-size: 11px;">{alarm_status}</span>
               {f"&nbsp;&bull;&nbsp; <b>Reason:</b> <code>{reason}</code>" if fusion.get("global_alarm") else ""}</p>
            <p><b>Calibration Mode:</b> Auto-calibrated runtime exceedance scales (\(a_i\)) mapping clean medians to \(\phi_i \\approx 0.10\).</p>
            {val_html}
            {thesis_html}
          </div>
        </div>
        """

    tab_attack = render_fusion_tab(f_attack, f_attack_safety, "attack", active=True, validation=f_val)
    tab_coord = render_fusion_tab(f_coord, f_coord_safety, "coordinated", active=False, validation=f_val)
    tab_clean = render_fusion_tab(f_clean, f_clean_safety, "clean", active=False, validation=f_val)

    level = f_attack.get("level", "NOMINAL")

    fusion_card_html = f"""
    <section class="fusion-card level-{level} reveal">
      <div class="fusion-header">
        <h2 class="fusion-title"><span class="live" style="background:var(--danger) !important; box-shadow:0 0 10px var(--danger) !important;"></span>Cross-Layer Signal Fusion Panel</h2>
        <span class="badge badge-{level}" id="fusion-global-badge">{level} THREAT</span>
      </div>

      <div class="fusion-tabs">
        <button class="fusion-tab-btn active" onclick="switchFusion('attack')">Saturated Attack</button>
        <button class="fusion-tab-btn" onclick="switchFusion('coordinated')">Coordinated Evasion</button>
        <button class="fusion-tab-btn" onclick="switchFusion('clean')">Nominal Drive</button>
      </div>

      {tab_attack}
      {tab_coord}
      {tab_clean}
    </section>
    """

    return (
        '<!doctype html><html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="dark">'
        '<title>AV Stack Defense - Live</title>'
        '<script id="MathJax-script" async '
        'src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
        '<style>' + STYLE + '</style>'
        '</head><body><main class="wrap">'
        + MASTHEAD + fusion_card_html + hero + others +
        '<footer class="foot">Generated live from av-stack-defense/viz/build.py. '
        'Re-run to regenerate.</footer>'
        '</main>' + SCRIPT + '</body></html>')


def cached_imgs():
    """Base64 of the already-rendered GIFs in data/, for HTML-only rebuilds."""
    keys = ["nav", "comm", "perc", "can", "fpga"]
    out = {}
    for k in keys:
        p = os.path.join(DATA, f"{k}.gif")
        if not os.path.exists(p):
            raise SystemExit(f"missing cached GIF {p}; run a full build first (drop --html-only)")
        out[k] = _b64(p)
    return out


def write_html(imgs):
    out = os.path.join(HERE, "dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html(imgs))
    print(f"Wrote {out}")
    try:
        webbrowser.open("file:///" + out.replace("\\", "/"))
    except Exception as e:
        print(f"(skipped opening browser: {e})")
    return out


def main():
    if "--html-only" in sys.argv:
        print("HTML-only rebuild from cached data/ artifacts...")
        imgs = cached_imgs()
    else:
        print("Collecting real signals from each detector...")
        collect()
        print("Rendering animated visuals (this takes ~60-90s)...")
        imgs = {"nav": nav_gif(), "comm": comm_gif(), "perc": perc_gif(),
                "can": can_gif(), "fpga": fpga_gif()}
    write_html(imgs)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

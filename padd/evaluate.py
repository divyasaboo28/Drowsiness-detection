"""Offline evaluation: replay recorded sessions through every detection system.

For each session folder (created by `python main.py record`):
  1. Extract features once from the video (cached in features.npz for speed)
  2. Fit the participant's personal profile from recorded calibration phases
  3. Replay the scripted segments through all baselines, PADD, and ablations
  4. Score alarms against ground-truth labels (alert vs drowsy segments)

Outputs comparison metrics: recall, false alarm rate, detection latency,
and false alarms broken down by category (phone, talk, mirror, laugh, normal).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baselines import BlendshapeSystem, FixedThresholdSystem, PersonalThresholdSystem
from .calibration import PhaseSamples, fit_profile
from .features import CACHE_FIELDS, FrameFeatures
from .fusion import PADDSystem
from .profile import UserProfile
from .system import ALARM

# Alarms within 10s after a drowsy segment are NOT counted as false alarms
# (the driver may still look drowsy right after the segment boundary).
GRACE_AFTER_DROWSY = 10.0
# An alarm up to 3s after a drowsy segment ends still counts as a detection
# (allows for natural response delay in the multi-cue fusion pipeline).
DETECT_SLACK = 3.0

SYSTEM_ORDER = [
    "baseline_fixed", "baseline_blendshape", "baseline_personal",
    "padd_no_pose", "padd_no_nod", "padd_no_adapt", "padd",
]
SYSTEM_LABELS = {
    "baseline_fixed": "A: fixed EAR",
    "baseline_blendshape": "Blendshapes",
    "baseline_personal": "B: personal threshold",
    "padd_no_pose": "PADD w/o pose norm",
    "padd_no_nod": "PADD w/o nod",
    "padd_no_adapt": "PADD w/o adaptation",
    "padd": "PADD (full)",
}


def make_systems(profile: UserProfile) -> dict:
    return {
        "baseline_fixed": FixedThresholdSystem(),
        "baseline_blendshape": BlendshapeSystem(),
        "baseline_personal": PersonalThresholdSystem(profile),
        "padd_no_pose": PADDSystem(profile, use_pose_norm=False, short="padd_no_pose"),
        "padd_no_nod": PADDSystem(profile, use_nod=False, short="padd_no_nod"),
        "padd_no_adapt": PADDSystem(profile, use_adapt=False, short="padd_no_adapt"),
        "padd": PADDSystem(profile),
    }


# --------------------------------------------------------------------------
# Session I/O
# --------------------------------------------------------------------------
@dataclass
class SegmentRow:
    start: float
    end: float
    phase: str
    label: str
    category: str


def load_segments(session: Path) -> list[SegmentRow]:
    with open(session / "segments.csv", newline="") as fh:
        return [SegmentRow(float(r["start"]), float(r["end"]), r["phase"], r["label"], r["category"])
                for r in csv.DictReader(fh)]


def extract_features(session: Path, tracker=None) -> tuple[list[FrameFeatures | None], np.ndarray]:
    """Features per frame (None = no face) and every frame's timestamp."""
    cache = session / "features.npz"
    if cache.exists():
        d = np.load(cache)
        out = []
        for i in range(len(d["t"])):
            if not d["valid"][i]:
                out.append(None)
                continue
            out.append(FrameFeatures(t=float(d["t"][i]), **{k: float(d[k][i]) for k in CACHE_FIELDS}))
        return out, d["t"].astype(float)

    from .landmarks import FaceTracker, Source

    own_tracker = tracker is None
    tracker = tracker or FaceTracker()
    src = Source(video=str(session / "video.mp4"))
    feats, times = [], []
    while True:
        frame, t = src.read()
        if frame is None:
            break
        feats.append(tracker.process(frame, t))
        times.append(t)
        if len(feats) % 300 == 0:
            print(f"  {session.name}: {len(feats)} frames")
    src.release()
    if own_tracker:
        tracker.close()

    arrays = {"t": np.array(times), "valid": np.array([f is not None for f in feats])}
    for k in CACHE_FIELDS:
        arrays[k] = np.array([getattr(f, k) if f else np.nan for f in feats])
    np.savez_compressed(cache, **arrays)
    for f in feats:
        if f is not None:
            f.points = None
    return feats, arrays["t"]


def calibration_samples(segments: list[SegmentRow], feats, times) -> PhaseSamples:
    samples: PhaseSamples = {}
    for s in segments:
        if s.label != "calib":
            continue
        samples[s.phase] = [(t - s.start, f) for t, f in zip(times, feats) if s.start <= t <= s.end]
    return samples


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_run(times: np.ndarray, levels: np.ndarray, segments: list[SegmentRow]) -> dict:
    """Score one system's alarm output against ground-truth segment labels.
    
    Returns a dict with:
      - false_alarms: total false alarms during alert segments
      - fa_per_hour: false alarm rate normalized per hour
      - recall: fraction of drowsy episodes that triggered at least one alarm
      - median_latency: median seconds between drowsy onset and first alarm
      - fa_by_category: false alarms broken down by scenario type
    """
    # Find all alarm onset times (rising edges only — not sustained alarms)
    alarm = levels == ALARM
    onset_idx = np.where(alarm & ~np.r_[False, alarm[:-1]])[0]
    onsets = times[onset_idx]
    drowsy = [s for s in segments if s.label == "drowsy"]
    alert = [s for s in segments if s.label == "alert"]

    # Count false alarms: alarms during alert segments, excluding the
    # grace period right after a drowsy segment ends
    fa_by_cat: dict[str, int] = {}
    false_alarms = 0
    for t in onsets:
        seg = next((s for s in alert if s.start <= t <= s.end), None)
        if seg is None:
            continue
        if any(0 <= t - d.end <= GRACE_AFTER_DROWSY for d in drowsy):
            continue
        false_alarms += 1
        fa_by_cat[seg.category] = fa_by_cat.get(seg.category, 0) + 1

    # Count detections: for each drowsy segment, check if any alarm fired
    # during the segment (with a small slack window after it ends)
    detected, latencies = 0, []
    for d in drowsy:
        hit = np.where(alarm & (times >= d.start) & (times <= d.end + DETECT_SLACK))[0]
        if len(hit):
            detected += 1
            latencies.append(max(0.0, times[hit[0]] - d.start))

    alert_hours = sum(s.end - s.start for s in alert) / 3600.0
    return {
        "false_alarms": false_alarms,
        "alert_hours": alert_hours,
        "fa_per_hour": false_alarms / alert_hours if alert_hours > 0 else np.nan,
        "drowsy_episodes": len(drowsy),
        "detected": detected,
        "recall": detected / len(drowsy) if drowsy else np.nan,
        "median_latency": float(np.median(latencies)) if latencies else np.nan,
        "fa_by_category": fa_by_cat,
    }


def evaluate_session(session: Path, tracker=None) -> tuple[UserProfile, dict[str, dict]]:
    segments = load_segments(session)
    feats, times = extract_features(session, tracker)

    profile = fit_profile(calibration_samples(segments, feats, times), session.name)
    script = [s for s in segments if s.label != "calib"]
    if not script:
        raise ValueError(f"{session} has no alert/drowsy segments to score")
    t0 = min(s.start for s in script)
    idx = np.where(times >= t0)[0]

    results = {}
    for key, system in make_systems(profile).items():
        levels = np.array([system.update(feats[i], times[i]).level for i in idx])
        results[key] = score_run(times[idx], levels, script)
    return profile, results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def find_sessions(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "segments.csv").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("segments.csv") if (p.parent / "video.mp4").exists()
                  or (p.parent / "features.npz").exists())


def run_evaluation(root: Path, out: Path) -> None:
    sessions = find_sessions(root)
    if not sessions:
        raise SystemExit(f"No sessions found under {root} (need segments.csv + video.mp4)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    tracker = None
    if any(not (s / "features.npz").exists() for s in sessions):
        from .landmarks import FaceTracker
        tracker = FaceTracker()

    per_session: dict[str, dict[str, dict]] = {}
    for s in sessions:
        print(f"Evaluating {s} ...")
        profile, res = evaluate_session(s, tracker)
        per_session[s.name] = res
        for w in profile.warnings:
            print(f"  calibration warning: {w}")
    if tracker:
        tracker.close()

    rows = []
    for sess, res in per_session.items():
        for key in SYSTEM_ORDER:
            r = res[key]
            rows.append({"session": sess, "system": key, **{k: v for k, v in r.items() if k != "fa_by_category"},
                         **{f"fa_{c}": n for c, n in r["fa_by_category"].items()}})
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("session", "system"), k))
    with open(out / "per_session.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, restval=0)
        wr.writeheader()
        wr.writerows(rows)

    summary = summarize(per_session)
    with open(out / "summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(next(iter(summary.values())).keys()))
        wr.writeheader()
        wr.writerows(summary.values())

    stats_line = wilcoxon_line(per_session)
    write_markdown(out / "summary.md", summary, stats_line, len(sessions))
    make_charts(out, summary, per_session)
    print((out / "summary.md").read_text())
    print(f"Results written to {out.resolve()}")


def summarize(per_session) -> dict[str, dict]:
    summary = {}
    for key in SYSTEM_ORDER:
        rs = [res[key] for res in per_session.values()]
        fa = sum(r["false_alarms"] for r in rs)
        hours = sum(r["alert_hours"] for r in rs)
        det = sum(r["detected"] for r in rs)
        eps = sum(r["drowsy_episodes"] for r in rs)
        per_fa = np.array([r["fa_per_hour"] for r in rs], dtype=float)
        lat = [r["median_latency"] for r in rs if np.isfinite(r["median_latency"])]
        cats: dict[str, int] = {}
        for r in rs:
            for c, n in r["fa_by_category"].items():
                cats[c] = cats.get(c, 0) + n
        summary[key] = {
            "system": key,
            "label": SYSTEM_LABELS[key],
            "false_alarms": fa,
            "fa_per_hour": fa / hours if hours else np.nan,
            "fa_per_hour_sd_across_people": float(np.nanstd(per_fa)) if len(per_fa) > 1 else 0.0,
            "recall": det / eps if eps else np.nan,
            "median_latency_s": float(np.median(lat)) if lat else np.nan,
            "fa_phone": cats.get("phone", 0),
            "fa_talk": cats.get("talk", 0),
            "fa_mirror": cats.get("mirror", 0),
            "fa_laugh": cats.get("laugh", 0),
            "fa_normal": cats.get("normal", 0),
        }
    return summary


def wilcoxon_line(per_session) -> str:
    a = np.array([r["baseline_fixed"]["fa_per_hour"] for r in per_session.values()])
    b = np.array([r["padd"]["fa_per_hour"] for r in per_session.values()])
    if len(a) < 5:
        return f"Paired Wilcoxon test needs at least 5 participants (have {len(a)})."
    try:
        from scipy.stats import wilcoxon
        p = wilcoxon(a, b).pvalue
        return f"Paired Wilcoxon signed-rank, false alarms/h, fixed vs PADD: p = {p:.4f} (n = {len(a)})"
    except ValueError as e:
        return f"Wilcoxon test not possible: {e}"


def write_markdown(path: Path, summary, stats_line: str, n_sessions: int) -> None:
    lines = [
        f"# PADD evaluation ({n_sessions} session{'s' if n_sessions != 1 else ''})",
        "",
        "| System | False alarms/h | SD across people | Recall | Median latency (s) | Phone | Talk | Mirror | Laugh | Normal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary.values():
        lines.append(
            f"| {s['label']} | {s['fa_per_hour']:.1f} | {s['fa_per_hour_sd_across_people']:.1f} | "
            f"{s['recall'] * 100:.0f}% | {s['median_latency_s']:.1f} | {s['fa_phone']} | {s['fa_talk']} | "
            f"{s['fa_mirror']} | {s['fa_laugh']} | {s['fa_normal']} |"
        )
    lines += ["", stats_line, ""]
    path.write_text("\n".join(lines))


def make_charts(out: Path, summary, per_session) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, light_blue, grey = "#2a78d6", "#86b6ef", "#b4b2ac"
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2,
        "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })

    def bar_color(k):
        return blue if k == "padd" else light_blue if k.startswith("padd") else grey

    def bar_chart(metric, title, fname, fmt, scale=1.0):
        keys = SYSTEM_ORDER
        vals = [summary[k][metric] * scale for k in keys]
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        y = np.arange(len(keys))
        ax.barh(y, vals, height=0.6, color=[bar_color(k) for k in keys], edgecolor="#fcfcfb", linewidth=2)
        ax.set_yticks(y, [SYSTEM_LABELS[k] for k in keys])
        ax.invert_yaxis()  # baselines on top, full PADD at the bottom
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, " " + fmt.format(v), va="center", color=ink, fontsize=9)
        ax.set_title(title, loc="left", color=ink, fontsize=12)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    bar_chart("fa_per_hour", "False alarms per hour of alert driving (lower is better)",
              "false_alarms_per_hour.png", "{:.1f}")
    bar_chart("recall", "Drowsy episodes detected (higher is better)", "recall.png", "{:.0f}%", 100)

    # Per-participant comparison: does personalization help everyone?
    names = list(per_session)
    a = [per_session[n]["baseline_fixed"]["fa_per_hour"] for n in names]
    b = [per_session[n]["padd"]["fa_per_hour"] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(names) + 1.6))
    y = np.arange(len(names))
    for yi, va, vb in zip(y, a, b):
        ax.plot([va, vb], [yi, yi], color=grid, linewidth=2, zorder=1)
    ax.scatter(a, y, s=64, color=grey, label=SYSTEM_LABELS["baseline_fixed"], zorder=2,
               edgecolor="#fcfcfb", linewidth=2)
    ax.scatter(b, y, s=64, color=blue, label=SYSTEM_LABELS["padd"], zorder=3,
               edgecolor="#fcfcfb", linewidth=2)
    ax.set_yticks(y, names)
    ax.set_xlabel("False alarms per hour")
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("False alarms per participant", loc="left", color=ink, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "per_participant.png", dpi=150)
    plt.close(fig)

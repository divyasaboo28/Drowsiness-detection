# PADD — Personalized Adaptive Drowsiness Detector

Real-time drowsiness detection from an ordinary webcam that **learns each user's face**
instead of applying one fixed threshold to everyone.

Most drowsiness demos use a single rule such as *"eye aspect ratio < 0.25 for 48 frames → alarm"*.
That rule fires constantly for people with naturally narrow eyes, misses people with wide eyes,
and treats glancing down at a phone as "eyes closing". PADD fixes this with a 90-second
personal calibration and several cues that must agree.

```
Webcam → MediaPipe Face Landmarker (478 landmarks + head pose)
       → features: EAR, MAR, pitch/yaw
       → personal normalization (profile + pose model)          ← calibration
       → detectors: blinks · PERCLOS · yawns · head nods
       → confidence-weighted evidence fusion + hysteresis
       → OK / CAUTION / ALARM   (+ safety override: eyes closed > 2 s)
```

**Contents**

1. [What is new](#1-what-is-new)
2. [How to run it (step by step)](#2-how-to-run-it-step-by-step)
3. [The five commands](#3-the-five-commands)
4. [Running the full experiment](#4-running-the-full-experiment)
5. [File-by-file guide](#5-file-by-file-guide)
6. [How the decision is made](#6-how-the-decision-is-made)
7. [Tuning](#7-tuning)
8. [Troubleshooting](#8-troubleshooting)
9. [Limitations](#9-limitations)

---

## 1. What is new

| # | Contribution | Problem it solves |
|---|---|---|
| **N1** | **Pose-normalized eye openness**: a per-user model of how open your eyes look at each head angle, learned from a short "follow the dot" sweep | Looking down narrows the eyelids; fixed-threshold systems read this as eye closure (the "phone glance" false alarm) |
| **N2** | **Personal blink signature**: your own blink duration and eyelid reopening speed (95th percentiles), not a universal number | Slow, heavy blinks are an early fatigue sign, but "slow" differs per person |
| **N3** | **Nod-off signature**: slow droop followed by a fast jerk back up | Separates drowsy nodding from deliberately looking down (fast drop, then hold) |
| **N4** | **Guarded online adaptation + confidence-weighted fusion**: the baseline follows lighting/posture drift only while you are clearly alert; cues count for less when landmarks are unreliable (head turned, face small) | Stops the system from slowly learning drowsiness as "normal"; fewer false alarms when checking mirrors |

The project also keeps your original two ideas, **personal calibration** and **multi-cue fusion**
(eyes + blinks + yawns + nods). A **safety override** means fusion can never delay the most
dangerous case: eyes closed for 2 s always raises the alarm.

---

## 2. How to run it (step by step)

These steps are for Windows. macOS/Linux differences are noted.

### Step 1: Check Python

You need **Python 3.10, 3.11 or 3.12** (MediaPipe does not support every newer version yet).
Open a terminal (PowerShell, or the terminal in VS Code) and run:

```bash
python --version
```

If it prints 3.10–3.12, you're set. If you have several versions installed, `py -0` lists them,
and you can pick one with, for example, `py -3.10`.

### Step 2: Open a terminal in the project folder

In GitHub Desktop: **Repository → Open in Command Prompt** (or **Open in Visual Studio Code**,
then **Terminal → New Terminal**). Or run:

```bash
cd "C:\Users\<you>\OneDrive\Documents\GitHub\Drowsiness detection"
```

### Step 3: Create a virtual environment (once)

A virtual environment keeps this project's packages separate from everything else on your PC.

```bash
python -m venv .venv
```

Activate it. You need to do this every time you open a new terminal:

```bash
.venv\Scripts\activate
```

Your prompt now starts with `(.venv)`. On macOS/Linux use `source .venv/bin/activate`.

> **Project inside OneDrive?** OneDrive will try to sync the ~150 MB `.venv` folder.
> Either right-click `.venv` → **Free up space**, or create the environment outside OneDrive,
> e.g. `python -m venv C:\venvs\padd`, and activate it with `C:\venvs\padd\Scripts\activate`.
> Git ignores `.venv` either way.

> **PowerShell says "running scripts is disabled"?** Run
> `Set-ExecutionPolicy -Scope Process Bypass` first, then activate again. This only affects
> the current window. Alternatively, skip activation and type `.venv\Scripts\python` instead
> of `python` in every command below.

### Step 4: Install the packages (once)

```bash
pip install -r requirements.txt
```

This installs MediaPipe, OpenCV, NumPy, SciPy, Matplotlib and pytest, about 150 MB in total.

### Step 5: Check everything works (no camera needed)

```bash
pytest
```

You should see `12 passed`. This tests the detection logic on a synthetic person.

### Step 6: Calibrate yourself (~90 seconds)

```bash
python main.py calibrate --name yourname
```

- The first run downloads the MediaPipe face model (~4 MB) into `models/`.
- A full-screen window opens. Sit normally, about an arm's length from the camera, with your face lit from the front.
- Follow the instruction at the bottom of the screen. A short beep marks each new phase.
- Press **Esc** to cancel at any time.

When it finishes, the terminal prints your profile, for example your open/closed eye values and
your blink rate, plus any warnings. The profile is saved to `profiles/yourname.json`.

### Step 7: Run it

```bash
python main.py compare --name yourname
```

The left half shows the classic fixed-threshold detector and the right half shows PADD, both on
the same camera feed. Try:

1. **Look down at your phone for a few seconds.** The left side alarms; PADD stays OK.
2. **Talk animatedly.** Neither should alarm.
3. **Act drowsy:** slow heavy blinks, a yawn, let your head droop slowly and jerk it back up,
   close your eyes for a moment. PADD's score rises and the "Why:" line shows which cues fired.
4. **Close your eyes for 2+ seconds.** PADD alarms immediately (safety override).

Keys: **q** quit · **m** mute the beep · **s** save a screenshot to `screenshots/`.

---

## 3. The five commands

All commands are run as `python main.py <command> [options]`.

| Command | What it does | Main options |
|---|---|---|
| `calibrate --name NAME` | Guided ~90 s calibration; saves `profiles/NAME.json` | `--skip-yawn`, `--camera N` |
| `run --name NAME` | Live detection with PADD only | `--no-sound`, `--camera N`, `--video FILE` |
| `compare --name NAME` | Split screen: fixed baseline (left) vs PADD (right), the demo view | `--no-sound`, `--camera N`, `--video FILE` |
| `record --name ID` | Records calibration + a ~6 min scripted, auto-labelled test session | `--out FOLDER`, `--skip-yawn`, `--camera N` |
| `evaluate` | Replays every recorded session through all systems; writes tables and charts | `--sessions FOLDER` (default `sessions`), `--out FOLDER` (default `results`) |

- `--camera 1` picks a second webcam (for example, an external one).
- `--video file.mp4` runs on a video file instead of the camera.
- `python main.py --help` and `python main.py <command> --help` list everything.

---

## 4. Running the full experiment

This is how you produce the evidence that PADD has fewer false alarms than the basic version.

### Record participants

For each person (ideally 8–12, with variety: glasses, eye shapes, lighting):

```bash
python main.py record --name p01
```

The recording starts with the calibration phases, then runs a script of 16 segments shown on screen:

| Type | Segments |
|---|---|
| Normal (alert) | Act normal, several times |
| False-alarm traps (alert) | Look at phone ×3, talk, check mirrors, laugh |
| Acted drowsiness (drowsy) | Slow blinks + yawn, head nodding, 3 s "microsleep", full combo |

The output is saved to `sessions/p01_<date>_<time>/`:

| File | Contents |
|---|---|
| `video.mp4` | Raw camera frames, without overlays |
| `frames.csv` | Timestamp of every frame, so replays keep the real timing |
| `segments.csv` | Start/end time and label of every phase and segment |

### Evaluate

```bash
python main.py evaluate
```

For each session, `evaluate` extracts face features (cached in `features.npz`, so later runs are
fast), fits that person's profile from their recorded calibration, then replays the scripted part
through seven systems:

| System | What it is |
|---|---|
| A: fixed EAR | Classic rule: EAR < 0.22 for 1.6 s, or MAR > 0.5 for 1 s |
| Blendshapes | MediaPipe's own `eyeBlink` / `jawOpen` scores with fixed cut-offs |
| B: personal threshold | Calibrated eye scale only: no pose model, no fusion |
| PADD w/o pose norm | Ablation: N1 removed |
| PADD w/o nod | Ablation: N3 removed |
| PADD w/o adaptation | Ablation: N4 adaptation removed |
| PADD (full) | Everything |

Results are written to `results/`:

| File | Contents |
|---|---|
| `summary.md` | The main table: false alarms/hour, spread across people, recall, latency, false alarms by trigger |
| `summary.csv` | The same, as a spreadsheet |
| `per_session.csv` | Every metric for every person × system |
| `false_alarms_per_hour.png` | Bar chart: the headline result |
| `recall.png` | Bar chart: share of drowsy episodes caught |
| `per_participant.png` | Fixed baseline vs PADD for each person |

How the scoring works:

- A **false alarm** is an alarm that *starts* during an alert segment. Alarms in the 10 s after a drowsy segment don't count, because that is the leftover from a real episode.
- A drowsy episode is **detected** if an alarm is active at any point from its start until 3 s after its end.
- **Latency** is the time from the start of the episode to the first alarm.
- With 5 or more participants, a paired **Wilcoxon signed-rank test** (fixed vs PADD) is added.

### Using public datasets

Any video can be evaluated. Put it in its own folder as `video.mp4`, next to a `segments.csv`:

```csv
start,end,phase,label,category
0,60,rest,calib,rest
60,600,s01_normal,alert,normal
600,900,s02_drowsy,drowsy,drowsy_real
```

- `label` must be `calib`, `alert` or `drowsy`.
- Only the `rest` calibration phase is required. Missing phases fall back to safe defaults and print a warning.
- Good datasets: **UTA-RLDD** (real drowsiness, 60 subjects), **NTHU-DDD** (glasses, night), **YawDD** (yawning vs talking).

---

## 5. File-by-file guide

```
Drowsiness detection/
├── main.py               command-line entry point
├── requirements.txt      Python packages
├── pytest.ini            test settings
├── .gitignore            keeps personal data and the model out of Git
├── README.md             this file
├── padd/                 the detector itself (a Python package)
│   ├── __init__.py
│   ├── config.py         every tunable number
│   ├── landmarks.py      camera/video input + MediaPipe face tracking
│   ├── features.py       per-frame measurements (EAR, MAR, head pose)
│   ├── profile.py        user profile + personal normalization + adaptation
│   ├── calibration.py    guided calibration and profile fitting
│   ├── detectors.py      blink, PERCLOS, yawn, nod detectors
│   ├── system.py         shared interface and alert levels
│   ├── fusion.py         PADD: combines all cues into one decision
│   ├── baselines.py      the systems PADD is compared against
│   ├── ui.py             on-screen drawing and beeps
│   ├── record.py         scripted session recorder
│   └── evaluate.py       offline replay, metrics and charts
└── tests/
    └── test_padd.py      automated tests on a synthetic person
```

Folders created while you use it (all git-ignored):

| Folder | Created by | Contents |
|---|---|---|
| `models/` | first run | `face_landmarker.task`, the MediaPipe model |
| `profiles/` | `calibrate` | One JSON profile per person |
| `sessions/` | `record` | Recorded test sessions |
| `screenshots/` | **s** key | Screenshots of the live view |
| `results/` | `evaluate` | Tables and charts (not ignored, so you can commit your results) |

**Data flow for one frame:**

```
landmarks.Source.read()           → frame + timestamp
landmarks.FaceTracker.process()   → features.FrameFeatures (EAR, MAR, pitch, yaw, ...)
fusion.PADDSystem.update()        → uses profile.PersonalModel + detectors.*
                                   → system.Status (level, score, cues, reasons)
ui.draw_status() / ui.Beeper      → what you see and hear
```

---

### `main.py` — command-line entry point

The only file you run. It reads the command (`calibrate`, `run`, `compare`, `record`,
`evaluate`), opens the camera or video, and calls the right module.

| Function | Role |
|---|---|
| `main()` | Defines all commands and options with `argparse` |
| `cmd_calibrate()` | Runs `calibration.run_calibration()`, prints and saves the profile |
| `live_loop()` | Shared loop for `run` and `compare`: read frame → track face → update each system → draw panels → beep → handle keys |
| `cmd_run()` / `cmd_compare()` | Start `live_loop` with `[PADD]` or `[Baseline, PADD]` |
| `cmd_record()` | Runs `record.run_recording()` |
| `cmd_evaluate()` | Runs `evaluate.run_evaluation()` |
| `load_profile()` | Loads `profiles/NAME.json`, or tells you to calibrate first |

Imports happen inside each command, so `--help` works even before MediaPipe is installed.

---

### `padd/config.py` — every tunable number

No logic, only constants, grouped by topic. Change values here; you never need to edit code to
experiment. Main groups:

| Group | Examples | Meaning |
|---|---|---|
| Model | `MODEL_URL`, `MODEL_PATH` | Where the face model comes from and is saved |
| Landmark indices | `LEFT_EYE`, `RIGHT_EYE`, `MOUTH_VERTICAL` | Which of MediaPipe's 478 points are used for EAR/MAR |
| Baseline A | `BASE_EAR_THRESHOLD = 0.22`, `BASE_EYE_CLOSED_SECONDS = 1.6` | The classic fixed rule |
| Openness | `OPEN_CLOSE_T = 0.35`, `PERCLOS_T = 0.2` | Thresholds on the *personal* 0–1 eye scale |
| Blinks | `LONG_CLOSURE_MIN = 1.0`, `OVERRIDE_CLOSURE = 2.0` | Long closure and safety-override durations |
| Yawns | `YAWN_MIN_DURATION = 1.2`, `YAWN_MAX_OSC_RATE` | Minimum yawn length; how much zig-zag counts as talking |
| Nods | `NOD_DROOP_MIN = 10`, `NOD_SLOW_MAX_VEL = 20`, `NOD_RECOVERY_VEL = 40` | Degrees and degrees/second for the nod signature |
| Head down | `HEAD_SLUMP_TIME = 3`, `HEAD_DOWN_ALARM = 12` | Seconds a slow droop may stay down (slump), and seconds any head-down may last, before ALARM |
| Evidence | `TAU_*`, `FULL_*` | How long each event "counts" and how many events saturate a cue |
| Fusion | `WEIGHTS`, `CAUTION_ON/OFF`, `ALARM_ON/OFF`, `ALARM_HOLD`, `RECENT_WINDOW` | Cue weights, alert thresholds with hysteresis, and how fresh the evidence must be for ALARM |
| Confidence | `YAW_FULL_CONF`, `YAW_ZERO_CONF` | How much head turning reduces trust in the eyes/mouth |
| Adaptation | `ADAPT_MAX_SCORE`, `ADAPT_TAU_*`, `ADAPT_MAX_*` | When and how fast the baseline may drift, and how far |

---

### `padd/landmarks.py` — camera input and face tracking

| Item | Role |
|---|---|
| `ensure_model()` | Downloads `face_landmarker.task` on first use |
| `FaceTracker` | Wraps MediaPipe's **Face Landmarker** in video mode. `process(frame, t)` returns a `FrameFeatures`, or `None` if no face is found. It also asks MediaPipe for head-pose matrices and blendshapes. |
| `Source` | One interface for a webcam or a video file. `read()` returns `(frame, timestamp)`. Webcam frames are mirrored and timed by the wall clock. Video frames take their timestamps from `frames.csv` if present, so a replay sees exactly the live timing. On Windows it tries the DirectShow backend first, because it opens webcams faster. |

---

### `padd/features.py` — per-frame measurements

Turns 478 landmark points into a few meaningful numbers.

| Item | Role |
|---|---|
| `FrameFeatures` | Dataclass for one frame: `ear_l`, `ear_r`, `ear`, `mar`, `pitch`, `yaw`, `roll`, `eye_width_px`, `blink_bs`, `jaw_bs`, plus pixel `points` for drawing |
| `eye_aspect_ratio()` | **EAR** = (two vertical eyelid distances) / (2 × eye width). Around 0.25–0.35 when open, near 0.05 when closed. |
| `mouth_aspect_ratio()` | **MAR** = average of three inner-lip vertical gaps / mouth width. Near 0 when closed, 0.6+ when yawning. |
| `euler_from_matrix()` | Converts MediaPipe's 4×4 face transform into pitch/yaw/roll in degrees |
| `from_points()` | Builds `FrameFeatures`. The two eyes are averaged, weighted by apparent width, so the eye turned away from the camera counts less. |
| `from_mediapipe()` | Same, starting from a raw MediaPipe result |
| `yaw_confidence()` | 1.0 when facing the camera, fading to 0 between 25° and 45° of head turn |
| `size_confidence()` | Lower confidence when the face is tiny (far from the camera) |
| `CACHE_FIELDS` | Which fields `evaluate.py` saves to `features.npz` |

---

### `padd/profile.py` — the user profile and personal normalization

| Item | Role |
|---|---|
| `UserProfile` | Everything calibration learned about one person, saved as JSON: open/closed EAR, the pose model coefficients, which way is "head down" (`pitch_sign`), neutral posture, blink percentiles and rates, baseline PERCLOS, mouth levels, yawn threshold, warnings. `save()`, `load()` and `summary()` are included. |
| `pose_design()` | The features of the pose model: `[1, pitch, pitch², |yaw|]` (scaled). Shared by fitting and runtime, so they always agree. |
| `PersonalModel` | The runtime wrapper around a profile: |
| `.rel_pitch()` | Head pitch relative to *your* neutral posture; positive = head down |
| `.expected_open_ear()` | **N1**: how open your eyes *should* look at this head angle when awake |
| `.openness()` | Your current EAR on your personal scale: 0 = your closed eye, 1 = your normal open eye *for this pose* |
| `.adapt()` | **N4**: slowly shifts the eye baseline (time constant 60 s) and neutral posture (120 s) to follow lighting and slouching. Changes are limited in size. Posture only adapts while the head is near level, so a phone glance never becomes the new "straight ahead". |

The adaptation state lives only in memory and is never written back to the JSON file.

---

### `padd/calibration.py` — guided calibration and profile fitting

| Item | Role |
|---|---|
| `Phase`, `PHASES` | The 9 calibration phases (rest, 5 pose targets, talk, closed eyes, yawn), with durations, on-screen text and how many seconds to skip at the start (reaction time) |
| `fit_profile(samples, name)` | **The core.** Takes features grouped by phase and builds a `UserProfile`. Steps: |
| | 1. **Eye scale**: open EAR from the rest phase; closed EAR from the closed-eyes phase (with a fallback if you didn't close them) |
| | 2. **Head direction**: compares "look down" and "look up" to find which pitch sign means "down" |
| | 3. **Pose model (N1)**: ridge regression of EAR on pitch/yaw across rest + pose phases. Blinks are removed per phase, because a downward-looking eye is legitimately narrow. |
| | 4. **Blink signature (N2)**: runs the blink detector over the rest phase to get your blink rate, 95th/99th percentile durations, reopening speed and baseline PERCLOS |
| | 5. **Mouth**: resting MAR, talking 95th percentile, and a yawn threshold placed above talking but below your yawn |
| `run_calibration()` | The live, full-screen version: a 3 s face check, then each phase with instructions, progress bar, target dot and beeps. It then calls `fit_profile`. |
| `draw_phase()`, `target_point()` | Draw the instructions, dot or arrow (also used by `record.py`) |

Because `fit_profile` only needs features grouped by phase, the same code fits profiles live and
from recorded videos.

---

### `padd/detectors.py` — event detectors

All detectors work on timestamps in seconds, so they behave the same at 15 fps or 60 fps.
Passing `None` means "no reliable reading this frame".

| Class | What it detects | How |
|---|---|---|
| `BlinkDetector` | Blinks, slow blinks, long closures | A state machine on openness: *open → closed* (below 0.35) → *reopening* (above 0.5) → *open* (above 0.8). It measures closed time and reopening time. **Slow** = longer than your 95th percentile, or slower to reopen. **Long** = at least 1 s. `closed_for(t)` gives the current closure length for the safety override. |
| `PerclosTracker` | PERCLOS | Share of the last 60 s with openness below 0.2 (eyes at least 80% closed) |
| `YawnDetector` | Yawns | MAR above your threshold for at least 1.2 s. It then counts direction changes in the mouth signal; more than 2 per second means **talking**, which is rejected. |
| `NodDetector` | Drowsy head nods (**N3**) | Smooths pitch and computes velocity. It enters *droop* only if the head went from level to 10° down **slowly** (≤ 20°/s average). A fast drop is a deliberate *look-down* and is ignored. A nod is counted when a droop ends with a **fast** upward snap (≥ 40°/s). A droop that stays down is a *slump*; `slumped_for()` reports how long. |
| `DecayingEvidence` | Turns events into a 0–1 score | Each event adds `exp(-age/τ)`. Your resting rate is subtracted, and the result is divided by a saturation count. |
| `direction_changes()` | Helper | Counts zig-zags larger than a given amplitude |

---

### `padd/system.py` — shared interface

| Item | Role |
|---|---|
| `OK`, `CAUTION`, `ALARM` | Alert levels 0, 1, 2 |
| `Status` | What every system returns each frame: `level`, `score` (0–1), `cues` (per-cue contribution), `reasons` (text for "Why:"), `info` (numbers for the overlay) |
| `DrowsinessSystem` | Base class. `update(features, t)` calls the subclass's `_update` and counts alarm onsets in `alarm_count`. |

Because every system shares this interface, `compare` and `evaluate` can treat baselines and PADD identically.

---

### `padd/fusion.py` — PADD, the main detector

`PADDSystem(profile, use_pose_norm=True, use_nod=True, use_adapt=True)`. The three flags exist
for the ablation study.

What `update()` does every frame:

1. **Normalize**: relative pitch/yaw, eye and mouth confidence, personal openness
2. **Detect**: feed the blink, PERCLOS, yawn and nod detectors (low-confidence frames are skipped per cue)
3. **Score cues**: eyes (PERCLOS or long closure), slow blinks, yawns, nods, each 0–1
4. **Fuse**: Σ weight × reliability × score, smoothed by a leaky integrator (τ = 1 s)
5. **Decide**: hysteresis state machine:
   - CAUTION when the score goes above 0.35; back to OK below 0.25
   - ALARM when the score stays above 0.50 for 1 s **and** there was a drowsy event (slow blink, long closure, yawn, nod) in the last 6 s or the eyes are closed now; leaves ALARM below 0.40
   - With no fresh event for 6 s, ALARM drops to the silent CAUTION level. Evidence is remembered for minutes, so without this rule the alarm would stay on long after you are alert again
6. **Override**: ALARM regardless of the score when
   - the eyes are closed for 2 s or more (only while the eyes are actually visible), or
   - the head drooped **slowly** and has stayed down for 3 s (a slump: falling asleep), or
   - the head has been down for 12 s, however it got there (eyes off the road too long). If the face disappears while the head is down, which happens when the chin drops to the chest, the head counts as still down
7. **Adapt**: only when the score is below 0.15, the eyes are open and nothing is closing
8. **Explain**: builds the "Why:" reasons and the overlay numbers

`counts` keeps running totals of slow blinks, long closures, yawns and nods. This is useful for debugging.

---

### `padd/baselines.py` — what PADD is compared against

All three share `_SustainedRule`: alarm while "eye closed" has lasted 1.6 s, or for 3 s after
"mouth open" has lasted 1 s.

| Class | "Eye closed" means | "Mouth open" means |
|---|---|---|
| `FixedThresholdSystem` (A) | EAR < 0.22 | MAR > 0.5 |
| `PersonalThresholdSystem` (B) | Personal openness < 0.35, **without** pose correction | MAR > your yawn threshold |
| `BlendshapeSystem` | MediaPipe `eyeBlink` > 0.5 | MediaPipe `jawOpen` > 0.5 |

---

### `padd/ui.py` — drawing and sound

| Item | Role |
|---|---|
| `draw_landmarks()` | Dots on the eye points, lines on the mouth |
| `draw_status()` | Coloured banner (green/yellow/red), measurement panel, alarm counter, cue bars with the drowsiness-score bar and threshold marks, the "Why:" line, and a red border when alarming |
| `side_by_side()`, `fit_to_screen()` | Build the split-screen view and shrink it to fit your screen |
| `Beeper` | Non-blocking alarm beep at most every 0.8 s (`winsound` on Windows, terminal bell elsewhere) |
| `chime()` | Short tone at calibration/recording phase changes |
| `text()`, `draw_bar()` | Small drawing helpers |

---

### `padd/record.py` — scripted session recorder

| Item | Role |
|---|---|
| `Segment`, `SCRIPT` | The 16 test segments (category, alert/drowsy label, duration, instruction). Edit `SCRIPT` to change the protocol. |
| `run_recording()` | Full-screen recorder: calibration phases, then the script. It saves frames **before** drawing instructions on them, so the video stays clean. It writes `video.mp4`, `frames.csv` and `segments.csv`. Esc stops early and keeps what was recorded. |
| `draw_segment()` | On-screen instruction, countdown and REC marker |

---

### `padd/evaluate.py` — offline replay and metrics

| Function | Role |
|---|---|
| `find_sessions()` | Finds every folder containing `segments.csv` with a video or cached features |
| `extract_features()` | Runs MediaPipe over a video once and caches the result in `features.npz` |
| `calibration_samples()` | Cuts the recorded calibration phases out of the session |
| `make_systems()` | Builds the 7 systems for one person's profile |
| `evaluate_session()` | Fits the profile, then replays the scripted part through each system |
| `score_run()` | False alarms (with category and grace period), recall, latency |
| `summarize()`, `wilcoxon_line()` | Totals across people, spread across people, significance test |
| `write_markdown()`, `make_charts()` | `summary.md` and the three PNG charts |

---

### `tests/test_padd.py` — automated tests

The tests build a **synthetic person** with narrow eyes (open EAR 0.24) whose EAR drops to 0.14
when looking down, and whose head-pitch sign is reversed. No camera is needed. The tests are:

| Test | Checks |
|---|---|
| `test_calibration_learns_the_person` | Calibration recovers open/closed EAR, pitch sign, blink rate, a sensible yawn threshold, and the pose model |
| `test_phone_glance_fools_baseline_but_not_padd` | **The core claim**: Baseline A alarms on a phone glance; PADD does not |
| `test_talking_is_not_a_yawn` | 30 s of talking produces no yawns |
| `test_drowsy_episode_raises_alarm` | Slow blinks + yawn + nod + long closure → ALARM, with no alarm before the episode |
| `test_alarm_turns_off_after_recovery` | After yawns, nods and slow blinks, the alarm turns off within seconds once you look alert |
| `test_head_slump_raises_alarm` | Head droops slowly and stays down → ALARM about 3 s later |
| `test_head_down_too_long_raises_alarm_even_if_face_is_lost` | A 10 s phone glance is fine; a 12 s head-down alarms, even after the face is lost |
| `test_microsleep_override` | A 2.5 s closure alarms 2.0–2.2 s after the eyes close |
| `test_nod_detector_separates_nod_from_look_down` | Slow droop + snap = 1 nod; fast look-down + hold = 0 |
| `test_yawn_detector` | Accepts a yawn, rejects talking |
| `test_profile_roundtrip` | Saving and loading a profile gives the same profile back |
| `test_scoring` | False alarm, grace period, recall and latency are computed correctly |

### Other files

| File | Role |
|---|---|
| `requirements.txt` | Packages for `pip install -r requirements.txt` |
| `pytest.ini` | Lets `pytest` find the `padd` package and the `tests/` folder |
| `.gitignore` | Keeps `models/`, `profiles/`, `sessions/`, `screenshots/` and videos out of Git. Faces and face profiles are personal data. |
| `padd/__init__.py` | Marks `padd` as a package; holds the version number |

---

## 6. How the decision is made

| Cue | Evidence | Weight |
|---|---|---:|
| Eyes | PERCLOS above your baseline, or a recent long closure (≥ 1 s) | 0.35 |
| Slow blinks | Blinks longer or slower to reopen than your 95th percentile (counts for ~60 s) | 0.20 |
| Yawns | Mouth above your yawn threshold for ≥ 1.2 s as one smooth hump, not the zig-zag of speech | 0.15 |
| Head nods | Slow droop ≥ 10° followed by a fast (≥ 40°/s) recovery | 0.30 |

```
score = smooth( Σ weight × reliability × cue_score )

CAUTION   score > 0.35          (back to OK below 0.25)
ALARM     score > 0.50 for 1 s  and a drowsy event in the last 6 s
          (back down below 0.40, or to CAUTION after 6 s with no new event)
OVERRIDE  eyes closed ≥ 2 s             →  ALARM immediately
          slow head droop, down ≥ 3 s   →  ALARM (head slumped)
          head down ≥ 12 s, any speed   →  ALARM (eyes off the road)
```

No single cue can reach ALARM on its own (the largest weight is 0.35), so several signs must
agree. The exception is the safety override for long eye closures.

---

## 7. Tuning

All values are in `padd/config.py`. Common adjustments:

| If... | Try |
|---|---|
| PADD alarms too easily | Raise `ALARM_ON` (e.g. 0.6) or `ALARM_HOLD` |
| PADD is too slow to alarm | Lower `ALARM_ON` (e.g. 0.45) or `OVERRIDE_CLOSURE` |
| Nods are missed | Lower `NOD_RECOVERY_VEL` (e.g. 30) or `NOD_DROOP_MIN` |
| Looking down counts as a nod | Lower `NOD_SLOW_MAX_VEL` |
| Talking counts as yawning | Lower `YAWN_MAX_OSC_RATE` or raise `YAWN_MIN_DURATION` |
| Baseline A seems unfair | Change `BASE_EAR_THRESHOLD` (0.25 is the textbook dlib value) |

After changing values, run `pytest` to check nothing broke. Re-run `evaluate` to see the effect;
it is fast because features are cached.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera 0` | Close other apps using the camera (Teams, Zoom, browser tabs). Try `--camera 1`. Check Windows **Settings → Privacy → Camera** allows desktop apps. |
| Black window or very slow start | Some webcams don't work with DirectShow; the code falls back automatically, but the first open can take a few seconds |
| `pip install mediapipe` fails | Use Python 3.10–3.12 (`py -3.10 -m venv .venv`) |
| `No profile for 'NAME'` | Run `python main.py calibrate --name NAME` first; names are case-sensitive |
| Calibration warns "face found in only …" | Improve front lighting, move closer, keep your whole face in view |
| Calibration warns "could not tell which way is 'head down'" | Repeat it and move your **head** (not just your eyes) for the up/down dots |
| Calibration warns "closed-eye phase unusable" | Repeat it and keep your eyes fully closed until the beep |
| Model download fails | Check your internet connection, or download the file at `MODEL_URL` in `config.py` manually into `models/face_landmarker.task` |
| No beep | Press **m** (it may be muted); `--no-sound` turns it off |
| Lots of alarms with glasses | Recalibrate **with** the glasses on; glare reduces EAR accuracy |

---

## 9. Limitations

- This is a research demo, **not a certified safety device**.
- Ordinary webcams struggle in the dark (real cars use infrared cameras). Sunglasses hide the eyes.
- Acted drowsiness is not real drowsiness; validate on UTA-RLDD as well.
- Covered yawns (a hand over the mouth) are not detected yet.
- Privacy: everything runs locally. `sessions/` and `profiles/` are git-ignored on purpose.
  Only record people who have consented.

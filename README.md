# PADD: Personalized Adaptive Drowsiness Detector

Real-time driver fatigue and drowsiness detection system using computer vision, personalized facial calibration, and multi-cue evidence fusion.

**Contributor / Maintainer:** Praga

---

## 🚗 Overview

Traditional driver drowsiness detectors rely on fixed, one-size-fits-all geometric thresholds (such as Eye Aspect Ratio $\text{EAR} < 0.22$). While simple, these rules suffer from severe false alarms in everyday driving scenarios:
- Looking down at a dashboard navigation display or phone narrows the eyes naturally, falsely triggering alarms.
- Talking or singing causes large mouth movements, falsely triggering yawn detectors.
- Natural eye size varies widely across people — a person with naturally narrow eyes may trigger perpetual alerts.

**PADD (Personalized Adaptive Drowsiness Detector)** resolves this with a three-layer architecture:
1. **Personal Calibration:** Learns each driver's individual eye open/closed scale, blink dynamics, resting posture, and yawn width in a ~90-second guided session.
2. **Pose-Aware Normalization:** Fits a polynomial model relating eye aspect ratio to head pitch and yaw, preventing false alarms when the driver tilts their head downward.
3. **Multi-Cue Evidence Fusion:** Integrates four distinct drowsiness indicators (PERCLOS, slow blinks, yawning, head nods) with confidence weighting and leaky-integrator hysteresis.

---

## 🔬 Core Features & Algorithms

### 1. 9-Phase Guided Personal Calibration (`padd/calibration.py`)
- Learns baseline EAR at rest (55th percentile) and Median Absolute Deviation (MAD).
- Estimates head pitch neutral angle and coordinate direction.
- Calibrates 95th/99th percentile blink duration and reopen latency.
- Determines individualized yawn threshold positioned safely above animated speech levels.
- Fits a 4-parameter ridge regression pose model to predict expected EAR at arbitrary pitch/yaw angles.

### 2. Multi-Indicator Drowsiness Cues (`padd/detectors.py`, `padd/fusion.py`)
- **Eye Closure / PERCLOS (35% weight):** Tracks percentage of eye closure over a sliding 60-second window, plus decaying evidence for sustained closures.
- **Slow Blinks (20% weight):** Evaluates blink duration against the user's calibrated 95th percentile, penalizing abnormally prolonged closures or sluggish reopenings.
- **Yawn Detection (15% weight):** MAR threshold with speech filtering (counts direction reversal rate to ignore high-frequency oscillations from talking).
- **Head Nodding (30% weight):** Differentiates drowsy nods (slow downward droop $<20^\circ/\text{s}$ followed by a rapid recovery jerk $>40^\circ/\text{s}$) from deliberate look-downs.
- **Microsleep Safety Override:** Bypasses gradual score accumulation and immediately fires an `ALARM` if eyes remain closed continuously for $\ge 2.0\text{s}$.

### 3. Guarded Online Adaptation (`padd/profile.py`)
- Continuously adapts to gradual lighting changes or seat adjustments using an exponential moving average.
- Only adapts when the driver is confirmed to be alert (drowsiness score $<0.15$, eyes open, head near level) to prevent learning a drowsy state as normal.

---

## 📦 Installation

Ensure you have Python 3.10 or higher installed.

```bash
# Clone or navigate to the project directory
cd Drowsiness-detection-main

# (Optional) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `mediapipe >= 0.10.14` — 478-point 3D facial landmark mesh & pose matrix
- `opencv-python >= 4.8` — video stream capture, UI overlays, and rendering
- `numpy >= 1.24` & `scipy >= 1.10` — statistical fitting and signal processing
- `matplotlib >= 3.7` — offline evaluation plotting
- `pytest >= 7.0` — unit testing suite

---

## 🚀 Usage Guide

All operations are run through the main command-line interface:

### 1. Verify Pipeline (No Camera Required)
Run the unit test suite on synthetic human data to ensure all detection and calibration math is working:
```bash
python -m pytest -v
```

### 2. Calibrate Profile (~90 Seconds)
Follow the on-screen green dots, look down, speak, close eyes, and yawn when prompted:
```bash
python main.py calibrate --name praga
```
*Creates `profiles/praga.json` containing your personalized model coefficients.*

### 3. Real-Time Detection
Run live drowsiness monitoring with visual and audio alerts:
```bash
python main.py run --name praga
```
*Keyboard shortcuts:*
- `q` or `ESC` — Quit
- `m` — Mute/unmute alarm beeps
- `s` — Save screenshot to `screenshots/`

### 4. Split-Screen Comparison (Baseline vs PADD)
Visually compare the standard fixed-threshold rule (left) against personalized PADD (right):
```bash
python main.py compare --name praga
```
*Notice how looking down at your phone triggers false alarms on the left while PADD stays green.*

### 5. Record a Scripted Test Session
Records raw webcam video along with synchronized ground-truth labels for evaluation:
```bash
python main.py record --name praga
```

### 6. Offline Evaluation & Benchmarking
Replay recorded sessions through multiple systems and generate analytical charts:
```bash
python main.py evaluate --sessions sessions --out results
```

---

## 📊 Experimental Results

Empirical evaluation against standard benchmarks across recorded test drives:

| System | False Alarms / Hr | Phone False Alarms | Recall | Median Latency |
|:---|:---:|:---:|:---:|:---:|
| **Fixed EAR Rule (Baseline)** | 94.5 | 4 | 100% | 5.2s |
| **MediaPipe Blendshapes** | 0.0 | 0 | 100% | 6.6s |
| **Personal Threshold (No Pose)** | 27.0 | 1 | 100% | 2.7s |
| **PADD (Full System)** | **13.5** | **0** | **100%** | **6.7s** |

- **85% reduction** in false alarm rate compared to generic fixed thresholding.
- **Zero false alarms** during glances down at mobile devices or dashboard displays.
- **100% recall** maintained across all acted drowsiness episodes.

---

## 📁 Project Structure

```
├── padd/
│   ├── __init__.py      # Package definition & author metadata
│   ├── config.py        # Centralized tunable constants, weights & timings
│   ├── system.py        # Abstract DrowsinessSystem interface & Status dataclass
│   ├── features.py      # MediaPipe landmark extraction (EAR, MAR, 3D head pose)
│   ├── landmarks.py     # Camera/video source handling & MediaPipe wrapper
│   ├── detectors.py     # State machines: BlinkDetector, PerclosTracker, YawnDetector, NodDetector
│   ├── profile.py       # UserProfile dataclass & PersonalModel runtime transformations
│   ├── baselines.py     # Baseline implementations for benchmarking
│   ├── calibration.py   # 9-phase guided calibration & profile fitting logic
│   ├── fusion.py        # Multi-cue evidence fusion, leaky integrator, & hysteresis
│   ├── record.py        # Synchronized video & label session recorder
│   ├── evaluate.py      # Offline playback, metric scoring, and report generation
│   └── ui.py            # OpenCV HUD overlays, cue progress bars, & beeper
├── tests/
│   └── test_padd.py     # Synthetic driver test cases
├── main.py              # CLI entrypoint with subcommands
├── requirements.txt     # Python package requirements
└── README.md            # Comprehensive project documentation
```

---

## 👤 Author & Contributor
Developed and enhanced by **Praga**.

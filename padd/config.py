"""All tunable numbers live here, so experiments never need code changes.

This is the central configuration for every threshold, weight, and timing
constant in the PADD system. Changing values here affects calibration,
detection, fusion, and the UI without touching any logic code.

Units: seconds for time, degrees for angles. "Openness" is the personal,
pose-normalized eye scale: 1.0 = your normal open eye at the current head
angle, 0.0 = your closed eye (both learned during calibration).
"""

# --------------------------------------------------------------------------
# MediaPipe model
# --------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "models/face_landmarker.task"

# Landmark indices (MediaPipe Face Mesh, 478-point topology), ordered p1..p6
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_VERTICAL = ((81, 178), (13, 14), (311, 402))
MOUTH_CORNERS = (78, 308)

# --------------------------------------------------------------------------
# Baseline A: the classic one-size-fits-all rule
# 0.25 is the dlib-era number; MediaPipe landmarks give slightly lower EAR
# values, so 0.22 is the fair equivalent.
# --------------------------------------------------------------------------
BASE_EAR_THRESHOLD = 0.22
BASE_EYE_CLOSED_SECONDS = 1.6  # = 48 frames at 30 fps
BASE_MAR_THRESHOLD = 0.5
BASE_YAWN_SECONDS = 1.0
BASE_YAWN_ALARM_HOLD = 3.0

# Blendshape baseline (MediaPipe's own eyeBlink / jawOpen scores)
BS_BLINK_THRESHOLD = 0.5
BS_JAW_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# Eye openness thresholds (personal scale)
# --------------------------------------------------------------------------
OPEN_CLOSE_T = 0.35  # below this: eye counts as closed
OPEN_REOPEN_T = 0.50  # above this: eye counts as reopened (hysteresis)
OPEN_FULL_T = 0.80  # reopening finishes when openness reaches this
PERCLOS_T = 0.20  # PERCLOS counts frames with eyes >= 80% closed
PERCLOS_WINDOW = 60.0

# Blinks
MIN_SLOW_BLINK_DURATION = 0.30  # floor for the personal slow-blink threshold
MIN_SLOW_REOPEN = 0.12  # floor for the personal slow-reopen threshold
LONG_CLOSURE_MIN = 1.0  # closures at least this long are "long closures"
OVERRIDE_CLOSURE = 2.0  # safety override: alarm regardless of fusion
REOPEN_TIMEOUT = 1.0
DEFAULT_BLINK_DUR_P95 = 0.40
DEFAULT_BLINK_DUR_P99 = 0.60
DEFAULT_REOPEN_P95 = 0.15

# Yawns
YAWN_MIN_DURATION = 1.2
YAWN_END_RATIO = 0.85  # yawn ends when MAR < threshold * this
YAWN_MAX_OSC_RATE = 2.0  # direction changes / s above this = talking
YAWN_OSC_AMPLITUDE = 0.06
YAWN_MIN_THRESHOLD = 0.25
DEFAULT_YAWN_THRESHOLD = 0.45

# Head nods (relative pitch, positive = head down)
NOD_LEVEL = 3.0  # below this the head counts as level
NOD_DROOP_MIN = 10.0  # head must drop at least this far
NOD_SLOW_MAX_VEL = 20.0  # avg deg/s while dropping; faster = voluntary look-down
NOD_RECOVERY_VEL = 40.0  # deg/s upward snap that completes a nod
HEAD_SLUMP_TIME = 3.0  # head drooped slowly and stays down this long = slumped asleep
# Head down this long (any speed, eyes open) = eyes off the road. This is
# distraction, not drowsiness: a separate soft warning, never an ALARM.
DISTRACTION_TIME = 3.0
NOD_SMOOTH_TAU = 0.08
NOD_VEL_TAU = 0.10
DEFAULT_PITCH_SIGN = 1.0

# --------------------------------------------------------------------------
# Evidence: each event adds exp(-age / tau); the cue saturates at FULL events
# --------------------------------------------------------------------------
TAU_LONG_CLOSURE, FULL_LONG_CLOSURE = 30.0, 1.0
TAU_SLOW_BLINK, FULL_SLOW_BLINK = 60.0, 4.0
TAU_YAWN, FULL_YAWN = 180.0, 2.0
TAU_NOD, FULL_NOD = 120.0, 2.0
PERCLOS_MARGIN = 0.02  # PERCLOS above your baseline + this starts to count
PERCLOS_FULL_EXCESS = 0.15

# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
WEIGHTS = {"eye": 0.35, "blink": 0.20, "yawn": 0.15, "nod": 0.30}
FUSION_TAU = 1.0  # leaky-integrator time constant
RELIABILITY_TAU = 3.0  # smoothing of per-cue landmark confidence
CAUTION_ON, CAUTION_OFF = 0.35, 0.25
ALARM_ON, ALARM_OFF = 0.50, 0.40
ALARM_HOLD = 1.0  # score must stay above ALARM_ON this long
# Evidence fades over minutes, so a high score alone can mean "was drowsy a
# while ago". ALARM also needs a fresh drowsy event (slow blink, long closure,
# yawn, nod) in the last RECENT_WINDOW seconds, or the eyes closed right now.
# Without one, ALARM drops to CAUTION (silent) until the evidence fades.
RECENT_WINDOW = 6.0

# Confidence: frames below this are not fed to a cue's detector
MIN_CONFIDENCE = 0.4
YAW_FULL_CONF, YAW_ZERO_CONF = 25.0, 45.0
MIN_EYE_WIDTH_PX = 12.0

# Guarded online adaptation (only while confidently alert)
ADAPT_MAX_SCORE = 0.15
ADAPT_TAU_EAR = 60.0
ADAPT_TAU_PITCH = 120.0
ADAPT_MAX_EAR_FRAC = 0.30  # max drift as a fraction of (open - closed)
ADAPT_MAX_PITCH = 10.0

# Pose model extrapolation limits (relative pitch, degrees)
POSE_PITCH_RANGE = (-30.0, 40.0)

# --------------------------------------------------------------------------
# Alerts / UI
# --------------------------------------------------------------------------
BEEP_FREQ = 1200
BEEP_MS = 250
BEEP_INTERVAL = 0.8
# Distraction warning: a softer, shorter, less frequent beep
DISTRACT_BEEP_FREQ = 700
DISTRACT_BEEP_MS = 100
DISTRACT_BEEP_INTERVAL = 1.5

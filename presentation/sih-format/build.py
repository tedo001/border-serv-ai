"""Fill the official SIH 2026 idea template for PS 26187 (IBVAP)."""

from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from pptx.dml.color import RGBColor

from lib import (
    AMBER, BLUE, GREEN, GREY, HEAD, INK, NAVY, RED, STEEL, TINT, TINT_G,
    TINT_R, TINT_W, WHITE, arrow, card, chip, panel, shape_by_id, textbox, write,
)

#: Caption grey on the dark figures band.
MUTED = RGBColor(0x9F, 0xB6, 0xCC)

PS_ID = "26187"
PS_TITLE = ("AI-Based Intelligent Video Analytics Platform for Border "
            "Surveillance using existing CCTV Infrastructure")
IDEA = "IBVAP — Intelligent Border Video Analytics Platform"
TEAM = "IBVAP"

prs = Presentation("template.pptx")

# -- structural work first: the template says six slides including the title,
#    and its own last slide is the instruction sheet it tells you to remove.
slide_ids = prs.slides._sldIdLst
slide_ids.remove(list(slide_ids)[6])

s1, s2, s3, s4, s5, s6 = list(prs.slides)


def set_team(slide):
    """The oval top-left carries the team mark on every content slide."""
    for shape in slide.shapes:
        if shape.has_text_frame and "your team name" in shape.text_frame.text.lower():
            tf = shape.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            write(tf, [{"text": TEAM, "size": 13, "bold": True, "color": NAVY,
                        "font": HEAD, "align": PP_ALIGN.CENTER}])


for slide in (s2, s3, s4, s5, s6):
    set_team(slide)


# ========================================================================== #
# 1 — TITLE PAGE
# ========================================================================== #
facts = shape_by_id(s1, 10)
facts.width = Inches(6.55)
facts.top = Inches(1.95)
facts.height = Inches(5.30)
write(facts.text_frame, [
    {"text": f"Problem Statement ID – {PS_ID}", "size": 15, "bold": True,
     "color": INK, "space_after": 9},
    {"text": f"Problem Statement Title – {PS_TITLE}", "size": 15, "bold": True,
     "color": INK, "space_after": 9},
    {"text": "Theme – Smart Automation", "size": 15, "bold": True, "color": INK,
     "space_after": 9},
    {"text": "PS Category – Software", "size": 15, "bold": True, "color": INK,
     "space_after": 9},
    {"text": "Organisation – Ministry of Home Affairs", "size": 15, "bold": True,
     "color": INK, "space_after": 9},
    {"text": "Department – Sashastra Seema Bal (SSB), Police II Division",
     "size": 15, "bold": True, "color": INK, "space_after": 9},
    {"text": "Team ID – ______________________", "size": 15, "bold": True,
     "color": INK, "space_after": 9},
    {"text": "Team Name (Registered on portal) – ______________________",
     "size": 15, "bold": True, "color": INK, "space_after": 0},
])

# The idea's own name, under the template's TITLE PAGE label.
title_ph = shape_by_id(s1, 4)
write(title_ph.text_frame, [
    {"text": "TITLE PAGE", "size": 28, "bold": True, "color": INK,
     "font": "Times New Roman", "align": PP_ALIGN.CENTER, "space_after": 4},
    {"text": IDEA, "size": 14, "bold": True, "color": BLUE, "font": HEAD,
     "align": PP_ALIGN.CENTER},
])


# Three plain claims under the facts, so the lower half of the page carries
# the shape of the idea rather than white space.
x = 0.45
for label, colour in (
    (["EXISTING CCTV", "no new sensors at the post"], NAVY),
    (["RUNS OFFLINE", "the alert path stays inside"], STEEL),
    (["SIGNED EVIDENCE", "hash-chained, day by day"], GREEN),
):
    chip(s1, x, 5.72, 1.80, 0.58, label, colour, size=9)
    x += 1.93

s1.notes_slide.notes_text_frame.text = (
    "Team ID and team name are filled from the SIH portal registration."
)


# ========================================================================== #
# 2 — IDEA TITLE
# ========================================================================== #
write(shape_by_id(s2, 15361).text_frame, [
    {"text": IDEA, "size": 26, "bold": True, "color": INK,
     "font": "Times New Roman", "align": PP_ALIGN.CENTER},
])

body = shape_by_id(s2, 15362)
body.left, body.top = Inches(0.45), Inches(1.20)
body.width, body.height = Inches(12.45), Inches(0.30)
write(body.text_frame, [
    {"text": "Proposed Solution (Describe your Idea/Solution/Prototype)",
     "size": 15, "bold": True, "color": BLUE, "font": HEAD},
])

card(s2, 0.45, 1.62, 3.98, 2.35, "DETAILED EXPLANATION", [
    "Software-only analytics on the CCTV a post already has — no new cameras.",
    "One edge node per BOP: detect → classify → track → recognise → rules → alert.",
    "Operators draw fences and zones on a live frame; no config files at the post.",
    "Alerts reach the control room and the sector C2 over a signed webhook.",
], NAVY, TINT)

card(s2, 4.68, 1.62, 3.98, 2.35, "HOW IT ADDRESSES THE PROBLEM", [
    "Every camera watched all night to one standard — attention never decays.",
    "Class-aware alerts naming the rule and zone, not bare motion.",
    "Face and plate reading in software — no FRS or ANPR appliance per post.",
    "Runs offline: the alert path never leaves the post.",
], STEEL, TINT)

card(s2, 8.91, 1.62, 3.98, 2.35, "INNOVATION AND UNIQUENESS", [
    "Livestock suppressed by class, not threshold — the dominant false alarm.",
    "A plate is decided by weighted vote across the vehicle's passage, not one frame.",
    "Degrades loudly: a node that cannot detect properly says so.",
    "Evidence hash-chained per day — alteration and deletion both detectable.",
], AMBER, TINT_W)

# The pipeline as a strip, because the rules ask for diagrams over paragraphs.
stages = [
    (["EXISTING", "CCTV"], GREY), (["DETECT", "RT-DETR / YOLO"], NAVY),
    (["CLASSIFY", "MobileNetV3"], STEEL), (["TRACK", "ByteTrack + Kalman"], STEEL),
    (["RECOGNISE", "ANPR · Face"], AMBER), (["ANALYSE", "10 rules"], NAVY),
    (["ALERT", "+ signed evidence"], GREEN),
]
x = 0.45
for index, (label, colour) in enumerate(stages):
    chip(s2, x, 4.35, 1.55, 0.72, label, colour, size=9)
    x += 1.55
    if index < len(stages) - 1:
        arrow(s2, x + 0.03, 4.63, 0.24, 0.16)
        x += 0.30

textbox(s2, 0.45, 5.30, 12.45, 0.32, [
    {"text": "Same pipeline behind every console, so a threshold tuned on a "
             "laptop behaves the same way at the post.",
     "size": 11, "italic": True, "color": GREY, "align": PP_ALIGN.CENTER},
])

textbox(s2, 0.45, 5.62, 12.45, 0.26, [
    {"text": "Every capability the problem statement asks for, delivered in software",
     "size": 12, "bold": True, "color": NAVY, "font": HEAD},
])
caps = [
    "Human detection\n& tracking", "Vehicle detection\n& classification",
    "Face detection\n& recognition", "ANPR\n(Indian plates)",
    "Virtual fence\nintrusion", "Suspicious\nactivity",
    "Night-time\nmovement", "Alerts, logging\n& C2 handoff",
]
x = 0.45
for text in caps:
    head, detail = text.split("\n")
    chip(s2, x, 5.94, 1.48, 0.60, [head, detail], STEEL, size=8.5)
    x += 1.56

s2.notes_slide.notes_text_frame.text = (
    "IBVAP turns installed CCTV into an analytics network. The three cards map "
    "one-to-one onto the template's pointers; the strip is the actual pipeline."
)


# ========================================================================== #
# 3 — TECHNICAL APPROACH
# ========================================================================== #
body = shape_by_id(s3, 17410)
body.left, body.top = Inches(0.45), Inches(1.16)
body.width, body.height = Inches(12.45), Inches(0.28)
write(body.text_frame, [
    {"text": "Technologies to be used", "size": 15, "bold": True, "color": BLUE,
     "font": HEAD},
])

stack = [
    ("VISION & ML", NAVY, TINT, [
        "ONNX Runtime · PyTorch (Ultralytics) · TensorRT",
        "RT-DETR · YOLO26 · YOLOv8 / v11 · YOLOv5",
        "MobileNetV3 (ImageNet-1k) · ArcFace · CRNN + CTC",
        "ByteTrack + Kalman · SciPy Hungarian · supervision",
        "OpenCV · NumPy",
    ]),
    ("PLATFORM", STEEL, TINT, [
        "Python 3.10–3.12 · FastAPI · Uvicorn",
        "Pydantic v2 · SQLAlchemy 2.0 async · Alembic",
        "SQLite (WAL) at the edge · PostgreSQL at sector",
        "JWT · bcrypt · 4-role RBAC · HMAC webhooks",
        "structlog · Prometheus",
    ]),
    ("DELIVERY & OPS", GREEN, TINT_G, [
        "Browser console — no build step, no CDN",
        "PyQt6 desktop + live analysis consoles",
        "Docker · systemd · unprivileged",
        "GitHub Actions · pytest · ruff · bandit",
        "427 automated tests",
    ]),
]
x = 0.45
for title, colour, fill, lines in stack:
    card(s3, x, 1.52, 3.98, 1.92, title, lines, colour, fill, body_size=10)
    x += 4.23

textbox(s3, 0.45, 3.62, 12.45, 0.28, [
    {"text": "Methodology and process for implementation",
     "size": 15, "bold": True, "color": BLUE, "font": HEAD},
])

flow = [
    ("1  INGEST", ["RTSP / ONVIF, reconnect", "bounded queue, drop-oldest"], NAVY),
    ("2  DETECT", ["ONNX / Torch / TensorRT", "motion fallback if no model"], NAVY),
    ("3  CLASSIFY", ["MobileNetV3, once per", "track — not per frame"], STEEL),
    ("4  TRACK", ["ByteTrack + Kalman", "IDs hold through occlusion"], STEEL),
    ("5  RECOGNISE", ["ANPR: plate Kalman + vote", "Face: ArcFace + margin"], AMBER),
    ("6  ACT", ["rules → gate → evidence", "→ store-and-forward to C2"], GREEN),
]
x = 0.45
for index, (head, detail, colour) in enumerate(flow):
    panel(s3, x, 3.98, 1.85, 1.05, TINT)
    chip(s3, x, 3.98, 1.85, 0.30, head, colour, size=10)
    textbox(s3, x + 0.08, 4.34, 1.69, 0.66, [
        {"text": line, "size": 8.5, "color": INK, "align": PP_ALIGN.CENTER,
         "space_after": 2}
        for line in detail
    ])
    x += 1.85
    if index < len(flow) - 1:
        arrow(s3, x + 0.02, 4.43, 0.20, 0.16)
        x += 0.24

textbox(s3, 0.45, 5.20, 12.45, 0.66, [
    {"text": "Why this approach", "size": 12, "bold": True, "color": NAVY,
     "font": HEAD, "space_after": 3},
    {"text": "Runs on cameras already installed  ·  ONNX rather than a training "
             "stack: ~200 MB, not 2 GB  ·  detection is expensive and tracking is "
             "cheap, so the detector samples frames and the tracker fills the gaps  ·  "
             "classification runs once per track, which is what makes livestock "
             "suppression affordable at the edge  ·  every fall to a weaker runtime "
             "is reported, never silent.",
     "size": 10, "color": INK},
])

textbox(s3, 0.45, 5.90, 12.45, 0.26, [
    {"text": "From a laptop to a border post", "size": 12, "bold": True,
     "color": NAVY, "font": HEAD},
])
path = [
    (["TUNE", "analyst console, Torch"], STEEL),
    (["EXPORT", "ibvap models fetch → ONNX"], NAVY),
    (["SHIP", "hashed, registered artefact"], NAVY),
    (["COMMISSION", "TensorRT plan built on the node"], AMBER),
    (["RUN", "ONNX Runtime alone, offline"], GREEN),
]
x = 0.45
for index, (label, colour) in enumerate(path):
    chip(s3, x, 6.20, 2.20, 0.50, label, colour, size=8.5)
    x += 2.20
    if index < len(path) - 1:
        arrow(s3, x + 0.02, 6.38, 0.16, 0.14)
        x += 0.22

s3.notes_slide.notes_text_frame.text = (
    "Top row is the stack, bottom row the six pipeline stages in order. "
    "Both are what the repository actually contains."
)


# ========================================================================== #
# 4 — FEASIBILITY AND VIABILITY
# ========================================================================== #
body = shape_by_id(s4, 17410)
body.left, body.top = Inches(0.45), Inches(1.16)
body.width, body.height = Inches(12.45), Inches(0.28)
write(body.text_frame, [
    {"text": "Analysis of the feasibility of the idea", "size": 15, "bold": True,
     "color": BLUE, "font": HEAD},
])

feas = [
    ("TECHNICAL", NAVY, TINT, [
        "Proven open-source stack, production-grade throughout.",
        "CPU-only path: a fanless mini-PC carries several cameras.",
        "A benchmark command sizes a post before hardware is bought.",
        "427 tests on Python 3.10 / 3.11 / 3.12.",
    ]),
    ("OPERATIONAL", STEEL, TINT, [
        "No change to the camera plant — existing feeds and cabling.",
        "Operators place fences on a live frame, not in a file.",
        "One service under systemd, or one container, unprivileged.",
        "Works air-gapped; four roles with a full audit trail.",
    ]),
    ("ECONOMIC", GREEN, TINT_G, [
        "Zero new sensors — the capital cost is already sunk.",
        "No face or plate appliance licence per post.",
        "Open-source stack; no per-camera analytics fee.",
        "Savings compound: fewer false-alarm sorties.",
    ]),
]
x = 0.45
for title, colour, fill, lines in feas:
    card(s4, x, 1.52, 3.98, 1.72, title, lines, colour, fill, body_size=10)
    x += 4.23

textbox(s4, 0.45, 3.40, 6.05, 0.28, [
    {"text": "Potential challenges and risks", "size": 15, "bold": True,
     "color": BLUE, "font": HEAD},
])
textbox(s4, 6.85, 3.40, 6.05, 0.28, [
    {"text": "Strategies for overcoming these challenges", "size": 15,
     "bold": True, "color": BLUE, "font": HEAD},
])

risks = [
    ("False alarms erode trust",
     "Class-aware suppression, confirm frames and a dedup gate. Measured on the "
     "livestock drill: 27 tracks reclassified, zero false alarms — a person in "
     "the same zone still alarms."),
    ("No annotated border imagery",
     "A deterministic scenario simulator drives drills and CI; the model registry "
     "accepts a fine-tuned export the moment real data exists."),
    ("Weak hardware at remote posts",
     "A CPU fallback path, and a benchmark command that sizes the post first."),
    ("Intermittent VSAT / radio uplink",
     "A persisted outbox with capped backoff, replaying alerts in order."),
    ("Evidence challenged months later",
     "SHA-256 per artefact and a per-day hash chain; deletion is detectable too."),
]
panel(s4, 0.45, 3.76, 12.45, 2.96, TINT_R)
y = 3.96
for risk, fix in risks:
    textbox(s4, 0.62, y, 5.75, 0.44, [
        {"text": f"▸  {risk}", "size": 10.5, "bold": True, "color": RED},
    ])
    textbox(s4, 6.85, y, 5.90, 0.46, [
        {"text": fix, "size": 9.5, "color": INK},
    ])
    y += 0.56

s4.notes_slide.notes_text_frame.text = (
    "Left column names the risk, right column the mitigation already built. "
    "The livestock figure is measured on the cattle drill, not projected."
)


# ========================================================================== #
# 5 — IMPACT AND BENEFITS
# ========================================================================== #
body = shape_by_id(s5, 17410)
body.left, body.top = Inches(0.45), Inches(1.16)
body.width, body.height = Inches(12.45), Inches(0.28)
write(body.text_frame, [
    {"text": "Potential impact on the target audience", "size": 15, "bold": True,
     "color": BLUE, "font": HEAD},
])

textbox(s5, 0.45, 1.50, 12.45, 0.72, [
    {"text": "Sashastra Seema Bal sentries, post commanders and sector "
             "headquarters — the people watching the wall tonight.",
     "size": 11.5, "color": INK, "space_after": 4},
    {"text": "Every camera is watched to one standard through the whole shift; "
             "alerts arrive naming the class, the rule and the zone; plate and "
             "face checks happen at the post in seconds; sentries move from "
             "staring at a video wall to responding to something specific.",
     "size": 11.5, "color": INK},
])

textbox(s5, 0.45, 2.36, 12.45, 0.28, [
    {"text": "Benefits of the solution (social, economic, environmental)",
     "size": 15, "bold": True, "color": BLUE, "font": HEAD},
])

benefits = [
    ("SOCIAL & GOVERNANCE", STEEL, TINT, [
        "Every alert carries its evidence and audit trail, so a decision can be reviewed.",
        "Face embeddings kept only for watchlist enrolment — never for passers-by.",
        "Fewer false-alarm sorties means less risk to the responding party.",
    ]),
    ("ECONOMIC", NAVY, TINT, [
        "Capability on CCTV already paid for, cabled and powered.",
        "No FRS or ANPR appliance to buy and licence per post.",
        "Skilled work in deployment and tuning rather than in watching screens.",
    ]),
    ("ENVIRONMENTAL", GREEN, TINT_G, [
        "Reuse over replacement: no new hardware to make, ship or dispose of.",
        "One low-power edge node per post, sized to the load.",
        "Fewer vehicle sorties chasing alarms that were cattle.",
    ]),
]
x = 0.45
for title, colour, fill, lines in benefits:
    card(s5, x, 2.72, 3.98, 1.90, title, lines, colour, fill, body_size=10)
    x += 4.23

# Measured figures, so the impact claim rests on something.
panel(s5, 0.45, 4.85, 12.45, 1.35, NAVY, outline=None)
stats = [
    ("0", "NEW CAMERAS REQUIRED", AMBER),
    ("27 → 0", "LIVESTOCK TRACKS RECLASSIFIED → FALSE ALARMS", GREEN),
    ("~200 MB", "INFERENCE RUNTIME FOOTPRINT", WHITE),
    ("427", "AUTOMATED TESTS PASSING", WHITE),
]
w = 12.45 / 4
for index, (value, label, colour) in enumerate(stats):
    x = 0.45 + index * w
    textbox(s5, x, 5.02, w, 0.46, [
        {"text": value, "size": 22, "bold": True, "color": colour,
         "font": HEAD, "align": PP_ALIGN.CENTER}])
    textbox(s5, x + 0.08, 5.52, w - 0.16, 0.50, [
        {"text": label, "size": 8, "bold": True, "color": MUTED,
         "align": PP_ALIGN.CENTER}])

textbox(s5, 0.45, 6.32, 12.45, 0.30, [
    {"text": "Measured in this repository, not projected — the livestock pair "
             "comes from the cattle drill scenario.",
     "size": 9.5, "italic": True, "color": GREY, "align": PP_ALIGN.CENTER},
])

s5.notes_slide.notes_text_frame.text = (
    "The four figures at the bottom are measured, not estimated."
)


# ========================================================================== #
# 6 — RESEARCH AND REFERENCES
# ========================================================================== #
body = shape_by_id(s6, 17410)
body.left, body.top = Inches(0.45), Inches(1.16)
body.width, body.height = Inches(12.45), Inches(0.28)
write(body.text_frame, [
    {"text": "Details / Links of the reference and research work", "size": 15,
     "bold": True, "color": BLUE, "font": HEAD},
])

refs = [
    ("Sultani, Chen & Shah (2018)",
     "Real-world Anomaly Detection in Surveillance Videos — CVPR 2018",
     "Weakly supervised anomaly detection over long, untrimmed surveillance video: "
     "the setting a border camera actually produces.",
     "https://arxiv.org/abs/1801.04264"),
    ("Zhao et al. (2024)",
     "DETRs Beat YOLOs on Real-time Object Detection (RT-DETR) — CVPR 2024",
     "An end-to-end detector with no NMS pass — a real saving on an edge node "
     "carrying many cameras on few cores.",
     "https://arxiv.org/abs/2304.08069"),
    ("Zhang et al. (2022)",
     "ByteTrack: Multi-Object Tracking by Associating Every Detection Box — ECCV 2022",
     "Associating low-scoring boxes keeps an identity alive through the occlusion "
     "a fence or a post causes.",
     "https://arxiv.org/abs/2110.06864"),
    ("Howard et al. (2019)",
     "Searching for MobileNetV3 — ICCV 2019",
     "The efficient backbone behind the once-per-track refinement that tells a cow "
     "from a person cheaply enough to run at the edge.",
     "https://arxiv.org/abs/1905.02244"),
    ("Deng et al. (2019)",
     "ArcFace: Additive Angular Margin Loss for Deep Face Recognition — CVPR 2019",
     "The objective producing the 512-d embeddings compared by cosine similarity "
     "in the face-matching stage.",
     "https://arxiv.org/abs/1801.07698"),
]
panel(s6, 0.45, 1.50, 12.45, 3.35, TINT)
y = 1.62
for index, (who, what, why, link) in enumerate(refs, 1):
    chip(s6, 0.60, y + 0.02, 0.30, 0.26, str(index), AMBER, colour=NAVY, size=10)
    textbox(s6, 1.00, y, 8.30, 0.26, [
        {"text": f"{who}  —  {what}", "size": 10.5, "bold": True, "color": NAVY},
    ])
    textbox(s6, 1.00, y + 0.26, 8.30, 0.34, [
        {"text": why, "size": 9, "color": INK},
    ])
    textbox(s6, 9.45, y + 0.04, 3.35, 0.26, [
        {"text": link, "size": 9, "color": BLUE, "align": PP_ALIGN.RIGHT},
    ])
    y += 0.64

textbox(s6, 0.45, 5.00, 6.05, 0.28, [
    {"text": "What exists today", "size": 13, "bold": True, "color": RED,
     "font": HEAD},
])
panel(s6, 0.45, 5.32, 6.05, 1.45, TINT_R)
textbox(s6, 0.62, 5.44, 5.72, 1.25, [
    {"text": "▸  Manual CCTV monitoring — attention decays within the hour.",
     "size": 9.5, "color": INK, "space_after": 4},
    {"text": "▸  Motion / tripwire alarms — a cow, a branch and a man look alike.",
     "size": 9.5, "color": INK, "space_after": 4},
    {"text": "▸  Dedicated FRS and ANPR appliances — priced and cabled per post.",
     "size": 9.5, "color": INK, "space_after": 4},
    {"text": "▸  Cloud analytics — needs an uplink a border post does not have.",
     "size": 9.5, "color": INK},
])

textbox(s6, 6.85, 5.00, 6.05, 0.28, [
    {"text": "Where IBVAP stands out", "size": 13, "bold": True, "color": GREEN,
     "font": HEAD},
])
panel(s6, 6.85, 5.32, 6.05, 1.45, TINT_G)
textbox(s6, 7.02, 5.44, 5.72, 1.25, [
    {"text": "▸  Class-aware alerts — livestock suppressed by name, not threshold.",
     "size": 9.5, "color": INK, "space_after": 4},
    {"text": "▸  Face and plate reading in software, on the same node.",
     "size": 9.5, "color": INK, "space_after": 4},
    {"text": "▸  Runs at the edge, offline; an outbox handles the uplink.",
     "size": 9.5, "color": INK, "space_after": 4},
    {"text": "▸  Defensible evidence, and a node that reports what it cannot do.",
     "size": 9.5, "color": INK},
])

s6.notes_slide.notes_text_frame.text = (
    "Five papers the implementation genuinely rests on, each with the reason it "
    "is in the design. Lower half is the competitive read."
)

prs.save("IBVAP-SIH2026-PS26187.pptx")
print("saved IBVAP-SIH2026-PS26187.pptx")

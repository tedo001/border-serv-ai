const { C, HEAD, BODY, header, card, panel, rows } = require('./lib.js');

// ========================================================================== //
// 3 - TECHNICAL APPROACH
// ========================================================================== //
function slide3(p) {
  const s = p.addSlide();
  s.background = { color: C.white };
  header(s, 'T E C H N I C A L   A P P R O A C H', 3);

  const stages = [
    ['1', 'INGEST & DECODE', C.navy, [
      'Existing RTSP / ONVIF CCTV feeds',
      'TCP transport, reconnect with backoff',
      'Bounded queue, oldest frame dropped',
      'Per-camera target-FPS sampling',
      'Low-light enhancement when the frame is dark',
      'Detect every Nth frame; tracking fills the gaps',
    ]],
    ['2', 'DETECTION', C.mid, [
      'ONNX Runtime, CUDA falling back to CPU',
      'YOLO26 - end-to-end, NMS folded in',
      'RT-DETR - set prediction, no NMS pass',
      'YOLOv8 / v5 legacy exports supported',
      'MOG2 motion fallback when no artefact',
      'Class allowlist and minimum object height',
    ]],
    ['3', 'CLASSIFY & TRACK', C.steel, [
      'MobileNetV3 over ImageNet-1k',
      'Once per track, not once per frame',
      'ByteTrack with a Kalman filter',
      'Hungarian assignment, three passes',
      'Identities hold through occlusion',
      'Trail history gives speed and direction',
    ]],
    ['4', 'RECOGNISE & ANALYSE', C.amber, [
      'ANPR: locate, deskew, CRNN + CTC decode',
      'Indian number-plate grammar correction',
      'Face: ArcFace 512-d, cosine plus margin',
      'Virtual fence and zone intrusion',
      'Loitering, abandoned object, tamper',
      'Night movement in IR and low light',
    ]],
    ['5', 'ALERT & EVIDENCE', C.green, [
      'Dedup, cooldown and rate-limit gate',
      'Annotated snapshot and clip written',
      'SHA-256 plus a per-day hash chain',
      'HMAC-signed webhook to the C2 system',
      'Store-and-forward outbox for outages',
      'WebSocket push to both consoles',
    ]],
  ];

  const cw = 3.6, gap = 0.25;
  stages.forEach(([num, name, colour, items], i) => {
    const x = 0.5 + i * (cw + gap);
    card(s, { x, y: 1.5, w: cw, h: 5.15, title: name, color: colour,
      titleColor: colour === C.amber ? C.navy : C.white, titleSize: 14,
      fill: 'F7FAFC', lineColor: C.line });
    s.addShape('ellipse', { x: x + 0.14, y: 1.6, w: 0.32, h: 0.32,
      fill: { color: C.white } });
    s.addText(num, { x: x + 0.14, y: 1.6, w: 0.32, h: 0.32, fontSize: 13, bold: true,
      color: colour === C.amber ? C.navy : colour, fontFace: HEAD,
      align: 'center', valign: 'middle', margin: 0, isTextBox: true });

    let y = 2.22;
    items.forEach((it) => {
      s.addShape('rect', { x: x + 0.22, y: y + 0.12, w: 0.1, h: 0.1, fill: { color: colour } });
      s.addText(it, {
        x: x + 0.46, y, w: cw - 0.68, h: 0.62, fontSize: 12.5, color: C.text,
        fontFace: BODY, valign: 'top', lineSpacingMultiple: 1.0, margin: 0, isTextBox: true,
      });
      y += 0.72;
    });
  });

  // -- technology stack
  card(s, { x: 0.5, y: 6.9, w: 13.0, h: 3.5, title: 'TECHNOLOGY STACK',
    color: C.navy, fill: 'F2F6FA', lineColor: C.line });
  const stack = [
    ['VISION & MODELS', [
      ['Inference', 'ONNX Runtime'],
      ['Detectors', 'YOLO26 · RT-DETR · YOLOv8'],
      ['Classifier', 'MobileNetV3 (ImageNet-1k)'],
      ['Tracker', 'ByteTrack + Kalman'],
      ['ANPR', 'CRNN + CTC decode'],
      ['Face', 'ArcFace 512-d embeddings'],
    ]],
    ['PLATFORM', [
      ['Language', 'Python 3.11+'],
      ['API', 'FastAPI + Uvicorn'],
      ['Data', 'SQLAlchemy 2.0 async'],
      ['Config', 'Pydantic v2'],
      ['Vision', 'OpenCV · NumPy · SciPy'],
      ['Store', 'SQLite (WAL) / PostgreSQL'],
    ]],
    ['DELIVERY & OPERATIONS', [
      ['Consoles', 'Browser · PyQt6 · Analyst'],
      ['Packaging', 'Docker · systemd'],
      ['Telemetry', 'Prometheus · structlog'],
      ['CI', 'GitHub Actions · pytest · ruff'],
      ['Security', 'JWT · bcrypt · 4-role RBAC'],
      ['Integration', 'HMAC-signed webhook to C2'],
    ]],
  ];
  stack.forEach(([groupTitle, pairs], i) => {
    const x = 0.78 + i * 4.28;
    s.addText(groupTitle, {
      x, y: 7.6, w: 4.0, h: 0.3, fontSize: 13, bold: true, color: C.amber,
      fontFace: HEAD, valign: 'middle', margin: 0, isTextBox: true,
    });
    let y = 7.98;
    pairs.forEach(([k, v]) => {
      s.addText([
        { text: k, options: { bold: true, color: C.navy } },
        { text: '   ' + v, options: { color: C.text } },
      ], {
        x, y, w: 4.0, h: 0.34, fontSize: 12.5, fontFace: BODY,
        valign: 'middle', margin: 0, isTextBox: true,
      });
      y += 0.36;
    });
  });

  // -- why this approach
  card(s, { x: 13.85, y: 6.9, w: 5.65, h: 3.5, title: 'WHY THIS APPROACH?',
    color: C.amber, titleColor: C.navy, fill: 'FDF6EC', lineColor: 'EBD3AC' });
  rows(s, {
    x: 14.1, y: 7.6, w: 5.15, rowH: 0.48, gap: 0.05, fontSize: 12.5, dotColor: C.amber,
    items: [
      { text: 'Runs on the cameras already installed - nothing to replace.' },
      { text: 'ONNX, not a training stack: about 200 MB rather than 2 GB.' },
      { text: 'Degrades loudly. A node that cannot detect properly says so.' },
      { text: 'One pipeline behind every console, so none of them drift.' },
      { text: 'Measurable: 326 automated tests and a benchmark command.' },
    ],
  });

  s.addNotes('Five stages, left to right, each one a real module in the codebase. The stack is deliberately boring and open-source; the interesting engineering is in the fallbacks, not the dependency list.');
}

// ========================================================================== //
// 4 - FEASIBILITY AND VIABILITY
// ========================================================================== //
function slide4(p) {
  const s = p.addSlide();
  s.background = { color: C.white };
  header(s, 'F E A S I B I L I T Y   &   V I A B I L I T Y', 4);

  const cw = 6.13, gapx = 0.31;
  const col = (i) => 0.5 + i * (cw + gapx);

  const blocks = [
    { x: col(0), y: 1.5, h: 4.35, title: 'TECHNICAL FEASIBILITY', color: C.navy,
      fill: 'F2F6FA', line: C.line, items: [
        { label: 'Proven stack.', text: ' ONNX Runtime, OpenCV, FastAPI and SQLite - production-grade and open source.', inline: true },
        { label: 'CPU-only path.', text: ' A fanless mini-PC carries several cameras; a GPU is an upgrade, not a requirement.', inline: true },
        { label: 'Sized before it is bought.', text: ' A benchmark command measures the throughput of the actual hardware.', inline: true },
        { label: 'Verified.', text: ' 326 automated tests across unit and integration, on Python 3.10, 3.11 and 3.12.', inline: true },
        { label: 'Degradation ladder.', text: ' Missing artefacts fall back and are reported, never hidden.', inline: true },
      ] },
    { x: col(1), y: 1.5, h: 4.35, title: 'OPERATIONAL FEASIBILITY', color: C.mid,
      fill: 'F2F6FA', line: C.line, items: [
        { label: 'No change to the camera plant.', text: ' Existing feeds, existing cabling, existing power.', inline: true },
        { label: 'Operators draw their own geometry.', text: ' Fences and zones are placed on a live frame, not in a config file.', inline: true },
        { label: 'One service.', text: ' A systemd unit or a single container, running unprivileged.', inline: true },
        { label: 'Works air-gapped.', text: ' No cloud call anywhere in the alert path.', inline: true },
        { label: 'Accountable.', text: ' Four roles - viewer, operator, supervisor, admin - with a full audit trail.', inline: true },
      ] },
    { x: col(2), y: 1.5, h: 4.35, title: 'ECONOMIC FEASIBILITY', color: C.steel,
      fill: 'F2F6FA', line: C.line, items: [
        { label: 'Zero new sensors.', text: ' The capital cost is already sunk in the installed CCTV.', inline: true },
        { label: 'No appliance licence.', text: ' Face matching and plate reading run in software on the same node.', inline: true },
        { label: 'No per-camera analytics fee.', text: ' The stack is open source end to end.', inline: true },
        { label: 'One box per post.', text: ' The sector tier reuses commodity servers already in place.', inline: true },
        { label: 'Savings compound.', text: ' Fewer false-alarm sorties and far less manual review time.', inline: true },
      ] },
    { x: col(0), y: 6.1, h: 4.3, title: 'OUR STRATEGY', color: C.amber, titleColor: C.navy,
      fill: 'FDF6EC', line: 'EBD3AC', numbered: true, items: [
        { label: 'Ingest what exists.', text: ' Connect the installed CCTV rather than replacing it.', inline: true },
        { label: 'Detect and track.', text: ' One detector, stable identities, class-aware from the first frame.', inline: true },
        { label: 'Refine the class.', text: ' A second pass that separates cattle from people before any rule runs.', inline: true },
        { label: 'Apply the operator’s geometry.', text: ' Their fences, their zones, their thresholds.', inline: true },
        { label: 'Gate, sign, forward.', text: ' Deduplicate, write evidence, and survive the uplink.', inline: true },
      ] },
    { x: col(1), y: 6.1, h: 4.3, title: 'RISKS & MITIGATION', color: C.red,
      fill: 'FDF1EF', line: 'F0C9C2', items: [
        { label: 'False alarms erode trust.', text: ' Class-aware suppression, confirm frames and a dedup gate. On the livestock drill: 27 tracks reclassified, zero false alarms.', h: 0.75, inline: true },
        { label: 'No annotated border imagery.', text: ' A deterministic scenario simulator drives drills and CI; the registry takes a fine-tuned export when real data exists.', h: 0.75, inline: true },
        { label: 'Weak hardware at remote posts.', text: ' A CPU fallback path, and a benchmark that sizes the post first.', h: 0.60, inline: true },
        { label: 'Intermittent uplink.', text: ' A persisted outbox with capped backoff, replaying in order.', h: 0.60, inline: true },
        { label: 'Evidence challenged later.', text: ' SHA-256 per artefact and a per-day hash chain.', h: 0.5, inline: true },
      ] },
    { x: col(2), y: 6.1, h: 4.3, title: 'VIABILITY & SCALE', color: C.green,
      fill: 'EFF8F3', line: 'C4E2D2', items: [
        { label: 'Edge tier at the post.', text: ' SQLite in WAL mode, no external dependency, survives isolation indefinitely.', inline: true },
        { label: 'Central tier at sector.', text: ' PostgreSQL aggregating many posts into one picture.', inline: true },
        { label: 'Adding a post adds a node.', text: ' No re-architecture, no shared bottleneck.', inline: true },
        { label: 'Detectors are swappable.', text: ' A versioned, checksummed registry: a better model is a file, not a rebuild.', inline: true },
        { label: 'One platform, many settings.', text: ' Fence lines, gates, border roads and check posts.', inline: true },
      ] },
  ];

  blocks.forEach((b) => {
    card(s, { x: b.x, y: b.y, w: cw, h: b.h, title: b.title, color: b.color,
      titleColor: b.titleColor || C.white, fill: b.fill, lineColor: b.line });
    rows(s, {
      x: b.x + 0.26, y: b.y + 0.72, w: cw - 0.52,
      rowH: (b.h - 0.95) / b.items.length - 0.08, gap: 0.08,
      fontSize: 12.5, dotColor: b.color, items: b.items,
    });
  });

  s.addNotes('Feasibility is argued from what the platform already does rather than from intent. The livestock number in Risks is a measured result from the cattle drill scenario, not a projection.');
}

module.exports = { slide3, slide4 };

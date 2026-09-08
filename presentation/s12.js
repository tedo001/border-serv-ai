const { C, HEAD, BODY, PS_ID, PS_TITLE, header, footer, card, rows, outline } = require('./lib.js');

// ========================================================================== //
// 1 - TITLE
// ========================================================================== //
function slide1(p) {
  const s = p.addSlide();
  s.background = { color: C.navy };

  // -- night scene: the subject of the whole platform, drawn rather than stock
  const PX = 11.55, PY = 1.62, PW = 7.95, PH = 8.4;
  s.addShape('rect', { x: PX, y: PY, w: PW, h: PH, fill: { color: '091E36' },
    line: { color: '1D4470', width: 1.25 } });
  s.addShape('rect', { x: PX, y: PY + 4.15, w: PW, h: PH - 4.15, fill: { color: '1B3F63' } });
  s.addShape('rect', { x: PX, y: PY + 4.11, w: PW, h: 0.05, fill: { color: '4E82B8' } });

  // fence: posts and two wires
  for (let i = 0; i < 11; i++) {
    s.addShape('rect', { x: PX + 0.3 + i * 0.73, y: PY + 3.3, w: 0.055, h: 1.55,
      fill: { color: '35618E' } });
  }
  s.addShape('rect', { x: PX + 0.3, y: PY + 3.62, w: 7.35, h: 0.035, fill: { color: '35618E' } });
  s.addShape('rect', { x: PX + 0.3, y: PY + 4.35, w: 7.35, h: 0.035, fill: { color: '35618E' } });

  // the operator-drawn virtual fence
  s.addShape('rect', { x: PX + 3.05, y: PY + 0.45, w: 0.05, h: PH - 1.1, fill: { color: C.red } });
  s.addText('VIRTUAL FENCE', {
    x: PX + 3.22, y: PY + 7.55, w: 2.4, h: 0.3, fontSize: 11, bold: true, color: 'F09A8E',
    fontFace: BODY, valign: 'middle', margin: 0, isTextBox: true,
  });

  // detections, in the bracket motif carried through the deck
  outline(s, PX + 0.95, PY + 4.55, 1.55, 2.3, C.green, 0.04);
  s.addShape('rect', { x: PX + 0.95, y: PY + 4.17, w: 1.55, h: 0.38, fill: { color: C.green } });
  s.addText('PERSON 0.91', {
    x: PX + 1.03, y: PY + 4.17, w: 1.45, h: 0.38, fontSize: 10, bold: true, color: C.white,
    fontFace: BODY, valign: 'middle', margin: 0, isTextBox: true,
  });

  outline(s, PX + 4.4, PY + 5.4, 2.65, 1.7, C.amber, 0.04);
  s.addShape('rect', { x: PX + 4.4, y: PY + 5.02, w: 1.7, h: 0.38, fill: { color: C.amber } });
  s.addText('VEHICLE 0.87', {
    x: PX + 4.48, y: PY + 5.02, w: 1.6, h: 0.38, fontSize: 10, bold: true, color: C.navy,
    fontFace: BODY, valign: 'middle', margin: 0, isTextBox: true,
  });

  // camera chip and mode chip
  s.addShape('roundRect', { x: PX + 5.25, y: PY + 0.35, w: 2.35, h: 0.5, rectRadius: 0.08,
    fill: { color: C.white } });
  s.addText('EXISTING CCTV', {
    x: PX + 5.25, y: PY + 0.35, w: 2.35, h: 0.5, fontSize: 12, bold: true, color: C.navy,
    fontFace: HEAD, align: 'center', valign: 'middle', margin: 0, isTextBox: true,
  });
  s.addShape('roundRect', { x: PX + 0.35, y: PY + 0.35, w: 1.55, h: 0.5, rectRadius: 0.08,
    fill: { color: '1D4470' } });
  s.addText('IR  NIGHT', {
    x: PX + 0.35, y: PY + 0.35, w: 1.55, h: 0.5, fontSize: 12, bold: true, color: '9FD8FF',
    fontFace: HEAD, align: 'center', valign: 'middle', margin: 0, isTextBox: true,
  });

  // -- headline
  s.addText('SMART INDIA HACKATHON 2026', {
    x: 0.7, y: 0.42, w: 18.6, h: 0.95, fontSize: 48, bold: true, color: C.white,
    fontFace: HEAD, align: 'center', valign: 'middle', charSpacing: 1, margin: 0, isTextBox: true,
  });

  s.addText('I B V A P', {
    x: 0.85, y: 1.72, w: 10.2, h: 1.1, fontSize: 60, bold: true, color: C.amber,
    fontFace: HEAD, charSpacing: 4, valign: 'middle', margin: 0, isTextBox: true,
  });
  s.addText('Intelligent Border Video Analytics Platform', {
    x: 0.9, y: 2.82, w: 10.2, h: 0.4, fontSize: 18, bold: true, color: C.white,
    fontFace: BODY, valign: 'middle', margin: 0, isTextBox: true,
  });
  s.addText('Watch the line  ·  Hold the line', {
    x: 0.9, y: 3.22, w: 10.2, h: 0.38, fontSize: 15, color: '9FB6CC',
    fontFace: BODY, italic: true, valign: 'middle', margin: 0, isTextBox: true,
  });

  const facts = [
    ['Problem Statement ID', PS_ID, false],
    ['Problem Statement Title', PS_TITLE, true],
    ['Theme', 'Smart Automation', false],
    ['PS Category', 'Software', false],
    ['Organisation', 'Ministry of Home Affairs  ·  Sashastra Seema Bal (SSB), Police II Division', true],
    ['Team ID', '', false],
    ['Team Name', '', false],
  ];
  let fy = 3.75;
  facts.forEach(([k, v, tall]) => {
    const h = tall ? 0.94 : 0.56;
    s.addShape('rect', { x: 0.9, y: fy + 0.15, w: 0.11, h: 0.11, fill: { color: C.amber } });
    const runs = [{ text: k + ' : ', options: { bold: true, color: C.white } }];
    if (v) runs.push({ text: v, options: { color: '8FE0B4', bold: true } });
    else runs.push({ text: '________________', options: { color: '5D7691' } });
    s.addText(runs, {
      x: 1.22, y: fy, w: 9.85, h, fontSize: 16, fontFace: BODY,
      valign: 'top', lineSpacingMultiple: 1.06, margin: 0, isTextBox: true,
    });
    fy += h + 0.06;
  });

  const caps = [
    ['DETECT & TRACK', 'People · vehicles · animals, with stable identities'],
    ['RECOGNISE', 'Number plates and watchlist faces, in software'],
    ['GUARD THE LINE', 'Virtual fences, intrusion zones and night movement'],
  ];
  caps.forEach(([t, d], i) => {
    const x = 0.9 + i * 3.45;
    s.addShape('roundRect', { x, y: 9.02, w: 3.2, h: 1.28, rectRadius: 0.1,
      fill: { color: '11304F' }, line: { color: '25547F', width: 1 } });
    s.addText(t, { x: x + 0.2, y: 9.16, w: 2.8, h: 0.32, fontSize: 13, bold: true,
      color: C.amber, fontFace: HEAD, valign: 'middle', margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.2, y: 9.5, w: 2.8, h: 0.68, fontSize: 12, color: 'BCD0E3',
      fontFace: BODY, valign: 'top', lineSpacingMultiple: 1.0, margin: 0, isTextBox: true });
  });

  footer(s, 1);
  s.addNotes('SIH 2026 idea submission for PS 26187 (Ministry of Home Affairs / Sashastra Seema Bal). IBVAP turns the CCTV already installed along the border into an analytics network: detection, tracking, ANPR, face matching, virtual fences and night movement, running on an edge node that keeps working when the uplink drops. Fill in Team ID and Team Name before submitting.');
}

// ========================================================================== //
// 2 - IDEA & SOLUTION
// ========================================================================== //
function slide2(p) {
  const s = p.addSlide();
  s.background = { color: C.white };
  header(s, 'I D E A   &   S O L U T I O N', 2);

  card(s, { x: 0.5, y: 1.5, w: 5.6, h: 8.9, title: 'THE PROBLEM TODAY',
    color: C.red, fill: 'FDF1EF', lineColor: 'F0C9C2' });
  rows(s, {
    x: 0.78, y: 2.24, w: 5.05, rowH: 1.40, gap: 0.20, fontSize: 14, dotColor: C.red,
    items: [
      { label: 'Cameras record, nobody watches.', text: 'Can one sentry hold attention on forty feeds through a twelve-hour night shift?' },
      { label: 'Alarm fatigue kills trust.', text: 'Cattle, foliage and rain trigger alerts until the operator stops believing any of them.' },
      { label: 'No face or plate hardware at a post.', text: 'Must every BOP buy a dedicated appliance to read a number plate or check a face?' },
      { label: 'Darkness is when it matters.', text: 'Infiltration happens at night, in IR, in haze - exactly where a motion alarm fails.' },
      { label: 'The uplink drops.', text: 'Do alerts survive a VSAT outage? Is the evidence still defensible weeks later?' },
    ],
  });

  card(s, { x: 6.4, y: 1.5, w: 6.6, h: 8.9, title: 'IBVAP  -  EDGE ANALYTICS NODE',
    color: C.navy, fill: 'F2F6FA', lineColor: C.line });
  const stages = [
    ['1   INGEST', 'Existing RTSP / ONVIF CCTV, video file or drill simulator'],
    ['2   DETECT', 'ONNX Runtime  ·  YOLO26  ·  RT-DETR  ·  YOLOv8'],
    ['3   CLASSIFY & TRACK', 'MobileNetV3 / ImageNet refinement  +  ByteTrack identities'],
    ['4   RECOGNISE', 'ANPR (CRNN + CTC)  ·  Face matching (ArcFace 512-d)'],
    ['5   ANALYSE', 'Fence crossing, intrusion, loitering, abandoned object, night movement'],
    ['6   ACT', 'Alert  ·  signed evidence  ·  store-and-forward to C2'],
  ];
  let sy = 2.26;
  stages.forEach(([t, d], i) => {
    s.addShape('roundRect', { x: 6.68, y: sy, w: 6.04, h: 1.10, rectRadius: 0.08,
      fill: { color: C.white }, line: { color: C.steel, width: 1.25 } });
    s.addText(t, { x: 6.85, y: sy + 0.09, w: 5.7, h: 0.34, fontSize: 15, bold: true,
      color: C.navy, fontFace: HEAD, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(d, { x: 6.85, y: sy + 0.44, w: 5.7, h: 0.58, fontSize: 12.5, color: C.muted,
      fontFace: BODY, align: 'center', valign: 'top', lineSpacingMultiple: 1.0, margin: 0, isTextBox: true });
    if (i < stages.length - 1) {
      s.addShape('rect', { x: 9.66, y: sy + 1.10, w: 0.04, h: 0.14, fill: { color: C.steel } });
      s.addShape('triangle', { x: 9.56, y: sy + 1.22, w: 0.24, h: 0.13, fill: { color: C.steel }, rotate: 180 });
    }
    sy += 1.38;
  });

  card(s, { x: 13.4, y: 1.5, w: 6.1, h: 8.9, title: 'UNIQUE VALUE PROPOSITION',
    color: C.amber, titleColor: C.navy, fill: 'FDF6EC', lineColor: 'EBD3AC' });
  const uvp = [
    ['USES THE CCTV YOU ALREADY HAVE', 'No new cameras and no face or plate appliance - software on one fanless edge box per post.'],
    ['DEGRADES, NEVER GOES DARK', 'With no model files it still detects, on classical motion, and reports itself degraded in health, console and metrics.'],
    ['LIVESTOCK SUPPRESSED BY CLASS', 'A second classifier turns "a wide blob" into "a cow", removing the dominant false alarm on a rural fence line.'],
    ['SURVIVES THE UPLINK', 'Alerts raised during an outage persist in an outbox and replay in order when VSAT or radio returns.'],
    ['EVIDENCE YOU CAN DEFEND', 'SHA-256 per artefact and a per-day hash chain: alteration and deletion are both detectable months later.'],
  ];
  let uy = 2.26;
  uvp.forEach(([t, d]) => {
    s.addText(t, { x: 13.68, y: uy, w: 5.54, h: 0.32, fontSize: 13.5, bold: true, color: C.navy,
      fontFace: HEAD, valign: 'middle', margin: 0, isTextBox: true });
    s.addShape('roundRect', { x: 13.68, y: uy + 0.34, w: 5.54, h: 1.04, rectRadius: 0.07,
      fill: { color: C.white }, line: { color: 'E3CDA8', width: 1 } });
    s.addText(d, { x: 13.82, y: uy + 0.42, w: 5.26, h: 0.9, fontSize: 12.5, color: C.text,
      fontFace: BODY, valign: 'top', lineSpacingMultiple: 1.0, margin: 0, isTextBox: true });
    uy += 1.61;
  });

  s.addNotes('Left: what a border post lives with today. Middle: the node itself, six stages, all of it on the edge. Right: the five things that separate IBVAP from a video management system with motion alarms bolted on.');
}

module.exports = { slide1, slide2 };

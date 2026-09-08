const { C, HEAD, BODY, header, card, panel, rows } = require('./lib.js');

// ========================================================================== //
// 5 - IMPACTS AND BENEFITS
// ========================================================================== //
function slide5(p) {
  const s = p.addSlide();
  s.background = { color: C.white };
  header(s, 'I M P A C T   &   B E N E F I T S', 5);

  // -- left: three impact cards
  const impacts = [
    ['SECURITY & OPERATIONAL IMPACT', C.navy, 'F2F6FA', C.line, [
      'Every camera watched, all night, to one standard - attention no longer decays.',
      'Alerts arrive class-aware, naming the rule and the zone that fired.',
      'Plate and face checks happen at the post, in seconds, without the frame leaving it.',
      'Sentries move from staring at a wall to responding to something specific.',
    ]],
    ['ECONOMIC IMPACT', C.steel, 'F2F6FA', C.line, [
      'Capability added to CCTV that is already paid for, cabled and powered.',
      'No face-recognition or plate-reader appliance to buy and licence per post.',
      'Fewer false-alarm sorties, each of which costs fuel, hours and risk.',
      'Skilled work in deployment and tuning rather than in watching screens.',
    ]],
    ['SOCIAL, ENVIRONMENTAL & GOVERNANCE', C.green, 'EFF8F3', 'C4E2D2', [
      'Every alert carries its evidence and audit trail, so a decision can be reviewed later.',
      'Face embeddings are kept only for watchlist enrolment - never for passers-by.',
      'Reuse over replacement: no new hardware to manufacture, ship or dispose of.',
      'One low-power edge node per post, sized to the load rather than over-provisioned.',
    ]],
  ];
  let iy = 1.5;
  impacts.forEach(([title, colour, fill, line, items]) => {
    card(s, { x: 0.5, y: iy, w: 8.3, h: 2.75, title, color: colour, fill, lineColor: line,
      titleSize: 15 });
    rows(s, {
      x: 0.76, y: iy + 0.7, w: 7.78, rowH: 0.45, gap: 0.05, fontSize: 12.5,
      dotColor: colour, items: items.map((t) => ({ text: t })),
    });
    iy += 2.9;
  });

  // -- right: how it works, three modes
  s.addShape('roundRect', { x: 9.1, y: 1.5, w: 10.4, h: 0.55, rectRadius: 0.1,
    fill: { color: C.amber } });
  s.addText('H O W   I B V A P   W O R K S', {
    x: 9.1, y: 1.5, w: 10.4, h: 0.55, fontSize: 17, bold: true, color: C.navy,
    fontFace: HEAD, align: 'center', valign: 'middle', charSpacing: 1, margin: 0, isTextBox: true,
  });

  const modes = [
    ['MODE 1', 'LIVE WATCH', C.navy, 'F2F6FA', C.line, [
      ['For', 'Every configured camera, continuously'],
      ['Process', 'Ingest → detect → classify → track → rules → event gate'],
      ['Output', 'Alert with class, zone, rule, track IDs, snapshot and clip'],
      ['Time', 'Sub-second, at the post'],
      ['Best for', 'Fence lines, gates and border roads under standing watch'],
    ]],
    ['MODE 2', 'ANALYST REVIEW', C.amber, 'FDF6EC', 'EBD3AC', [
      ['For', 'One source, tuned by hand'],
      ['Process', 'Pick source → pick detector → move thresholds → draw the fence line'],
      ['Output', 'Annotated video, live counters, alert log, JSON export'],
      ['Time', 'On demand, no node required'],
      ['Best for', 'Tuning a site, reviewing footage, briefing and training'],
    ]],
    ['MODE 3', 'SECTOR PICTURE', C.green, 'EFF8F3', 'C4E2D2', [
      ['For', 'Sector headquarters and the C2 system'],
      ['Process', 'Signed webhook → outbox replay → aggregation across posts'],
      ['Output', 'Cross-post trends, recurring hotspots, verified evidence chain'],
      ['Time', 'Live, and resilient to uplink loss'],
      ['Best for', 'Command decisions and post-incident review'],
    ]],
  ];
  const mw = 3.3;
  modes.forEach(([tag, name, colour, fill, line, pairs], i) => {
    const x = 9.1 + i * (mw + 0.25);
    card(s, { x, y: 2.25, w: mw, h: 6.3, title: name, color: colour,
      titleColor: colour === C.amber ? C.navy : C.white, titleSize: 14,
      fill, lineColor: line });
    s.addText(tag, {
      x: x + 0.14, y: 2.32, w: 1.0, h: 0.28, fontSize: 10, bold: true,
      color: colour === C.amber ? C.navy : 'BFD4E8', fontFace: HEAD,
      valign: 'middle', margin: 0, isTextBox: true,
    });
    let y = 2.95;
    pairs.forEach(([k, v]) => {
      s.addText([
        { text: k, options: { bold: true, color: colour === C.amber ? C.navy : colour, breakLine: true } },
        { text: v, options: { color: C.text } },
      ], {
        x: x + 0.2, y, w: mw - 0.4, h: 1.0, fontSize: 12, fontFace: BODY,
        valign: 'top', lineSpacingMultiple: 1.0, margin: 0, isTextBox: true,
      });
      y += 1.08;
    });
  });

  // -- measured figures
  const stats = [
    ['0', 'NEW CAMERAS REQUIRED', C.amber],
    ['27 → 0', 'LIVESTOCK TRACKS RECLASSIFIED → FALSE ALARMS', C.green],
    ['~200 MB', 'INFERENCE RUNTIME FOOTPRINT', C.white],
    ['326', 'AUTOMATED TESTS PASSING', C.white],
  ];
  s.addShape('roundRect', { x: 9.1, y: 8.75, w: 10.4, h: 1.65, rectRadius: 0.12,
    fill: { color: C.navy } });
  const tw = 10.4 / 4;
  stats.forEach(([value, label, colour], i) => {
    const x = 9.1 + i * tw;
    if (i > 0) s.addShape('rect', { x, y: 9.05, w: 0.02, h: 1.05, fill: { color: '2C4A6B' } });
    s.addText(value, {
      x: x + 0.1, y: 8.9, w: tw - 0.2, h: 0.62, fontSize: 26, bold: true, color: colour,
      fontFace: HEAD, align: 'center', valign: 'middle', margin: 0, isTextBox: true,
    });
    s.addText(label, {
      x: x + 0.12, y: 9.54, w: tw - 0.24, h: 0.7, fontSize: 10, bold: true, color: '9FB6CC',
      fontFace: BODY, align: 'center', valign: 'top', lineSpacingMultiple: 1.0,
      margin: 0, isTextBox: true,
    });
  });

  s.addNotes('The four figures at the bottom are measured, not projected: the livestock pair comes from the cattle drill scenario, where the classical fallback alone raises a false intrusion alarm and the classifier stage removes it while a person in the same zone still alarms.');
}

// ========================================================================== //
// 6 - RESEARCH AND REFERENCES
// ========================================================================== //
function slide6(p) {
  const s = p.addSlide();
  s.background = { color: C.white };
  header(s, 'R E S E A R C H   &   R E F E R E N C E S', 6);

  card(s, { x: 0.5, y: 1.5, w: 19.0, h: 5.25, title: 'RESEARCH THE DESIGN RESTS ON',
    color: C.navy, fill: 'F2F6FA', lineColor: C.line, titleAlign: 'left' });

  const refs = [
    ['1', 'Sultani, Chen & Shah (2018)',
      'Real-world Anomaly Detection in Surveillance Videos - CVPR 2018.',
      'Weakly supervised anomaly detection over long, untrimmed surveillance video: the setting a border camera actually produces, where labels are scarce and the interesting minute is buried in a quiet week.',
      'https://arxiv.org/abs/1801.04264'],
    ['2', 'Zhao et al. (2024)',
      'DETRs Beat YOLOs on Real-time Object Detection (RT-DETR) - CVPR 2024.',
      'An end-to-end transformer detector that removes the non-maximum-suppression pass entirely - a real saving on an edge node carrying many cameras on few cores, and why RT-DETR is a first-class detector option in IBVAP.',
      'https://arxiv.org/abs/2304.08069'],
    ['3', 'Zhang et al. (2022)',
      'ByteTrack: Multi-Object Tracking by Associating Every Detection Box - ECCV 2022.',
      'Associating low-scoring boxes as well as high-scoring ones keeps an identity alive through the partial occlusion a fence, a post or a vehicle causes - the basis of the tracker used here.',
      'https://arxiv.org/abs/2110.06864'],
    ['4', 'Howard et al. (2019)',
      'Searching for MobileNetV3 - ICCV 2019.',
      'The efficient classification backbone behind the once-per-track refinement stage, which is what lets the platform tell a cow from a person cheaply enough to run on every new track at the edge.',
      'https://arxiv.org/abs/1905.02244'],
    ['5', 'Deng et al. (2019)',
      'ArcFace: Additive Angular Margin Loss for Deep Face Recognition - CVPR 2019.',
      'The additive angular margin objective that produces the 512-dimensional embeddings compared by cosine similarity in the face-matching stage, with a margin over the runner-up rather than a bare threshold.',
      'https://arxiv.org/abs/1801.07698'],
  ];

  let ry = 2.24;
  refs.forEach(([n, who, what, why, link]) => {
    s.addShape('ellipse', { x: 0.78, y: ry + 0.02, w: 0.34, h: 0.34, fill: { color: C.amber } });
    s.addText(n, { x: 0.78, y: ry + 0.02, w: 0.34, h: 0.34, fontSize: 13, bold: true,
      color: C.navy, fontFace: HEAD, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText([
      { text: who + '  ', options: { bold: true, color: C.navy } },
      { text: what, options: { bold: true, color: C.steel } },
    ], {
      x: 1.26, y: ry, w: 17.9, h: 0.34, fontSize: 13.5, fontFace: BODY,
      valign: 'middle', margin: 0, isTextBox: true,
    });
    s.addText(why, {
      x: 1.26, y: ry + 0.36, w: 14.1, h: 0.5, fontSize: 12, color: C.text,
      fontFace: BODY, valign: 'top', lineSpacingMultiple: 1.0, margin: 0, isTextBox: true,
    });
    s.addText(link, {
      x: 15.5, y: ry + 0.36, w: 3.7, h: 0.34, fontSize: 11, color: C.mid, fontFace: BODY,
      hyperlink: { url: link }, align: 'right', valign: 'top', margin: 0, isTextBox: true,
    });
    ry += 0.92;
  });

  // -- what exists vs where this stands out
  card(s, { x: 0.5, y: 7.0, w: 9.35, h: 3.4, title: 'WHAT EXISTS TODAY', color: C.red,
    fill: 'FDF1EF', lineColor: 'F0C9C2' });
  rows(s, {
    x: 0.78, y: 7.72, w: 8.8, rowH: 0.58, gap: 0.09, fontSize: 12.5, dotColor: C.red,
    items: [
      { label: 'Manual CCTV monitoring.', text: ' A sentry watching a video wall; attention measurably decays within the hour.', inline: true },
      { label: 'Motion and tripwire alarms in a VMS.', text: ' Pixels changed - but a cow, a branch and a man look identical.', inline: true },
      { label: 'Dedicated FRS and ANPR appliances.', text: ' Capable, but bought, cabled and licensed separately for every post.', inline: true },
      { label: 'Cloud video analytics.', text: ' Needs the bandwidth and the uninterrupted uplink a border post does not have.', inline: true },
    ],
  });

  card(s, { x: 10.15, y: 7.0, w: 9.35, h: 3.4, title: 'WHERE IBVAP STANDS OUT', color: C.green,
    fill: 'EFF8F3', lineColor: 'C4E2D2' });
  rows(s, {
    x: 10.43, y: 7.72, w: 8.8, rowH: 0.58, gap: 0.09, fontSize: 12.5, dotColor: C.green,
    items: [
      { label: 'Class-aware alerts.', text: ' Person, vehicle, animal or bag - livestock suppressed by name, not by a threshold.', inline: true },
      { label: 'Face and plate reading in software.', text: ' Same node, same frame, no extra appliance at the post.', inline: true },
      { label: 'Runs at the edge, offline.', text: ' The alert path never leaves the post; an outbox handles the uplink.', inline: true },
      { label: 'Defensible evidence and honest degradation.', text: ' Hash-chained artefacts, and a node that reports what it cannot do.', inline: true },
    ],
  });

  s.addNotes('References are the papers the implementation genuinely rests on, each with the reason it is in the design. The lower half is the competitive read: what a border post can buy today, and what this adds.');
}

module.exports = { slide5, slide6 };

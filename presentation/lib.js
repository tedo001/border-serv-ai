// Shared chrome and card helpers for the IBVAP SIH deck.
// Canvas is 20 x 11.25in, matching the official SIH idea-submission template.

const C = {
  navy:   '0B1F3A',   // dominant: night, command
  mid:    '14406E',
  steel:  '2E5C8A',
  amber:  'E08A1E',   // accent: IR / alert
  red:    'C0392B',
  green:  '1E8E5A',
  tint:   'F2F6FA',   // card ground
  tintW:  'FDF6EC',   // warm card ground
  line:   'C9D6E4',
  text:   '15222E',
  muted:  '5A6B7B',
  white:  'FFFFFF',
};

const HEAD = 'Arial';
const BODY = 'Calibri';

const PS_ID = '26187';
const PS_TITLE = 'AI-Based Intelligent Video Analytics Platform for Border Surveillance using existing CCTV Infrastructure';
const BRAND = 'I B V A P';

const M = { left: 0.5, right: 19.5, top: 1.45, bottom: 10.42 };

// -- chrome ---------------------------------------------------------------- //

function footer(slide, page) {
  slide.addShape('rect', { x: 0, y: 10.55, w: 20, h: 0.70, fill: { color: C.navy } });
  slide.addText('IBVAP', {
    x: 4.6, y: 10.60, w: 1.7, h: 0.6, fontSize: 20, bold: true, color: C.amber,
    fontFace: HEAD, align: 'right', valign: 'middle', margin: 0, isTextBox: true,
  });
  slide.addText('|', {
    x: 6.4, y: 10.60, w: 0.2, h: 0.6, fontSize: 18, color: '4A6480',
    fontFace: HEAD, align: 'center', valign: 'middle', margin: 0, isTextBox: true,
  });
  slide.addText('AI-Based Intelligent Video Analytics Platform for Border Surveillance', {
    x: 6.65, y: 10.60, w: 11.9, h: 0.6, fontSize: 15, bold: true, color: C.white,
    fontFace: BODY, valign: 'middle', margin: 0, isTextBox: true,
  });
  slide.addText(String(page), {
    x: 18.85, y: 10.60, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.white,
    fontFace: HEAD, align: 'right', valign: 'middle', margin: 0, isTextBox: true,
  });
}

// A hairline rectangle outline, drawn as four bars so the weight is exact.
function outline(slide, x, y, w, h, color, t) {
  const k = t || 0.035;
  slide.addShape('rect', { x, y, w, h: k, fill: { color } });
  slide.addShape('rect', { x, y: y + h - k, w, h: k, fill: { color } });
  slide.addShape('rect', { x, y, w: k, h, fill: { color } });
  slide.addShape('rect', { x: x + w - k, y, w: k, h, fill: { color } });
}

// The repeated motif: a detection-box bracket, the one mark this subject owns.
function bracket(slide, x, y, w, h, color, thick) {
  const t = thick || 0.045;
  const arm = Math.min(w, h) * 0.3;
  const seg = (sx, sy, sw, sh) =>
    slide.addShape('rect', { x: sx, y: sy, w: sw, h: sh, fill: { color } });
  seg(x, y, arm, t);            seg(x, y, t, arm);
  seg(x + w - arm, y, arm, t);  seg(x + w - t, y, t, arm);
  seg(x, y + h - t, arm, t);    seg(x, y + h - arm, t, arm);
  seg(x + w - arm, y + h - t, arm, t); seg(x + w - t, y + h - arm, t, arm);
}

function header(slide, title, page) {
  slide.addShape('rect', { x: 0, y: 0, w: 20, h: 1.30, fill: { color: C.navy } });

  slide.addText(BRAND, {
    x: 0.5, y: 0.18, w: 3.3, h: 0.56, fontSize: 26, bold: true, color: C.white,
    fontFace: HEAD, charSpacing: 2, margin: 0, valign: 'middle', isTextBox: true,
  });
  slide.addText('Watch the line  ·  Hold the line', {
    x: 0.5, y: 0.72, w: 3.6, h: 0.36, fontSize: 12, color: '9FB6CC',
    fontFace: BODY, margin: 0, valign: 'middle', isTextBox: true,
  });

  // One size for every section title. 30pt keeps the longest of them
  // ("RESEARCH & REFERENCES", letter-spaced) inside the slot; at 34pt it
  // wrapped onto a second line and collided with the badge.
  slide.addText(title, {
    x: 4.35, y: 0.24, w: 11.0, h: 0.82, fontSize: 30, bold: true, color: C.white,
    fontFace: HEAD, charSpacing: 2, align: 'center', valign: 'middle',
    margin: 0, isTextBox: true,
  });

  slide.addShape('roundRect', {
    x: 15.5, y: 0.40, w: 1.95, h: 0.50, rectRadius: 0.1,
    fill: { color: C.amber },
  });
  slide.addText('PS-' + PS_ID, {
    x: 15.5, y: 0.40, w: 1.95, h: 0.50, fontSize: 15, bold: true, color: C.navy,
    fontFace: HEAD, align: 'center', valign: 'middle', margin: 0, isTextBox: true,
  });

  slide.addText([
    { text: 'SMART INDIA', options: { breakLine: true } },
    { text: 'HACKATHON 2026', options: {} },
  ], {
    x: 17.7, y: 0.26, w: 1.85, h: 0.78, fontSize: 13, bold: true, color: C.white,
    fontFace: HEAD, align: 'right', valign: 'middle', lineSpacingMultiple: 1.05,
    margin: 0, isTextBox: true,
  });

  footer(slide, page);
}

// -- cards ----------------------------------------------------------------- //

function panel(slide, o) {
  slide.addShape('roundRect', {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.12,
    fill: { color: o.fill || C.tint },
    line: { color: o.line || C.line, width: 1 },
  });
}

// A titled card: caption bar in solid colour, body area beneath.
function card(slide, o) {
  panel(slide, { x: o.x, y: o.y, w: o.w, h: o.h, fill: o.fill, line: o.lineColor });
  const capH = o.capH || 0.52;
  slide.addShape('roundRect', {
    x: o.x, y: o.y, w: o.w, h: capH, rectRadius: 0.12,
    fill: { color: o.color || C.navy },
  });
  slide.addShape('rect', {
    x: o.x, y: o.y + capH - 0.13, w: o.w, h: 0.13,
    fill: { color: o.color || C.navy },
  });
  slide.addText(o.title, {
    x: o.x + 0.16, y: o.y, w: o.w - 0.32, h: capH,
    fontSize: o.titleSize || 17, bold: true, color: o.titleColor || C.white,
    fontFace: HEAD, align: o.titleAlign || 'center', valign: 'middle',
    margin: 0, isTextBox: true,
  });
}

// Label + description rows, the shape most of this deck's content takes.
function rows(slide, o) {
  const gap = o.gap === undefined ? 0.12 : o.gap;
  let y = o.y;
  o.items.forEach((it) => {
    const h = it.h || o.rowH || 0.78;
    if (o.dot !== false) {
      slide.addShape('ellipse', {
        x: o.x, y: y + 0.13, w: 0.15, h: 0.15,
        fill: { color: it.dot || o.dotColor || C.amber },
      });
    }
    const tx = o.dot === false ? o.x : o.x + 0.30;
    const tw = o.dot === false ? o.w : o.w - 0.30;
    const runs = [];
    if (it.label) {
      runs.push({
        text: it.label,
        options: { bold: true, color: it.labelColor || C.navy, breakLine: !it.inline },
      });
    }
    if (it.text) {
      runs.push({ text: it.text, options: { color: o.textColor || C.text } });
    }
    slide.addText(runs, {
      x: tx, y, w: tw, h,
      fontSize: o.fontSize || 14, fontFace: BODY, valign: 'top',
      lineSpacingMultiple: o.lineSpacing || 1.02, margin: 0, isTextBox: true,
    });
    y += h + gap;
  });
  return y;
}

module.exports = { C, HEAD, BODY, PS_ID, PS_TITLE, BRAND, M, header, footer, card, panel, rows, bracket, outline };

const pptxgen = require('pptxgenjs');
const { slide1, slide2 } = require('./s12.js');
const { slide3, slide4 } = require('./s34.js');
const { slide5, slide6 } = require('./s56.js');

const p = new pptxgen();
p.defineLayout({ name: 'SIH', width: 20, height: 11.25 });
p.layout = 'SIH';
p.author = 'IBVAP';
p.company = 'SIH 2026';
p.title = 'IBVAP - SIH 2026 - PS 26187 - Border Video Analytics';
p.subject = 'AI-Based Intelligent Video Analytics Platform for Border Surveillance';

[slide1, slide2, slide3, slide4, slide5, slide6].forEach((fn) => fn(p));

const out = process.argv[2] || 'IBVAP-SIH2026-PS26187.pptx';
p.writeFile({ fileName: out }).then((f) => console.log('wrote', f));

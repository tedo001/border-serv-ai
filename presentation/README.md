# SIH 2026 idea submission — PS 26187

Two decks are kept here. `sih-format/` is the submission built into the
**official** SIH 2026 Idea Presentation Format and is the one to upload;
the files below are a self-designed deck in the same six-section order,
useful for a longer walkthrough.

`IBVAP-SIH2026-PS26187.pptx` is the six-slide walkthrough deck on a wide
canvas (20 × 11.25 in), following the same six sections: title, idea and
solution, technical approach, feasibility and viability, impact and
benefits, research and references.

The deck is generated rather than hand-drawn, so a content change is a code
change and the alignment cannot drift:

```bash
cd presentation
npm install pptxgenjs          # once
node build.js IBVAP-SIH2026-PS26187.pptx
```

| File | Slides |
|---|---|
| `lib.js` | Palette, header and footer chrome, card and row helpers |
| `s12.js` | 1 Title · 2 Idea & solution |
| `s34.js` | 3 Technical approach · 4 Feasibility & viability |
| `s56.js` | 5 Impact & benefits · 6 Research & references |
| `build.js` | Assembles the six slides into one file |

**Before submitting**, fill in Team ID and Team Name on slide 1 — the two ruled
blanks in `s12.js`'s `facts` table. Everything else is complete.

Figures quoted on slide 5 are measured from this repository, not projected:
the livestock pair comes from the `cattle` drill scenario, where the classical
fallback alone raises a false intrusion alarm and the classifier stage removes
it while a person in the same zone still alarms.

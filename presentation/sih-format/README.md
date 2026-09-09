# SIH 2026 idea submission — official template (PS 26187)

`IBVAP-SIH2026-PS26187.pptx` is the six-slide idea submission filled directly
into `template.pptx`, the Idea Presentation Format published for Smart India
Hackathon 2026 (13.33 × 7.5 in). The template's own chrome — title styling,
the team-name oval, the footer bar, the SIH artwork — is left untouched, and
the pointer on each slide is the one the format prescribes:

| Slide | Prescribed pointer(s) |
|---|---|
| 1 | Title page: PS ID, PS title, theme, category, organisation, department, team |
| 2 | Proposed Solution — detailed explanation · how it addresses the problem · innovation and uniqueness |
| 3 | Technical Approach — technologies to be used · methodology and process for implementation |
| 4 | Feasibility and Viability — feasibility analysis · challenges and risks · strategies for overcoming them |
| 5 | Impact and Benefits — potential impact on the target audience · benefits (social, economic, environmental) |
| 6 | Research and References — details / links of the reference and research work |

The template's seventh slide is its instruction sheet, which the format tells
you to remove; `build.py` removes it.

## Rebuilding

```bash
cd presentation/sih-format
pip install python-pptx        # once
python build.py                # reads template.pptx, writes IBVAP-SIH2026-PS26187.pptx
```

| File | Contents |
|---|---|
| `template.pptx` | The unmodified official format, as published |
| `lib.py` | Palette and the card, panel, chip, arrow and text helpers |
| `build.py` | Slide-by-slide content and layout |

## Before submitting

- Fill in **Team ID** and **Team Name** on slide 1 — the two ruled blanks.
- SIH accepts the idea submission as **PDF**, not `.pptx`:
  `soffice --headless --convert-to pdf IBVAP-SIH2026-PS26187.pptx`.

Figures on slide 5 are measured in this repository, not projected. The
livestock pair (27 → 0) comes from the `cattle` drill scenario, where the
classical fallback alone raises a false intrusion alarm and the classifier
stage removes it while a person in the same zone still alarms; 427 is the
automated test count; ~200 MB is the ONNX Runtime inference footprint against
a full training stack.

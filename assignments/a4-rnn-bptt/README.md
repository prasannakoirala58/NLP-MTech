# a4 — RNN / BPTT

**Classroom:** *"RNN - Hand Written Assignment."* · 10 points · **due 28 Aug** · status was
**Missing**. Instruction: *"Complete the attached assignment and submit a scanned handwritten
note before the beginning of next week's class."*

## What to submit

| # | File | |
|---|---|---|
| 1 | your scanned handwritten note | **the actual submission** |
| 2 | `PrasannaKoirala_RNN_BPTT.ipynb` | attach to Classroom as a second file |

The note is **self-contained** — all eight questions answered in it. The notebook and repo
link are supporting evidence, never a substitute. A required answer must never depend on the
grader following a link.

## Copy from this

**`DRAFT.pdf`** — 5 typeset A4 pages.

Two things to leave blank, ruled with a straight edge:

| box | size | what goes in |
|---|---|---|
| **FIGURE 1** | ~14 × 8 cm | gradient decay across 30 timesteps |
| **FIGURE 2** | ~14 × 9 cm | the printed `.grad` for every parameter |

**Q1's diagram you draw yourself** — it is boxes and arrows, no figure needed. The draft shows
the layout.

## Files

| File | |
|---|---|
| `DRAFT.pdf` | **copy from this** |
| `DRAFT.html` | source the PDF renders from |
| `render.sh` | rebuilds `DRAFT.pdf` |
| `PrasannaKoirala_RNN_BPTT.ipynb` | all 8 answers, live code, 2 figures |

## Status — COMPLETE

- [x] Notebook built, runs clean, 14/14 checks pass
- [x] Draft typeset, 5 pages
- [x] Copied out by hand — **15 pages**, both boxes left blank
- [x] Scanned
- [x] Figures inserted, final PDF committed

**Submit `PrasannaKoirala_RNN_BPTT.pdf`** (15 pages, 5.2 MB) + attach the `.ipynb`.

### How the figures got in

`insert_figures.py`, re-runnable if a figure changes:

1. Renders the scan at 175 dpi
2. Finds the ruled box on **page 10** (1090x673 px) and **page 13** (990x754 px) by
   looking for peaks in ink density inside the region each box occupies
3. Whites out the interior, keeping the hand-drawn border, and pastes the figure scaled
   to fit and centred
4. Reassembles — handwriting pages greyscale to halve the size, **the two figure pages in
   colour** since the plot lines are colour-coded

Source scan 7.15 MB, finished file 5.19 MB. Legibility of the smallest text (the printed
gradient values) was verified by reading the compressed output back.

# C2 — Embedding Space Exploration  (handwritten + figures)

The brief puts this under *"Write Handwritten Notes. Scan and upload PDF."*

## Copy from this

**`DRAFT.pdf`** — 4 typeset A4 pages.

**Two figures are NOT written by hand.** Where the draft shows a dashed box, rule an
empty box in your notes about **15 cm wide × 10 cm tall** and write `FIGURE 1` (or
`FIGURE 2`) inside it. Leave it blank.

## The hand-off

```
1. Write the notes, leaving the two ruled boxes empty
2. Scan / photograph the pages  ->  one PDF
3. Send that PDF over
4. The two plots get pasted into your boxes and the PDF is rebuilt
5. Final file saved here as PrasannaKoirala_Embeddings_C2.pdf
```

Rule the boxes with a straight edge — a clean rectangle is a much more reliable target
to paste into than a rough gap.

## Files

| File | What it is |
|---|---|
| `DRAFT.pdf` | **the one to copy from** |
| `DRAFT.html` | source the PDF is rendered from |
| `render.sh` | rebuilds `DRAFT.pdf` from `DRAFT.html` |
| `make_figures.py` | regenerates the two plots |
| `outputs/figure1_pca.png` | goes in your FIGURE 1 box |
| `outputs/figure2_tsne.png` | goes in your FIGURE 2 box |

## What the brief requires

- [x] Cosine similarity computations between words → §1
- [x] Vector arithmetic (`king - man + woman ≈ queen`) → §2
- [x] Visualise embeddings using PCA or t-SNE *(marked optional — we do both)* → §3, §4
- [x] Interpret the visualised clusters *(marked optional — we do it)* → §5

## Status — COMPLETE

- [x] Draft written and typeset
- [x] Figures generated
- [x] Copied out by hand — 11 pages, both boxes left blank
- [x] Scanned
- [x] Figures pasted in, final PDF committed

**Submit `PrasannaKoirala_Embeddings_C2.pdf`** (11 pages, 3.9 MB).

### How the figures got in

`insert_figures.py` does it, and can be re-run if a figure changes:

1. Renders the scan at 175 dpi
2. Finds the ruled box on pages 7 and 8 by looking for peaks in ink density
   inside the region where the box sits
3. Whites out the interior (keeping the hand-drawn border) and pastes the figure,
   scaled to fit and centred
4. Reassembles the PDF — handwriting pages in greyscale to keep the size down,
   **the two figure pages in colour** because the clusters are colour-coded

Detected boxes: page 7 at 1033x813 px, page 8 at 1039x857 px. Source scan was
5.67 MB; the finished file is 3.9 MB.

## Findings worth defending in a viva

All measured from GloVe `glove-wiki-gigaword-100`.

1. **`nepal — france` = 0.22 but `nepal — india` = 0.68.** Both pairs are "two
   countries", so a taxonomy would score them alike. Embeddings learn *usage*, not
   category.
2. **`bigger − big + small` returns `larger` (0.89) ahead of `smaller` (0.87).** The
   analogy fails — antonyms sit too close together.
3. **`doctor − man + woman` returns `nurse` (0.77).** The vectors learned a human bias
   straight out of the training text. Directly relevant to syllabus Unit 9.
4. **`apple` clusters with technology, not food**, in both plots. One word, two meanings,
   one vector — the exact limitation contextual embeddings (P5) exist to fix.
5. **PCA retains only 35% of the variance.** 65% of the information is not in the
   picture, so the plot is a summary and never proof.

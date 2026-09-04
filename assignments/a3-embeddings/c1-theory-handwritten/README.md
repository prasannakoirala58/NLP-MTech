# C1 — Word Embeddings: Theory and Analysis  (handwritten)

The brief puts this under *"Write Handwritten Notes. Scan and upload PDF."*

## Copy from this

**`DRAFT.pdf`** — 4 typeset A4 pages with 3 proper vector figures. This is the one to
read from while writing.

Other files here:

| File | What it is |
|---|---|
| `DRAFT.pdf` | **the one to copy from** — typeset, with figures |
| `DRAFT.html` | source the PDF is rendered from |
| `DRAFT.md` | same content as plain text, for diffing and search |
| `render.sh` | regenerates `DRAFT.pdf` from `DRAFT.html` |

If the content changes, edit `DRAFT.html`, run `bash render.sh`, and update `DRAFT.md`
to match.

## What the brief requires — all four are covered

- [x] Compare one-hot encoding with dense word embeddings → §1
- [x] Explain Word2Vec (CBOW vs Skip-Gram), GloVe, FastText → §2
- [x] Discuss semantic similarity with examples → §3
- [x] Analyse the role of dimensionality → §4

## Status — COMPLETE

- [x] Draft written
- [x] Typeset as PDF with figures
- [x] Copied out by hand — 10 pages
- [x] Scanned to PDF
- [x] Final scan committed as `PrasannaKoirala_Embeddings_C1.pdf`

**Submit `PrasannaKoirala_Embeddings_C1.pdf`** (10 pages, 2.6 MB).

### Content verified against the brief

All four required tasks present, all three figures drawn by hand:

| Required | Where | |
|---|---|---|
| One-hot vs dense embeddings | §1, pp. 1–3 | ✅ |
| Word2Vec (CBOW vs Skip-Gram), GloVe, FastText | §2, pp. 3–6 + Figure 1 | ✅ |
| Semantic similarity with examples | §3, pp. 6–8 + Figure 2 | ✅ |
| Role of dimensionality | §4, pp. 9–10 + Figure 3 | ✅ |

### Compression

The scan arrived as 4.63 MB (10 pages of 310 dpi RGB photographs). Re-encoded to
greyscale at 175 dpi with Ghostscript: **4.63 MB → 2.64 MB, a 43% reduction**, page count
unchanged and legibility verified by reading the compressed pages back. Ink is not
coloured, so greyscale loses nothing.

### Minor transcription slips (not worth rewriting for)

Spotted while reading the scan. None affect the marks materially:

- p. 8 — "both horizontal arrows have the same **dimension** and length" should be
  **direction** (the draft says direction).
- p. 9 — "costs more **money** and time" should be **memory**.
- p. 8 — Figure 2's two horizontal arrows are missing their "royalty" labels.
- The handwritten capital **B** in CBOW reads as an **R** at a glance, so CBOW can look
  like "CROW". Legible on close reading, but worth a clearer B next time.

## The numbers are real

Every similarity figure in §3 is **measured**, not quoted — computed from GloVe
`glove-wiki-gigaword-100` (400,000 words, 100 dimensions). Two are worth remembering:

- **good — bad = 0.77**, nearly as high as king — queen, because cosine similarity
  measures shared *context*, not shared *meaning*. Static embeddings cannot separate
  antonyms from synonyms.
- **paris − france + nepal = kathmandu (0.81)**. The country→capital direction transfers
  to a country the examples never mentioned.

The same GloVe vectors are reused in C2.

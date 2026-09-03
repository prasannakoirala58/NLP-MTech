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

## Status

- [x] Draft written
- [x] Typeset as PDF with figures
- [ ] Copied out by hand
- [ ] Scanned to PDF
- [ ] Final scan saved here as `PrasannaKoirala_Embeddings_C1.pdf` and committed

## The numbers are real

Every similarity figure in §3 is **measured**, not quoted — computed from GloVe
`glove-wiki-gigaword-100` (400,000 words, 100 dimensions). Two are worth remembering:

- **good — bad = 0.77**, nearly as high as king — queen, because cosine similarity
  measures shared *context*, not shared *meaning*. Static embeddings cannot separate
  antonyms from synonyms.
- **paris − france + nepal = kathmandu (0.81)**. The country→capital direction transfers
  to a country the examples never mentioned.

The same GloVe vectors are reused in C2.

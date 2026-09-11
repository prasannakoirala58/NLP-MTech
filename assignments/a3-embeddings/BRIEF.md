# a3 — Word Embeddings

Sources: `questions/Embedding_assignment.pdf` + `questions/Assignment-3.jpg`
Syllabus Unit 2 · Labs 2–3

## The rule
> "Out of total 5 assignments, you are free to pick **any 3** assignments including
> **at least one Conceptual Assignment**."

Conceptual assignments must be **handwritten, scanned, and uploaded as PDF**.

## The Classroom post (`Assignment-3.jpg`) adds
- **Group A (Expert):** implement **Skip-gram with Negative Sampling (SGNS)** of Word2Vec
  in **pure Python**. Deep learning frameworks **not allowed**. NumPy permitted.
- **Group B:** if the above is challenging, complete Assignments 1–3 from Materials.
- Encouraged to complete **both**.
- Separate item: *"Embedding Assignment – other than SKIPgram implementation"*,
  **10 points**, assigned 9 Aug, status **Assigned** (still open).

## The 5 options
| # | Type | Title |
|---|---|---|
| C1 | Conceptual | Word Embeddings – Theory and Analysis |
| C2 | Conceptual | Embedding Space Exploration |
| P3 | Programming | Build a Word2Vec Model (Skip-Gram or CBOW) |
| P4 | Programming | Use Pre-trained Embeddings (GloVe/FastText) downstream |
| P5 | Programming | Contextualised vs Static Embeddings |

## Our scope: **all 5**, not the required 3

The "pick any 3" is a floor, not a target. We do every one — for learning, not marks.
Doing all five is less than 5x the work because they build on each other.

**Key insight:** Group A's SGNS task *is* P3 ("implement training using negative
sampling"). One pure-Python implementation satisfies both the expert track and P3. We also
do Group B's list, so both Classroom tracks are covered.

**Build order: C1 -> C2 -> P3 -> P4 -> P5.** Not arbitrary — C1 gives the theory that C2
explores numerically; together they are the preparation for implementing P3 from scratch.
P4 then swaps in *pre-trained* vectors so your own SGNS output can be measured against
production embeddings — a comparison that only exists because P3 came first. P5 finally
breaks the static assumption entirely.

### C1 — Word Embeddings: Theory and Analysis  -> `c1-theory-handwritten/`
- [ ] Compare one-hot encoding with dense word embeddings
- [ ] Explain differences between Word2Vec (**CBOW vs Skip-Gram**), GloVe, and FastText
- [ ] Discuss semantic similarity in embedding spaces, with examples
- [ ] Analyse the role of **dimensionality** in embedding representations
- [ ] **Handwritten** -> scanned -> PDF

### C2 — Embedding Space Exploration  -> `c2-embedding-space-exploration/`
- [ ] Cosine similarity computations between words
- [ ] Vector arithmetic (`king - man + woman ~= queen`)
- [ ] Visualise embeddings using PCA or t-SNE *(marked optional in brief — we do it)*
- [ ] Interpret the visualised clusters *(marked optional — we do it)*
- [ ] **Handwritten** derivations + printed plots attached to the scan

### P3 — Build a Word2Vec Model (SGNS, pure Python)  -> `p3-sgns-pure-python/`  ✅ DONE
- [x] Preprocess text (tokenisation, vocabulary indexing) — 685k tokens, 5,617 vocab
- [x] Implement training using **negative sampling** — 8.35M pairs, loss 3.20 -> 2.45
- [x] Evaluate with similarity tasks or analogies — both; similarity worked, analogies
      mostly failed (corpus 10,000x smaller than where analogies were discovered)
- [x] Tools: Python + NumPy **only** — asserted programmatically in the checklist

### P4 — Use Pre-trained Embeddings  -> `p4-pretrained-embeddings/`  ✅ DONE
- [x] Load GloVe or FastText embeddings — GloVe 100d
- [x] Use as input for sentiment classification — NLTK Movie Reviews, 2,000 labelled
- [x] **Compare performance with vs without** embeddings — bag of words 82.0% vs GloVe 72.8%
- [x] Bonus: benchmarked against our own P3 vectors — 64.8%, limited by 65.7% coverage
- [x] Extra: fairness control at equal feature budget (BoW 74.0% vs GloVe 72.8% at 100)

### P5 — Contextualised vs Static Embeddings  -> `p5-contextual-vs-static/`
- [ ] Use HF Transformers to extract contextual embeddings
- [ ] Analyse sentence similarity **or** word sense disambiguation
- [ ] Report differences in performance / interpretability

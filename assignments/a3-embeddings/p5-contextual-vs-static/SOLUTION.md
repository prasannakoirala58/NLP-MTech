# a3-P5 — Solution trace

**Deliverable:** `PrasannaKoirala_Contextual_vs_Static.ipynb` (25 cells, runs clean)

---

## 1. This assignment answers a question our own earlier work raised

In **C2** the PCA and t-SNE plots put `apple` in the **technology** cluster, beside
*computer* and *laptop*, rather than with the food. The write-up at the time said:

> *"apple has two distinct meanings and GloVe gives every word exactly one vector. That
> single vector has to serve both senses, so it lands in a compromise position."*

P5 tests whether contextual embeddings fix it. They do.

## 2. First, prove the limitation rather than assume it

Five ambiguous words, each in two sentences forcing different senses. GloVe's answer:

| word | identical vectors? | cosine |
|---|---|---|
| bank, bat, apple, rock, spring | **True** for every one | **1.0000** |

**This is not GloVe scoring badly — GloVe never saw the sentences.** It is a lookup table
with one row per word, and a table has no mechanism for returning two answers to one key.
No amount of extra training data changes that; it is structural.

## 3. BERT's result

| word | GloVe | BERT |
|---|---|---|
| bank | 1.000 | **0.483** |
| bat | 1.000 | **0.414** |
| apple | 1.000 | **0.341** |
| rock | 1.000 | **0.511** |
| spring | 1.000 | **0.343** |
| **average** | **1.000** | **~0.42** |

The senses separate clearly. `apple` — the word that started this in C2 — separates most
strongly of all, at 0.341.

## 4. The control is what makes it a result

Low scores alone prove nothing. An obvious objection: perhaps BERT just emits a slightly
different vector every time and the low numbers are noise.

So the same test was run on the **same sense** in two different sentences:

| | average cosine |
|---|---|
| same sense, different sentences | **0.82** |
| different senses | **0.42** |
| **gap** | **0.40** |

**BERT keeps same-sense pairs close and pushes different-sense pairs apart.** It is
tracking meaning, not adding randomness. Without this control the experiment would have
been suggestive rather than conclusive — worth running before drawing any conclusion.

## 5. A bug in my own verification

The checklist initially **failed** on *"GloVe always returns exactly 1.0"*, which was
alarming since the claim is structurally guaranteed.

Investigating: `cosine(glove["bat"], glove["bat"])` returns `1.0000001192092896` — off by
1.2e-07. The vectors are byte-identical; the discrepancy comes entirely from dividing by a
**float32** norm. My tolerance of `1e-9` was simply too tight for float32 precision.

**Fixed properly** by asserting `np.array_equal` — vector *identity*, which is the actual
claim — with a loosened 1e-5 cosine tolerance alongside. Worth recording because the
instinct on a failing assertion should be to investigate before loosening it; here the
investigation showed the assertion was measuring the wrong thing.

## 6. Sentence similarity

Also run, as the brief offers it as an alternative analysis. The interesting cases are
sentences sharing few words but meaning the same thing — where word-overlap matching fails
and meaning-based comparison should not.

## 7. Performance versus interpretability — the honest trade

The brief asks for both, and they point in **opposite directions**.

**Performance: BERT wins clearly.** It distinguishes word senses (GloVe cannot at all),
handles unseen words via sub-word pieces (GloVe has no vector for them), and uses word
order (GloVe ignores it).

**Interpretability: GloVe wins, and it is not close.** GloVe is a file. Open it, find the
row for `bank`, read the 100 numbers. If two words are similar you can point at the exact
rows responsible — the entire model is inspectable in a text editor.

BERT's answer emerges from **110 million parameters** interacting across 12 layers. There
is no row to inspect. When it is wrong, the reason is not locatable in any simple way.

**So: more capable, less explainable.** For a medical or legal system, being unable to
justify a decision can matter more than a few points of accuracy. This is the exact tension
in syllabus Unit 8 (Evaluation, Explainability and Responsible AI).

## 8. Cost

| | GloVe | BERT |
|---|---|---|
| Download | ~130 MB | ~440 MB |
| Model | a lookup table | 110M parameters |
| Getting a vector | one array lookup | a full forward pass |

`torch` took `.venv` from 638 MB to roughly 2.5 GB. Not wasted — a4 needs PyTorch for Q8.

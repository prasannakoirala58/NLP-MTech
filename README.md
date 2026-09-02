# NLP Assignments — AIAC 536

Coursework for **Natural Language Processing (AIAC 536)**, Kathmandu University,
Department of Artificial Intelligence — MTech in AI, Semester II.

## Layout

| Path | Contents |
|---|---|
| `questions/` | Original assignment briefs and the course syllabus (read-only) |
| `assignments/` | One folder per assignment: brief, solution trace, notebook, data, outputs |
| `docs/learning/` | Topic notes written before implementation, aligned to syllabus units |
| `common/` | Shared helpers, once something is genuinely reused twice |

## Assignments

| ID | Topic | Syllabus unit |
|---|---|---|
| `a1-pos-tagging` | POS extraction from a news article | 1 — Foundations |
| `a2-ner` | Named Entity Recognition from a news article | 1 — Foundations |
| `a3-embeddings` | Word embeddings: theory, SGNS from scratch, contextual vs static | 2 — Text Representation |
| `a4-rnn-bptt` | RNNs, Backpropagation Through Time, gradient dynamics | 3 — Neural NLP |

Each assignment folder contains a `BRIEF.md` (requirements as a checklist) and a
`SOLUTION.md` (how it was solved, and why).

## Setup

Managed with [uv](https://docs.astral.sh/uv/). Python 3.12.

```bash
uv sync --group dev --group nlp      # a1, a2
uv sync --group dev --group embed    # a3
uv sync --group dev --group deep     # a3-p5, a4

uv run python -m spacy download en_core_web_sm
uv run jupyter lab
```

Do not use `pip` directly — it desyncs `uv.lock`.

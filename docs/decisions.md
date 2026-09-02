# Decisions

Short record of choices that would otherwise get re-litigated.

## uv instead of pip/conda
One resolver, one lockfile, reproducible environments, and fast. Dependency **groups**
(`nlp`, `embed`, `deep`, `dev`) keep PyTorch out of the environment for assignments that
only need spaCy. Never `pip install` into the venv — it desyncs `uv.lock`.

## Python 3.12, pinned
spaCy/thinc/gensim wheel availability is most reliable on 3.12. 3.13+ still has
occasional gaps in this specific stack. Already present at
`/opt/homebrew/bin/python3.12`, so no download needed.

## Repo, not Google Colab
The repo is the source of truth. Colab breaks version control, makes notebook diffs
meaningless, and hides the environment. If a later assignment needs a CUDA GPU, Colab is a
**runner, not a home**: export from here, run there, bring outputs back.

## Per-assignment folders, not one global code tree
Each assignment is submitted as one self-contained bundle (notebook + `.txt` + `.csv` +
writeup). Three parallel top-level trees would mean three places to look while solving one
assignment and three places to gather from at submission. Top-level `docs/learning/`
remains for cross-cutting topic notes; `common/` for code genuinely shared by two or
more assignments. `common/` stays empty until a second assignment actually needs
something — no package scaffolding, no premature structure.

## a3: all 5, not the required 3
The brief allows any 3 of 5 with at least one conceptual. We do all five. The requirement
is a floor, not a target, and the five build on each other so the marginal cost is well
under 5x. The Classroom post's Group A "expert" task (SGNS in pure Python) is the *same
work* as Programming Assignment 3, so one implementation covers both tracks; we do Group
B's list too. Build order C1 -> C2 -> P3 -> P4 -> P5, so that theory precedes
implementation and our own vectors exist before we benchmark against pre-trained ones.

Same principle applies repo-wide: every "optional" and "bonus" item in every brief is
treated as required.

## Order: a1 -> a2 -> a3 -> a4, syllabus order
Not deadline order. Each assignment is preparation for the next: a1/a2 build the
tokenisation and text pipeline, a3 turns tokens into vectors, a4 feeds vectors into a
sequence model. a3 is the only one with a known open deadline, so revisit if time tightens.

## One shared news article for a1 and a2
The brief permits either. Running POS tagging and NER over identical text makes the
comparison between the two pipelines meaningful, and halves the sourcing work.

## CLAUDE.md is excluded locally, not gitignored
Listed in `.git/info/exclude`, which is repo-scoped and never committed. This keeps
working notes out of the repo without advertising their existence in `.gitignore`.

## Apple Silicon / MPS
Machine is arm64. PyTorch uses the MPS backend. No assignment in a1–a4 needs CUDA:
POS/NER is CPU-trivial, SGNS is deliberately small-corpus, BERT usage here is
inference-only on short sentences.

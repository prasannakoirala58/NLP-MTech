# Corpus source

## Why not the a1/a2 news article

That article is **843 words**. Word2Vec learns from repeated co-occurrence — a word seen
twice teaches nothing. P3 needs hundreds of thousands of words, so it uses a different
corpus. This is the only assignment that does.

## What this corpus is

Six books from **Project Gutenberg**, bundled with NLTK (`nltk.corpus.gutenberg`).
All are public domain.

| Book | Words | Why it is here |
|---|---|---|
| `melville-moby_dick.txt` | 260,819 | sea and whaling — whale, ship, harpoon, pequod |
| `edgeworth` *(not used)* | — | — |
| `austen-emma.txt` | 192,427 | Austen's society vocabulary |
| `austen-sense.txt` | 141,576 | same |
| `austen-persuasion.txt` | 98,171 | same |
| `carroll-alice.txt` | 34,110 | its own cast — gryphon, dormouse, hatter |
| `chesterton-brown.txt` | 86,063 | detective fiction |
| **total** | **~813,000** | 685,835 tokens after cleaning |

## Why these six

Chosen for **distinct, non-overlapping vocabularies**, so the result is verifiable by
eye. If the training works, whaling words should end up near whaling words and Wonderland
characters near each other — with no supervision and no labels.

It worked: `ship` → `pequod`, `alice` → `gryphon, dormouse`, `captain` → `wentworth,
benwick`. Those are the right characters from the right books.

## Files

| File | |
|---|---|
| `corpus.txt` | the raw text the notebook reads, 3.6 MB — **committed** |
| `corpus_tokens.txt` | after lowercasing and `[a-z]+` cleaning — gitignored, derivable in one line |

To regenerate either:

```python
from nltk.corpus import gutenberg
raw = " ".join(gutenberg.raw(b) for b in BOOKS)      # corpus.txt
tokens = re.findall(r"[a-z]+", raw.lower())          # corpus_tokens.txt
```

## Note on the trade-off

A corpus this size trains in 95 seconds in pure NumPy, which is what makes the assignment
practical. It is also **~10,000× smaller** than GloVe's 6 billion words, which is why the
analogy tests in the notebook largely fail. That trade-off is discussed in `SOLUTION.md`.

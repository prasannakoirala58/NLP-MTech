# C1 — Word Embeddings: Theory and Analysis

> **Copy out everything below the line. Skip this box.**
> Target: **4–5 pages** of handwriting. 3 small diagrams — all quick to draw.
> The tables save you writing: a table row says in 6 words what a sentence needs 20 for.

---

## Word Embeddings — Theory and Analysis

**Prasanna Koirala** · AIAC 536 Natural Language Processing · Kathmandu University

### Introduction

A computer cannot understand the word *cat*. It can only do arithmetic. So every word must
first be turned into numbers.

A **word embedding** is exactly that — a word represented as a list of numbers, called a
**vector**:

```
"cat"  →  [ 0.21, -0.44, 0.87, ... ]
```

What matters is *which* numbers. Chosen well, **words with similar meanings get similar
numbers**, so a computer can tell that *cat* is closer to *dog* than to *rocket* — without
being told what any of them mean.

---

### 1. One-Hot Encoding vs Dense Embeddings

**One-hot encoding.** Each word becomes a vector as long as the whole vocabulary: all
zeros, with a single 1 marking its position.

Vocabulary = [ cat, dog, king, queen, rocket ]

```
cat  = [ 1, 0, 0, 0, 0 ]
dog  = [ 0, 1, 0, 0, 0 ]
king = [ 0, 0, 1, 0, 0 ]
```

Two serious problems:

**(a) Size.** A real vocabulary holds ~50,000 words, so every word becomes 50,000 numbers
of which 49,999 are zero. Nearly all storage is wasted.

**(b) No meaning.** Every pair of different words is equally far apart:

> cos(cat, dog) = 0   and   cos(cat, rocket) = 0

The encoding claims *cat* is as unrelated to *dog* as to *rocket*.

**Dense embeddings** fix both. Each word becomes a short vector of real numbers **learned
from text**. Because *cat* and *dog* appear in similar sentences, training pulls their
vectors together.

| | One-hot | Dense embedding |
|---|---|---|
| Length | vocabulary size (~50,000) | small and fixed (50–300) |
| Values | zeros and one 1 | real numbers |
| Created by | indexing | learned from data |
| Captures meaning | **No** | **Yes** |
| Similar words close | no — all equidistant | yes |

---

### 2. Word2Vec, GloVe and FastText

All three rest on the **distributional hypothesis**:

> *"You shall know a word by the company it keeps."* — J.R. Firth, 1957

Words used in similar contexts have similar meanings.

**Word2Vec (2013)** slides a window over the text and trains a small neural network.
Sentence `the cat sat on the mat`, centre word **sat**, context [the, cat, on, the].
There are two ways round:

```
┌─────────── DIAGRAM 1 ────────────────────────┐
│                                              │
│  CBOW       [the, cat, on, the] ──► sat      │
│             context predicts the word        │
│                                              │
│  SKIP-GRAM  sat ──► [the, cat, on, the]      │
│             word predicts the context        │
│                                              │
└──────────────────────────────────────────────┘
```

| | CBOW | Skip-Gram |
|---|---|---|
| Direction | context → word | word → context |
| Speed | faster | slower |
| Rare words | weaker | **better** |
| Small corpus | weaker | **better** |

**GloVe (2014)** — *Global Vectors*. Word2Vec sees only one window at a time. GloVe first
counts how often every pair of words co-occurs across the **whole corpus**, then
factorises that count matrix.

> Word2Vec = local windows.  GloVe = global counts.

**FastText (2016)** treats a word as a bag of **character n-grams**:

```
"playing" → pla, lay, ayi, yin, ing
"played"  → pla, lay, aye, yed        (shares pla, lay)
```

Two advantages: an **unseen word** still gets a vector from its pieces, and **morphology**
comes free — *play / played / playing* share n-grams so they get related vectors. This
matters greatly for morphologically rich languages such as **Nepali**.

| Model | Core idea | Strength |
|---|---|---|
| Word2Vec | predict within a local window | fast, simple |
| GloVe | factorise global co-occurrence counts | whole-corpus statistics |
| FastText | word = sum of character n-grams | unseen words, morphology |

---

### 3. Semantic Similarity

Closeness of meaning is measured by **cosine similarity** — the *angle* between two
vectors, ignoring their length:

```
              A · B
cos(A, B) = ─────────        −1 opposite  ·  0 unrelated  ·  +1 identical
            |A| × |B|
```

**Why the angle, not the distance?** Frequent words develop longer vectors. Plain distance
would call a common word "far" from a rare one purely because of frequency. The angle
discards length and keeps direction — and meaning lives in the direction.

**Measured on GloVe (100 dimensions, 400,000 words):**

| Pair | cos | |
|---|---|---|
| cat — dog | **0.88** | same contexts |
| good — bad | **0.77** | *see warning* |
| king — queen | **0.75** | both royalty |
| cat — rocket | **0.19** | unrelated |

**Warning.** *good — bad* scores 0.77 — nearly as high as *king — queen*, though they are
opposites. Cosine similarity really measures **"appears in similar contexts"**, not "means
the same". Both fit "this film was very ___". So **static embeddings cannot separate
antonyms from synonyms** — a real limitation.

**Vector arithmetic.** Relationships become *directions*:

```
┌─────────── DIAGRAM 2 ────────────────────────┐
│                                              │
│     man  ─────────────►  king                │
│      │                    │                  │
│      ▼ "female"           ▼ "female"         │
│    woman ─────────────►  queen               │
│                                              │
│    horizontal arrow = "royalty"              │
└──────────────────────────────────────────────┘
```

| Arithmetic | Result | Score |
|---|---|---|
| king − man + woman | **queen** | 0.77 |
| paris − france + nepal | **kathmandu** | 0.81 |

The vectors were never told what a capital city is. They learned *country → capital* from
usage alone, and it transfers correctly to Nepal.

---

### 4. The Role of Dimensionality

**d** = how many numbers represent each word.

**Too small** (d = 5): not enough room, different meanings collide — the model
**underfits**. **Too large** (d = 1000): each dimension sees too little data, so the model
memorises noise instead of meaning — it **overfits** — and costs more memory and time.

```
┌─────────── DIAGRAM 3 ────────────────────────┐
│ quality                                      │
│   ▲        ______                            │
│   │     __/      \____                       │
│   │   _/               \___                  │
│   │ _/                                       │
│   └─┴─────┴──────┴──────┴──►  d              │
│    10    50    300   1000                    │
│           sweet spot                         │
└──────────────────────────────────────────────┘
```

| d | Effect |
|---|---|
| < 20 | underfits — meanings collide |
| **50 – 300** | **usual sweet spot** |
| > 1000 | overfits, slower, diminishing returns |

**Important:** individual dimensions are **not interpretable**. Dimension 42 does not mean
"royalty". Meaning is spread across the whole vector and appears as *directions* through
the space, not along single axes. This is why embeddings work well but are hard to
explain.

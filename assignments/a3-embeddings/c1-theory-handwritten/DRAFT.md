# C1 — Word Embeddings: Theory and Analysis

> **This is the draft to copy out by hand.** Roughly 3 pages of handwriting.
> Four sections, one per task in the brief. Diagrams are simple on purpose — all
> are quick to sketch with a pen.
>
> Write the headings, keep the tables, draw the 4 boxed diagrams. Skip this box.

---

## Word Embeddings — Theory and Analysis

**Prasanna Koirala** · AIAC 536 Natural Language Processing · Kathmandu University

---

### 1. One-Hot Encoding vs Dense Embeddings

A computer cannot work with words, only numbers. So every word must first be turned
into a vector. There are two ways to do it.

**One-hot encoding.** Each word becomes a vector as long as the whole vocabulary — all
zeros, except a single 1 marking that word's position.

Vocabulary = [ cat, dog, king, queen, rocket ]

```
cat    = [ 1, 0, 0, 0, 0 ]
dog    = [ 0, 1, 0, 0, 0 ]
king   = [ 0, 0, 1, 0, 0 ]
```

This has two serious problems.

**Problem 1 — size.** A real vocabulary has ~50,000 words. So every single word becomes
a vector of 50,000 numbers, of which 49,999 are zero. Almost all storage is wasted.

**Problem 2 — no meaning.** Any two different one-hot vectors are exactly the same
distance apart:

> cos(cat, dog) = 0  and  cos(cat, rocket) = 0

The encoding claims *cat* is as unrelated to *dog* as it is to *rocket*. It carries no
information about meaning at all.

**Dense embeddings.** Each word becomes a short vector of real numbers, **learned from
text** rather than assigned by hand.

```
cat = [ 0.21, -0.44,  0.87, ... ]     100 numbers
dog = [ 0.19, -0.40,  0.91, ... ]     100 numbers
```

Because *cat* and *dog* appear in similar sentences, training pushes their vectors close
together. Meaning is now encoded in the numbers.

| | One-hot | Dense embedding |
|---|---|---|
| Length | vocabulary size (~50,000) | fixed and small (50–300) |
| Values | zeros and a single 1 | real numbers |
| Created by | indexing/counting | learned from data |
| Captures meaning | **No** | **Yes** |
| Similar words close | No — all equidistant | Yes |
| Storage | huge, sparse | compact, dense |

---

### 2. Word2Vec, GloVe and FastText

All three rest on one idea, the **distributional hypothesis**:

> *"You shall know a word by the company it keeps."* — J.R. Firth, 1957

Words that appear in similar contexts tend to have similar meanings. So the way to learn
what a word means is to look at the words around it.

#### Word2Vec (2013)

Slides a window across the text and trains a small neural network on it.

Sentence: `the cat sat on the mat`, window size 2, centre word **sat**
Context = [ the, cat, on, the ]

There are two ways to set up the prediction:

```
┌──────── DIAGRAM 1 ────────────────────────────────┐
│                                                   │
│  CBOW      [the, cat, on, the]  ──►  sat          │
│            context predicts the centre            │
│                                                   │
│  SKIP-GRAM      sat  ──►  [the, cat, on, the]     │
│            centre predicts the context            │
│                                                   │
└───────────────────────────────────────────────────┘
```

| | CBOW | Skip-Gram |
|---|---|---|
| Direction | context → centre | centre → context |
| Training speed | faster | slower |
| Rare words | weaker | **better** |
| Small corpus | weaker | **better** |
| Frequent words | good | good |

#### GloVe (2014) — *Global Vectors*

Word2Vec only ever sees one small window at a time — it uses **local** information.

GloVe instead first counts, across the **entire corpus**, how often every pair of words
occurs together. That gives a large co-occurrence matrix. It then factorises that matrix
so the vectors reproduce the co-occurrence counts.

> Word2Vec = local windows.  GloVe = global counts.

#### FastText (2016)

Word2Vec and GloVe treat each word as an indivisible unit. FastText breaks a word into
**character n-grams** and adds them up:

```
"playing"  →  <pl, pla, lay, ayi, yin, ing, ng>

vector("playing") = sum of the vectors of its n-grams
```

This gives two real advantages:

1. **Unknown words.** A word never seen in training — a misspelling, a new name — still
   gets a sensible vector, built from its pieces. Word2Vec and GloVe simply fail here.
2. **Morphology.** *play, played, playing, player* share n-grams, so they automatically
   receive related vectors. This matters enormously for morphologically rich languages
   such as **Nepali**, where one root produces many surface forms.

| Model | Core idea | Main strength |
|---|---|---|
| **Word2Vec** | predict within a local window | fast and simple |
| **GloVe** | factorise global co-occurrence counts | uses whole-corpus statistics |
| **FastText** | word = sum of character n-grams | unseen words + morphology |

---

### 3. Semantic Similarity in Embedding Space

Once words are vectors, "similar meaning" becomes "close together". Closeness is measured
with **cosine similarity**:

```
              A · B
cos(A, B) = ───────────
            |A| × |B|
```

This measures the **angle** between two vectors, ignoring their length.

```
Range:   -1  ─────────  0  ─────────  +1
      opposite      unrelated      identical
```

**Why the angle rather than the distance?** Frequently-occurring words tend to develop
longer vectors. If we used ordinary straight-line distance, a common word would look
"far" from a rare one purely because of frequency, not meaning. The angle discards length
and keeps only direction — and direction is where the meaning lives.

**Real values, measured with GloVe (100 dimensions, 400,000-word vocabulary):**

| Word pair | cos | Comment |
|---|---|---|
| cat — dog | **0.88** | both common pets, appear in the same contexts |
| good — bad | **0.77** | *see the warning below* |
| king — queen | **0.75** | both royalty |
| rocket — spacecraft | **0.61** | related but not interchangeable |
| cat — rocket | **0.19** | unrelated |
| king — banana | **0.16** | unrelated |

**An important warning.** Notice that **good — bad scores 0.77**, almost as high as
*king — queen*. These are opposites, yet the embedding calls them very similar.

Why? Because cosine similarity really measures **"appears in similar contexts"**, not
"means the same thing". *Good* and *bad* slot into identical sentences — "this film was
very ___" — so they end up with nearly the same vector.

This is a genuine limitation of static embeddings: **they capture relatedness, not
synonymy, and they cannot distinguish antonyms from synonyms.**

#### Vector arithmetic

Relationships between words appear as **directions** in the space. The famous example:

```
king − man + woman  ≈  queen
```

```
┌──────── DIAGRAM 2 ────────────────────────────────┐
│                                                   │
│      man  ──────────────►  king                   │
│       │                     │                     │
│       │  "female"           │  "female"           │
│       ▼                     ▼                     │
│    woman  ──────────────►  queen                  │
│                                                   │
│      the horizontal arrow = "royalty"             │
│      both arrows are the same direction & length  │
└───────────────────────────────────────────────────┘
```

The step from *man* to *king* is roughly the same direction and length as the step from
*woman* to *queen*. That shared direction encodes the concept "royalty" — and nobody
programmed it. It emerged from reading text.

**Tested on real GloVe vectors:**

| Arithmetic | Top result | Score |
|---|---|---|
| king − man + woman | **queen** | 0.77 |
| paris − france + nepal | **kathmandu** | 0.81 |

The second one is worth pausing on. The vectors were never told what a capital city is.
They learned the relationship *country → capital* purely from how those words are used in
text, and that same direction transfers correctly from France to Nepal.

---

### 4. The Role of Dimensionality

The dimension **d** is simply how many numbers represent each word.

**If d is too small** (say 5), there is not enough room to separate meanings. Different
concepts are forced to share space and collide. The model *underfits*.

**If d is too large** (say 1000), each dimension gets very little data to learn from, so
the model starts memorising noise instead of meaning. It also costs more memory and is
slower to train. The model *overfits*.

```
┌──────── DIAGRAM 3 ────────────────────────────────┐
│  quality                                          │
│    ▲                                              │
│    │           ______                             │
│    │        __/      \____                        │
│    │      _/               \___                   │
│    │    _/                                        │
│    │  _/                                          │
│    └──┴─────┴──────┴──────┴──────►  d             │
│      10    50    300    1000                      │
│              sweet spot                           │
└───────────────────────────────────────────────────┘
```

| Dimension | Effect |
|---|---|
| Very small (< 20) | underfits — distinct meanings collide |
| **50 – 300** | **usual sweet spot** |
| Very large (> 1000) | overfits, more memory, slower, diminishing returns |

**One important caution.** The individual dimensions are **not interpretable**. Dimension
42 does not mean "royalty" or "animal". Meaning is distributed across the whole vector,
and appears as *directions through the space* rather than along single axes. This is why
embeddings are powerful but hard to explain — a limitation that motivates the
explainability topics later in the course.

---

### Summary

1. One-hot vectors are huge and carry no meaning; dense embeddings are compact and
   learned, so similar words end up close together.
2. Word2Vec learns from local windows (CBOW predicts the centre, Skip-Gram predicts the
   context); GloVe factorises global co-occurrence counts; FastText builds words from
   character n-grams and so handles unseen words and rich morphology.
3. Similarity is measured by the cosine of the angle between vectors, and relationships
   appear as consistent directions, which is why `king − man + woman ≈ queen` works.
4. Dimensionality is a trade-off: too few dimensions underfit, too many overfit, and
   50–300 is the usual range. Individual dimensions carry no meaning on their own.

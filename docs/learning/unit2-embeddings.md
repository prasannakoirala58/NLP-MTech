# Unit 2 — Embeddings, from the ground up

Revision notes for everything covered while doing a3. Written to be re-read cold.
Syllabus Unit 2: Statistical NLP & Text Representation.

---

## The four foundations

**Vector** — a list of numbers. `[2, 3]`. That is the whole definition. It is called a
vector rather than a list only because we do arithmetic on it.

**Embedding** — a word represented as a vector. Both the *process* of assigning the
numbers and the *result* are called the embedding.

**Corpus** — the pile of text you learn from. Nothing more. GloVe's corpus was Wikipedia
plus a news archive, about 6 billion words.

**Dimensions / features** — how many numbers each word gets. GloVe uses 100. These are
also correctly called *features*.

---

## Why words must become numbers

To a computer a word is just characters:

```
"cat" == "dog"     ->  False
"cat" == "rocket"  ->  False
```

Every answer is False. There is **no notion of "how similar"** — only same or not same.

The obvious fix, a lookup table of every pair's similarity, is impossible:
400,000 words means **80 billion pairs**, ~2,500 years to fill by hand, and adding one
new word costs 400,000 more answers.

Storing a small *description* of each word instead: 400,000 × 100 = **40 million
numbers**, **2,000× smaller**, and any pair can be compared on demand. It also lets you
feed words into a neural network, which raw text can never do. That last point is why
every modern NLP system starts here.

---

## How a list of numbers names one location

This is the part that takes longest to click.

The **axes are rulers**, fixed, shared by every word. The **numbers are measurements**
along those rulers. Together they name **one point**.

Each number removes one freedom:

```
before any numbers  ->  could be anywhere
first number fixed  ->  stuck on a plane      (2 freedoms left)
second number fixed ->  stuck on a line       (1 freedom left)
third number fixed  ->  stuck on ONE POINT    (0 freedoms left)
```

100 freedoms minus 100 numbers = 0 freedoms = exactly one point. **You do not need to
picture it, you need to count it.**

The postal-address analogy works well: *Nepal, Kathmandu, Ward 5, Durbar Marg, House 22,
Floor 3, Room 4* — seven numbers, **one room**, not seven rooms. Nobody can picture
7-dimensional address space and the postman still finds the room.

An alternative framing that some people find easier: think of **100 questions on a form**
that every word must answer. `cat` has 100 answers, `dog` has 100 answers, and all
400,000 words answer **the same 100 questions** — which is exactly why they can be
compared. The catch is that nobody wrote the questions; the computer invented them, and
they do not correspond to tidy human ideas.

### Are the dimensions interpretable?

Mostly no. Checked directly on GloVe by listing the highest and lowest scoring words in
individual dimensions:

```
dimension 42   highest: leader, musharraf, sheikh, singh, ahmed
               lowest : researchers, science, sciences, analyst
```

A faint political-versus-academic flavour, but no clean question. Other dimensions are
unreadable mush. **The slots do not need to mean anything** — what matters is only that
every word is measured on the same 100 slots. Like a fingerprint: ridge #42 means
nothing, but two fingerprints still match or do not.

---

## Cosine similarity

One number saying how alike two vectors are.

```
              A . B
cos(A, B) = -----------          -1 opposite  ·  0 unrelated  ·  +1 identical
            |A| x |B|
```

**It measures the angle, not the size.** This trips everyone up:

```
cos([2,3], [3,4])      = 1.00    close-looking numbers
cos([2,3], [200,300])  = 1.00    nothing alike, still PERFECT
cos([2,3], [3,2])      = 0.92    very close numbers, LOWER score
```

`[2,3]` and `[200,300]` sit on the same line out of the origin — same direction, one just
further along. `[2,3]` and `[3,2]` lean differently, so they score lower despite using the
same digits.

**Why ignore size?** Frequent words develop longer vectors (`the` is longer than
`telescope`). If length counted, every common word would look "far" from every rare one
purely because of frequency. The angle strips that out and keeps only meaning.

### Measured on GloVe 100d

| Pair | cos | |
|---|---|---|
| cat — dog | 0.88 | same contexts |
| **good — bad** | **0.77** | **opposites, yet high** |
| king — queen | 0.75 | both royalty |
| cat — rocket | 0.19 | unrelated |

The `good`/`bad` result is the important one: cosine measures **"appears in similar
contexts"**, not "means the same thing". Both fit "this film was very ___". So **static
embeddings cannot separate antonyms from synonyms.**

---

## Where meaning actually lives

Not in the numbers. Here is `cat`'s real vector:

```
[0.23, 0.28, 0.63, -0.59, -0.59, 0.63, 0.24, ...]
```

Meaningless on its own. There is no "animal" number. Meaning appears only on **comparison**:

```
cat  ->  dog 0.88, rabbit 0.74, cats 0.73, pet 0.72, puppy 0.68
```

> **A word's meaning is its position relative to every other word.** The numbers are the
> address; the neighbourhood is the meaning.

---

## The pipeline

```
1. TEXT       billions of words
2. COUNTING   who appears near whom
3. SQUEEZE    each word -> 100 numbers        <- the embedding
4. COMPARE    cosine / arithmetic / neighbours <- meaning lives HERE
5. USE IT     search, translation, RAG, feeding a neural net
```

Steps 1–3 build the map. **Step 4 is the only place anything means anything.**

---

## Word2Vec vs GloVe vs FastText

All three rest on the **distributional hypothesis** — *"you shall know a word by the
company it keeps"* (Firth, 1957).

| | Approach | Note |
|---|---|---|
| **Word2Vec** (Google, 2013) | **predict** neighbours, learn by trial and error | CBOW and Skip-Gram flavours |
| **GloVe** (Stanford, 2014) | **count** co-occurrences over the whole corpus, then factorise | "Global Vectors" — global = whole corpus at once |
| **FastText** (Facebook, 2016) | word = sum of its **character n-grams** | handles unseen words and morphology; matters for Nepali |

Naming trap: **GloVe and FastText are both an algorithm AND the name of the published
file** of pre-computed vectors. "We downloaded GloVe" means we downloaded Stanford's
finished output — we trained nothing.

Character n-grams, concretely:

```
"playing" -> pla, lay, ayi, yin, ing
"played"  -> pla, lay, aye, yed      shares pla, lay
```

---

## Skip-Gram with Negative Sampling (SGNS)

The naming is confusing because four names describe one thing:
**Word2Vec** is the method, **Skip-Gram** is one of its two flavours, **negative
sampling** is the training trick, and **SGNS** is the abbreviation. The instructor's
"Group A expert task" is a fifth name for the same work.

### The algorithm

1. **Make pairs.** Slide a window. Each word near another becomes a pair — a fact that
   these two really appeared together.
2. **Add fakes.** For each real pair, draw a few random impostor words.
3. **Nudge.** Pull the real pair's vectors together; push the fakes' apart.
4. **Repeat** millions of times.

### Why negative sampling exists

The original Skip-Gram asked *"out of all 400,000 words, which is the context?"* — that
means scoring the whole vocabulary on every step. About **20 trillion operations** for a
real corpus. Impossible, which is why Firth's 1957 idea sat unusable for decades.

Negative sampling replaces it with a yes/no question:

> *"Did these two words really appear together?"*

Score the true context plus 5 random impostors. **6 numbers instead of 400,000 — roughly
66,000× less work.** That single change is what made Word2Vec trainable, and it is
Mikolov's actual contribution.

### It knows nothing about meaning

The negatives are drawn **at random** — they are not chosen for being unrelated, nobody
checks. It works purely by frequency: `technology` genuinely appears near `rocket`
thousands of times so it gets pulled thousands of times, while `banana` gets pulled never
and pushed occasionally. **Meaning is a side effect of counting.**

### What "pull closer" means numerically

There is **no target value.** Every number just moves a nudge toward the partner's value
in that slot, so the dot product rises:

```
new value = old value + LR x error x (the other word's value in that slot)
```

Push apart is the identical formula with the sign flipped.

---

## Clusters, PCA, t-SNE

**Cluster** — a bunch of dots sitting together. That is the whole definition.

**PCA** — the words live in 100 directions; paper has 2. Think of a **shadow**: a 3D
object casts a flat shadow, and how useful the shadow is depends on how you turn the
object. PCA finds the angle giving the **most spread**, because spread is information.

**t-SNE** — a different goal. Not an honest shadow; it **bends space** to keep each word
next to its true neighbours.

| | PCA | t-SNE |
|---|---|---|
| Method | honest shadow | bends space |
| Preserves | overall distances | near neighbours only |
| Gaps between clusters | meaningful | **meaningless** |
| Same result twice | yes | no |

**Both are cameras, not tools.** They add nothing — the relationships are already
complete inside the 100 numbers, and cosine reads them exactly. PCA and t-SNE only exist
so human eyes can look, and they *lose* information doing it: our C2 plot retained only
**35% of the variance**, so 65% of the arrangement is simply not in the picture.

The t-SNE trap is worth repeating: **a wide gap between two clusters does not mean they
are especially unrelated.** Only "who is next to whom" is trustworthy.

---

## Findings from our own work

All measured, not quoted.

1. **`good — bad` = 0.77** — nearly as high as king/queen. Static embeddings cannot tell
   antonyms from synonyms.
2. **`paris − france + nepal` = kathmandu (0.81)** — the country→capital direction
   transfers to a country the training never highlighted.
3. **`doctor − man + woman` = nurse (0.77)** — the vectors learned a human bias straight
   out of the text. Relevant to syllabus Unit 9.
4. **`apple` clusters with technology, not food** — one word, two meanings, one vector.
   Exactly what contextual embeddings (BERT) exist to fix.
5. **`bigger − big + small` returns `larger` above `smaller`** — analogies are unreliable,
   for the same antonym reason as finding 1.
6. **Our own SGNS vectors put `king` near `caterpillar` and `gryphon`** — because "king"
   in our corpus is mostly the King of Hearts from *Alice in Wonderland*.

Finding 6 is the one to remember: **an embedding is a mirror of its corpus.** The
algorithm holds no knowledge of its own. Feed it whaling novels and it learns whaling.
Feed it a company's support tickets and it learns that company's language — which is
precisely why domain-specific embeddings are worth training.

---

## Why this is commercially useful

RAG (Retrieval-Augmented Generation) works like this:

```
1. chop a company's documents into chunks
2. embed each chunk                    <- this unit
3. store them in a vector database     <- Qdrant, pgvector, Pinecone, Chroma
4. user asks a question -> embed the question
5. find the nearest chunks             <- cosine similarity
6. hand those chunks to an LLM as context
```

**Only step 6 needs an LLM.** Steps 1–5 are embeddings and cosine similarity.

A **vector database** stores embeddings and finds nearest neighbours fast. A normal
database answers *"find the row where name = X"* (exact match); a vector database answers
*"find the 10 vectors closest to this one"* (similarity match). Postgres cannot do
"closest by meaning" — pgvector, Qdrant, Weaviate and Pinecone can.

---

## Potted history

The idea sat in linguistics books for 34 years before anyone could compute it.

| Year | Who | What |
|---|---|---|
| 1954 | Zellig Harris | words in similar contexts have similar meanings |
| 1957 | J. R. Firth | *"know a word by the company it keeps"* |
| 1988 | Deerwester, Dumais | LSA — first computational version, slow and huge |
| 2003 | Yoshua Bengio | proposes learning vectors with a neural network |
| **2013** | **Tomáš Mikolov** (Google) | **word2vec** — makes it fast via negative sampling; notices `king − man + woman = queen` |
| 2014 | Pennington, Socher, Manning (Stanford) | GloVe |
| 2018 | Devlin et al. (Google) | BERT — a different vector per context, fixing the `apple` problem |

**The breakthrough was not the insight — Firth had that in 1957. It was making it fast
enough to run.** That speed trick is negative sampling, which is what P3 implements.

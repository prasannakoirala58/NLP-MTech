# a3-P3 — Solution trace

**Deliverable:** `PrasannaKoirala_Word2Vec_SGNS.ipynb` (38 cells, runs clean)
Skip-Gram with Negative Sampling, **pure Python + NumPy**, every gradient hand-written.

---

## 1. Corpus: why not the a1/a2 article

The NASA article is **843 words**. Word2Vec learns from repeated co-occurrence; a word
seen twice teaches nothing. So P3 needed a real corpus.

Chose six NLTK Gutenberg books totalling **685,835 tokens** — *Moby Dick*, three Austen
novels, *Alice in Wonderland*, and a Father Brown collection. Picked deliberately for
**distinct vocabularies**, so success is checkable by eye: if the model works, whaling
words should cluster with whaling words and Wonderland characters with each other.

That turned out to be the right call — see §5.

## 2. Pipeline

| Stage | Result |
|---|---|
| Tokenise (lowercase, letters only) | 685,835 tokens |
| Vocabulary, `min_count=8` | **5,617 words** |
| Subsample frequent words (t=1e-3) | 462,994 training tokens |
| Negative table, freq^0.75 | 10M entries |
| Train: dim=100, window=5, neg=5, 3 epochs | **8.35M pairs** in **95s** |

## 3. The one real engineering problem

First version used `rng.choice(V, NEG, p=noise)` to draw negatives. It worked but training
took **359 seconds**, and profiling showed the sampling call was roughly two-thirds of the
total.

**Fix:** the original word2vec trick — build one 10-million-entry array with each word id
repeated in proportion to its probability, then draw by picking a random slot. Measured
**15× faster** on the draw itself.

**359s → 95s, a 3.7× overall speedup**, identical results. Worth knowing that the naive
NumPy call is the bottleneck in any from-scratch implementation.

## 4. Did it learn?

| | |
|---|---|
| Loss at start | 3.200 |
| Loss at end | **2.450** |
| Random guessing baseline | 4.160 |

Comfortably below the random baseline and falling throughout. The baseline is worth
stating explicitly: with one real target and five fakes, a model that knows nothing scores
`-ln(0.5) x 6 = 4.16`. Without that number, "loss = 2.45" means nothing.

## 5. Nearest neighbours — this worked well

```
whale    ->  ship, sperm, line, sea, jaw, boat
ship     ->  pequod, line, boat, sail, whale, main
alice    ->  flambeau, mouse, gryphon, hatter, duchess, dormouse
captain  ->  wentworth, charles, benwick, louisa, admiral
lady     ->  russell, middleton, lucy, colonel, brandon, jennings
```

`pequod` is the ship in *Moby Dick*. `gryphon` and `dormouse` are Wonderland characters.
`wentworth` and `benwick` are from Austen's *Persuasion*. **The model separated the books'
vocabularies without ever being told there were books.**

## 6. Analogies — mostly failed, and that is the honest result

| Asked | Got | Verdict |
|---|---|---|
| `woman - man + he` | indifferent, thoroughly, deserve | failed |
| `queen - king + man` | standing, bald, stood, crooked | failed |
| `ship - sea + land` | hills, cruising, visible | partly worked |
| `night - day + light` | fell, slowly, stern, rose | failed |

**Why.** Analogy arithmetic needs a relationship seen thousands of times to become a
consistent direction. Our corpus is 685k words; the models where `king - man + woman =
queen` was found used **6 billion** — about 10,000× more.

Worse, the corpus works against these specific analogies. **"King" in our books is mostly
the King of Hearts from *Alice in Wonderland***, so its neighbours are *caterpillar*,
*duchess* and *gryphon*. The gender direction barely exists in Austen and Melville.

Reported as failures rather than cherry-picking four that happened to work. Step 10 already
proves the training succeeded; analogies are simply the harder test and the first to break
on a small corpus.

## 7. Ours vs GloVe — the actual lesson

```
whale   ours : ship, sperm, line, sea, jaw, boat
        GloVe: whales, shark, dolphin, humpback, fish, tuna

king    ours : caterpillar, duchess, falconroy, gryphon, dubosc
        GloVe: prince, queen, son, brother, monarch, throne

money   ours : lose, learn, convenience, ones, deemed
        GloVe: funds, cash, fund, paying, pay, paid
```

Three different outcomes, and each says something:

- **`whale`** — ours is *not worse*, it is **different**. We read hundreds of pages of
  whaling, so our *whale* is about harpoons and boats. GloVe read encyclopedia articles,
  so its *whale* is about marine biology. Neither is wrong.
- **`king`** — ours is genuinely wrong for general English, and the reason is precisely
  traceable to *Alice in Wonderland*.
- **`money`** — ours is weak noise. 19th-century novels rarely discuss money in the
  concrete way news does, so there was little to learn from.

**An embedding is a mirror of its corpus.** The algorithm holds no knowledge of its own.
This is exactly why domain-specific embeddings are worth training: feed it a company's
support tickets and it learns that company's language, which no general model has seen.

## 8. Design decisions worth defending

- **Two vector tables** (`W_centre`, `W_context`). "A appears near B" is not symmetric, so
  each word needs both roles. `W_context` discarded at the end, as in the original paper.
- **`W_context` initialised to zero**, `W_centre` to small random values. Every initial
  score is then 0 and every prediction 0.5 — a model that starts by admitting ignorance.
- **Random window size** per centre word (1..WINDOW). Weights near neighbours more
  heavily, since they survive every window size.
- **Subsampling** frequent words. Without it, *the* dominates the gradients while carrying
  almost no information about its neighbours.
- **freq^0.75** for negatives. Flattens the distribution so impostors are not almost always
  obscure words.

## 9. Verification

All 9 requirement checks pass programmatically in the final cell, including an explicit
assertion that no deep-learning framework is in scope.

## Outputs

| File | |
|---|---|
| `outputs/PrasannaKoirala_SGNS_vectors.npz` | trained vectors, committed |
| `outputs/PrasannaKoirala_SGNS_vectors.txt` | word2vec text format, gitignored — regenerates in 95s |

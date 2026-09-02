# a1 — Solution trace

How this assignment was actually solved, including the decisions and the things that
were not obvious at the start.

**Deliverable:** `PrasannaKoirala_POS_01.ipynb` (run top to bottom, all outputs saved)

---

## 1. Choosing the article

The brief says "any English-language news article from a reputable news source". Three
criteria drove the choice beyond that:

- **Short.** ~850 words. Long enough for meaningful counts, short enough to read the
  tagger's entire output by eye and actually check it. On a 5,000-word article you stop
  checking and start trusting, which defeats the purpose.
- **Clean prose.** No tables, bullet lists, or heavy markup to strip out. Preprocessing
  overhead is not what this assignment is teaching.
- **Entity-dense**, because a2 (NER) reuses the same article.

Chosen: a NASA press release on the Nancy Grace Roman Space Telescope launch.

**Unplanned bonus:** NASA press releases are US Government works and therefore public
domain. Reproducing the full text in a submitted assignment raises no copyright question
at all. Most news outlets are copyrighted; this sidesteps it entirely. Worth remembering
for future assignments.

## 2. Tool choice

spaCy `en_core_web_sm` as the primary tagger, NLTK's averaged perceptron as a
cross-check. The brief requires only one; we ran both under the repo's standing "do
everything" rule.

The comparison earned its place — see §4.

## 3. The decision the brief hides

"Extract all Nouns and Verbs" sounds unambiguous. It is not.

spaCy uses **four** relevant coarse tags, not two:

| Tag | What it is | Count in this article |
|---|---|---|
| `NOUN` | common noun (telescope, mission) | 215 |
| `PROPN` | **proper** noun — a name (NASA, Florida) | 159 |
| `VERB` | main verb (launched, explore) | 75 |
| `AUX` | **auxiliary** verb (will, is, could) | 35 |

So: do proper nouns count as nouns? Do auxiliaries count as verbs?

**Decision: include both, but label them** via a `POS_Tag` column recording the exact
tag. A reader who wants strict `NOUN`/`VERB` can filter; nothing is lost. Silently
excluding them would discard real data without telling anyone.

**This decision is worth 43% of the nouns.** Proper nouns are 159 of 374. That is not a
rounding error — it is nearly half the answer. The cause is the subject matter: a space
mission article is saturated with names (NASA, Roman, Falcon Heavy, Goddard, Florida,
Maryland). An opinion column would look completely different.

The lesson generalises: *check the tag distribution before assuming what a category
contains.*

## 4. What the NLTK cross-check bought

This is where the assignment got interesting, and where it produced a real bug.

### The bug: you cannot compare two taggers position by position

First attempt compared the two taggers by walking both token lists side by side with
`zip()`. It reported **0 disagreements** — which is impossible, because the two taggers
reported different totals (373 vs 368 nouns).

The cause: **the tokenisers do not agree on what a word is.** The article contains
"Dark Universe-Seeking", and:

- spaCy splits it into 3 tokens: `Universe` + `-` + `Seeking`
- NLTK keeps it as 1 token: `Universe-Seeking`

From token 3 onward the two lists are offset. Only **10 of 973** positions still lined
up, so the comparison silently compared unrelated words and found nothing.

**Fix:** align the sequences properly first, using `difflib.SequenceMatcher` to find
matching blocks and comparing tags only inside them. That lifted coverage from ~1% to
**95%** (948 of 998 tokens).

The lesson is bigger than this assignment: *a comparison that returns a suspiciously
clean answer is a reason to check the comparison, not to celebrate.* A zero here looked
like agreement; it was actually a broken harness.

### The verified result

| | |
|---|---|
| Tokens aligned | 948 of 998 (95.0%) |
| Tags agree | 895 (94.4%) |
| Tags disagree | 53 (5.6%) |

### Two different kinds of disagreement

The patterns split cleanly into two categories, and conflating them would be a mistake:

**Kind 1 — tagset convention differences.** The largest single pattern is `IN` vs `TO`
(8 instances), and every one is the word "to". NLTK gives infinitival "to" its own tag;
spaCy files it under preposition. Neither is wrong — the tagsets just draw the boundary
differently. Same for `RB` vs `IN` on "about"/"before". **Counting these as tagger
errors would be misleading.**

**Kind 2 — genuine linguistic ambiguity.** The `VB` vs `NN` disagreements are the real
ones. The words: **image, power, probe, survey** — every one can be a noun or a verb
depending on the sentence:

- "Roman's Coronagraph will **image** Earth-like planets" → verb
- "It will **survey** the universe" → verb

This is precisely the ambiguity used to introduce POS tagging at the top of the
notebook ("launch" as noun vs verb), turning up unprompted in real data. Also `JJ` vs
`NNP` on "Roman" — proper noun, or adjective modifying "telescope"? No clean answer.

**Why this justified running two taggers.** A single tagger reports a confident tag for
every token and hides all of this. Running two makes the model's uncertainty visible,
and separates "the tagsets disagree" from "the English is ambiguous".

## 5. One real bug caught

The extraction loop skips tokens where `is_space or is_punct`. That guard turned out to
matter: spaCy tags a stray hyphen `-` as `NOUN`. Without the guard it would have landed
in the CSV as an extracted noun.

Recon counted 374 nouns; the CSV has 373. The difference is exactly that hyphen. Worth
verifying rather than assuming a miscount — a one-row discrepancy is precisely the kind
of thing that is either trivial or a real bug, and you cannot tell without looking.

## 6. Output format

One row per **occurrence**, not per unique word — no information is discarded, and
frequency counts can be derived from occurrences but not the reverse.

| Column | Why |
|---|---|
| `Word` | required by the brief |
| `POS_Tag` | required by the brief; coarse tag, makes the PROPN/AUX decision auditable |
| `Fine_Tag` | Penn Treebank tag — distinguishes singular/plural, tense |
| `Lemma` | dictionary form; lets "launched"/"launches"/"launching" be counted together |
| `Category` | our Noun/Verb grouping, for easy filtering |
| `Sentence_No` | traces each word back to its sentence in the source |

**Result:** 483 rows — 373 nouns, 110 verbs.

## 7. Verification

The notebook's final cell checks all ten requirements programmatically rather than
asserting them in prose — file loaded, tagging performed, both categories extracted, CSV
written, required columns present, row count matches, no empty cells, filename correct.
All pass.

---

## Carried forward to a2

- Same article, already saved and cited — no sourcing work needed
- The `PROPN` density (43% of nouns) predicts a rich NER result, since proper nouns are
  largely what NER operates on
- spaCy tagged **"Prasanna" as ORG** in an early smoke test, which is wrong — it should
  be PERSON. Small models handle non-Western names poorly. That is a concrete, honest
  finding for a2's "compare two NER models and discuss" bonus.

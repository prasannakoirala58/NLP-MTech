# a3-P4 — Solution trace

**Deliverable:** `PrasannaKoirala_Pretrained_Embeddings.ipynb` (34 cells, runs clean)

---

## 1. The experiment

Sentiment classification on NLTK's Movie Review corpus — 2,000 reviews, perfectly balanced
1,000 positive / 1,000 negative. One classifier, three input representations, one fixed
train/test split.

| Arm | Input | Why |
|---|---|---|
| 1 | Bag of words (TF-IDF) | the "without embeddings" baseline the brief asks for |
| 2 | Our own SGNS vectors from P3 | does the from-scratch model work on a real task? |
| 3 | GloVe, 6B words | what professional embeddings buy |

Arm 2 is beyond the brief. It exists because P3 produced real vectors and this is the only
place they can be tested against a measurable outcome.

**Classifier: logistic regression** — one layer of weights and a sigmoid, the simplest
neural network there is. Chosen deliberately: a larger network could compensate for poor
input and blur the comparison. A weak classifier lets input quality show through, which is
the point when the input is what is on trial.

## 2. Result

| Input | Accuracy on 400 unseen reviews |
|---|---|
| **Bag of words — no embeddings** | **82.0%** |
| GloVe (6B words) | 72.8% |
| Our SGNS (685k words) | 64.8% |
| random guessing | 50.0% |

**The method with no embeddings won, by 9 percentage points.**

This is the opposite of the expected result and is reported as it came out. Picking a task
where embeddings win would have been easy and dishonest.

## 3. Testing whether the comparison was fair

The obvious objection: bag of words got **20,000 features**, the embeddings only **100**.
Maybe it simply won on budget. So it was capped and re-run.

| features | accuracy |
|---|---|
| **100** | **74.0%** ← same budget as the embeddings |
| 300 | 78.8% |
| 1,000 | 82.8% |
| 20,000 | 82.0% |

**At an identical 100-feature budget bag of words still edges GloVe, 74.0% vs 72.8%.**
The win is real, not an artefact of budget. Running this control before drawing the
conclusion was the difference between a claim and a finding.

## 4. Why the embeddings lost

**Averaging destroys the signal — the main cause.** Sentiment lives in a handful of
decisive words: *dreadful*, *masterpiece*, *boring*. Reviews run to several hundred words,
so averaging buries those few strong signals under hundreds of neutral ones — *the*,
*film*, *scene*, *plot*. Bag of words keeps every word as its own feature, so one
appearance of *dreadful* stays visible; averaging dilutes it several-hundred-fold.

**The task flatters bag of words.** Sentiment is close to keyword spotting. Knowing that
*brilliant* relates to *wonderful* — precisely what embeddings add — buys little when
spotting *brilliant* already nearly settles the answer. On a task needing generalisation
to unseen words the ranking would likely reverse: bag of words is helpless with a word it
never saw in training, while an embedding places it sensibly from its vector alone.

**Coverage explains our own model's last place.**

| | review words recognised |
|---|---|
| our SGNS | **65.7%** |
| GloVe | **99.7%** |

Every unrecognised word is silently dropped before averaging, so **roughly a third of each
review is discarded** in Arm 2. Words GloVe knows and ours does not include *blandness,
irrelevant, musician, landslide, substantially* — ordinary modern English absent from
Austen, Melville, Carroll and Chesterton.

## 5. Guarding against leakage

Raised as a concern before implementation, and worth recording how each was handled:

1. **Vocabulary built from all 2,000 reviews** would let the test set shape the setup.
   → Split first; `fit_transform` on train, `transform` only on test.
2. **Peeking at the test score and tuning** slowly fits the model to the test set.
   → All settings fixed up front, scored once.
3. **A different split per arm** would make the comparison meaningless.
   → One seed, identical split reused by all three arms.

**One honesty note that cannot be engineered away:** GloVe was trained on 6 billion words
of internet text which very likely includes film reviews. That is not leakage of our 400
test reviews specifically, but GloVe arrives already knowing how people write about films.
That is the *purpose* of pre-trained embeddings rather than a flaw — but it means the
comparison measures "how much does having read the internet help", not "which algorithm is
better". Stated in the notebook rather than glossed over.

## 6. The conclusion worth keeping

**Embeddings are not automatically better.** They are better *for problems that need
meaning*. This problem mostly needs keyword detection, and the simplest possible method
won.

Choosing a representation means matching it to the task, not reaching for the most
sophisticated option available. That is a more useful thing to have learned than a tidy
result confirming what everyone expects.

# a2 — Solution trace

**Deliverable:** `PrasannaKoirala_NER_01.ipynb` (48 cells, runs clean, outputs saved)

---

## 1. Same article as a1, deliberately

The brief allows any article. We reused a1's NASA piece so the two techniques run over
**identical text**, which makes the POS-vs-NER comparison meaningful instead of anecdotal.
No sourcing work was needed.

a1 also predicted this would work well: `PROPN` was 43% of all nouns there, and proper
nouns are precisely what NER operates on.

## 2. What NER actually produced

| | |
|---|---|
| Entities found | 116 |
| Unique entity texts | 79 |
| Entity types present | 11 |

Types found: `ORG` (45), `GPE` (13), `PERSON` (12), `DATE` (10), `NORP` (10), `TIME` (6),
`CARDINAL` (6), `ORDINAL` (5), `LOC` (4), `FAC` (3), `QUANTITY` (2).

## 3. The structural difference from a1

POS tagging labels **one token at a time**. NER groups **several tokens into one entity**:

- `Kennedy Space Center` — 3 tokens, 1 entity
- `$500 million` — 3 tokens, 1 entity

So the CSV has one row per **entity**, not per word. This is not a cosmetic difference —
splitting "Kennedy Space Center" apart would turn a facility into a person.

## 4. The conceptual relationship: NER refines PROPN

The clearest result in the notebook. On one sentence:

| Word | POS says | NER says |
|---|---|---|
| NASA | `PROPN` | `ORG` |
| Kennedy | `PROPN` | `FAC` |
| Florida | `PROPN` | `GPE` |
| Sunday | `PROPN` | `DATE` |

**POS gave all four the identical label.** It can say "this is a name" and nothing more.
NER says *which kind* of name. NER is a refinement of `PROPN` — it sorts the pile that
POS tagging could only lump together.

## 5. The model comparison, and the trap in it

Compared spaCy `en_core_web_sm` against `en_core_web_md`. Same architecture, same 18
labels, so any difference is model capacity rather than differing schemes.

**The trap:** both models found **exactly 116 entities**. It would be easy to conclude
they agree. They do not — identical counts, different content.

Comparing properly, by character position in the text:

| | |
|---|---|
| Found by both, same label | 90 |
| Found by both, **different** label | 10 |
| Found only by small | 16 |
| Found only by medium | 16 |
| **Full agreement** | **68.2%** |

Two models can produce the same total while agreeing on barely two-thirds of the actual
entities. *Comparison must be done by position, never by count.* This is the same lesson
as a1's alignment bug, in a different disguise.

## 6. The best finding: "Roman"

The disagreements are not spread evenly. They concentrate on one word.

"Roman" appears **20 times**. The small model labelled it **three different ways** across
those 20 occurrences:

    NORP, ORG, PERSON

And it is inconsistent *within the same article* — occurrence 3 is `ORG`, occurrence 4 is
`NORP`, occurrence 17 is `PERSON`. The medium model is inconsistent too, and the two
models differ on 7 of the 20.

**Why this word is genuinely hard.** The telescope is named after Nancy Grace Roman, an
astronomer. So "Roman" is simultaneously:

- a person's surname → `PERSON`
- the name of a mission/instrument → `ORG`
- an ordinary English word meaning "from Rome" → `NORP` (nationalities)

**The honest conclusion is not "the bigger model is better".** Model size did not resolve
the ambiguity, because the ambiguity is in the text, not in the model. That is a more
useful result than a clean win would have been.

## 7. Error analysis on Nepali names — and a correction

During a1 an early smoke test tagged bare **"Prasanna"** as `ORG`, and that was noted as a
likely a2 finding. Testing it properly here refines that:

| Input | small | medium |
|---|---|---|
| "Prasanna Koirala studies NLP at Kathmandu University." | `PERSON` ✅ | `PERSON` ✅ |
| "Prasanna studies at Kathmandu University." | less reliable | less reliable |

With the **full name**, both models get it right — the surname supplies enough signal. The
earlier note was based on a first-name-only test and overstated the problem.

The real limitation stands but is narrower than first claimed: these models are trained
largely on Western news text, so non-Western names are recognised less reliably,
especially without surrounding context. Directly relevant to syllabus Unit 7 (Multilingual
NLP, Nepali/Indic case studies).

## 7a. Error analysis: the headline breaks NER three separate ways

Reading the `displacy` output revealed two bad entities at the very start. Chasing them
down produced the strongest analysis in this assignment, because the cause is general.

### Problem 1 — the model invents entities in headlines

Running NER on the headline alone:

| Found | Labelled | Verdict |
|---|---|---|
| NASA | `ORG` | correct |
| Dark Universe-Seeking | `PERSON` | **wrong** — a descriptive phrase, not a person |
| Nancy Grace Roman Space | `PERSON` | **wrong span** — truncates "...Space Telescope" |

Two of three wrong. "Dark Universe-Seeking" names nothing; the model fabricated it.

**Why:** headlines are not sentences. Title Case On Every Word defeats capitalisation,
which is one of the strongest cues a model uses to spot names. There is no main verb.
"Dark Universe-Seeking" is a stacked modifier that barely occurs in ordinary prose. NER
models are trained on running prose, so a headline is out of distribution.

### Problem 2 — rewriting does not fix it, it inverts the error

Testing the theory by restating the same facts as an ordinary sentence:

> "NASA launched the Nancy Grace Roman Space Telescope, which seeks to study the dark universe."

The hallucinated `PERSON` disappears — which supports the diagnosis. But the model now
finds **only** `NASA`, missing the telescope completely.

| | Headline | Ordinary sentence |
|---|---|---|
| Failure | **over-detects**, invents entities | **under-detects**, misses a real one |

Neither is correct. Recording this honestly matters more than claiming a fix.

### Problem 3 — the headline has no full stop, so sentence splitting breaks

The headline ends "...Telescope Launches" with no terminal punctuation. Sentence splitters
break on punctuation, so spaCy never breaks there and glues the headline onto the first
body paragraph as a single "sentence".

Consequences: the `Sentence_No` column is skewed (sentence 1 is really headline +
paragraph 1), and headline text is read continuously with body text — the exact
genre-mixing that caused Problem 1.

**Deliberately not fixed.** The brief says to load and process the article as saved.
Silently editing the source would misrepresent how the model behaves on real input.
Documenting the limitation is the more honest choice, and the fix (split headline from
body before tagging) is stated in the notebook.

### The deepest problem — the label set has no correct answer

Even in clean body prose, "Nancy Grace Roman Space Telescope" is tagged `PERSON`.

That is not absurd: the telescope **is** named after Nancy Grace Roman, NASA's first
Chief of Astronomy. But what *should* it be? Against spaCy's 18 types:

- `PERSON` — named after one, but it is a machine
- `ORG` — not an organisation
- `PRODUCT` — closest, but built once for science
- `FAC` — facilities are buildings; this one is a million miles from Earth

**No correct label exists.** The category is missing from the scheme. This finally explains
the "Roman" inconsistency in §6: the models are not choosing badly among good options,
they are choosing among options that are all partly wrong. That is a limitation of the
**label set**, not of the models.

## 8. Bonus items — both done

- **`displacy` visualisation** — first 3 sentences rendered inline with colour-coded
  entities. Verified present as HTML output in the executed notebook.
- **Two-model comparison** — §5 and §6 above.

## 9. Output format

| Column | Why |
|---|---|
| `Entity` | required by the brief |
| `Entity_Type` | required by the brief |
| `Description` | plain-English meaning of the type code |
| `Token_Count` | how many words the entity spans — makes multi-word entities visible |
| `Sentence_No` | traces each entity back to its sentence |

The brief specifies `.csv` **only** for the entity list. The summary is listed as a
deliverable with **no format given**, so a second CSV is not required.

- `PrasannaKoirala_NER_01.csv` — 116 rows, one per entity — **required as .csv**
- `PrasannaKoirala_NER_01_summary.csv` — counts by type — **convenience only**

The real summary lives in the notebook, where the table sits beside a bar chart and a
written interpretation. "A brief summary" implies something you read, not a data file.
The CSV is kept as redundancy so a grader browsing `outputs/` finds it without running
anything.

## 10. Verification

All 12 checks pass programmatically in the notebook's final cell, including both bonus
items.

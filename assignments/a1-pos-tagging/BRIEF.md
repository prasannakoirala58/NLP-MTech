# a1 — POS Extraction from a News Article

Source: `questions/Assignment_1.docx.pdf` (Assignment 01) · Syllabus Unit 1 · Lab 1

## Tasks
- [x] Download an English news article from a **reputable** source
- [x] Save the article content as plain text `.txt`
- [x] Load the text file into Python / Jupyter
- [x] Perform **POS tagging**
- [x] Extract **all Nouns and Verbs**
- [x] Save extracted words to CSV with appropriate columns (e.g. `Word`, `POS_Tag`)
- [x] Submit as a Jupyter Notebook

## Deliverables
- [x] `PrasannaKoirala_POS_01.ipynb`
- [x] `data/article.txt` — the original article
- [x] `outputs/PrasannaKoirala_POS_01.csv`

## Evaluation criteria — verify each before submitting
- [x] Correct loading and processing of the text file
- [x] Accurate POS tagging and extraction of nouns and verbs
- [x] Proper CSV export
- [x] Code readability, documentation, and organisation
- [x] Adherence to the required file naming convention

## Decisions made (see SOLUTION.md for reasoning)
- **Tagger:** spaCy `en_core_web_sm` as primary, NLTK as cross-check. After proper
  sequence alignment: 94.4% agreement, 53 disagreements of two kinds — tagset
  convention (`IN` vs `TO`) and genuine ambiguity (`VB` vs `NN` on image/power/
  probe/survey).
- **PROPN and AUX are included**, and labelled in the `POS_Tag` column so they can be
  filtered out. Excluding PROPN would have dropped 43% of the nouns.
- **Every occurrence kept** (483 rows: 373 nouns, 110 verbs), with frequency analysis
  done in the notebook rather than baked into the CSV.

## Status: COMPLETE
All 10 requirement checks pass in the notebook's final cell.

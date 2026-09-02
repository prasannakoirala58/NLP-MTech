# a2 — Named Entity Recognition from a News Article

Source: `questions/Assignment_1.docx.pdf` (Assignment 02) · Syllabus Unit 1 · Lab 1

## Tasks
- [x] Download an English news article from a **reputable** source
- [x] Save the article content as plain text `.txt`
- [x] Load the text file into Python / Jupyter
- [x] Perform **NER** using spaCy / NLTK / HF Transformers
- [x] Extract all named entities **and their types** (PERSON, ORG, GPE, DATE, MONEY, …)
- [x] Save to CSV with columns `Entity`, `Entity_Type`
- [x] **Provide a summary of the different entity types found**
- [x] Submit as a Jupyter Notebook

## Deliverables
- [x] `PrasannaKoirala_NER_01.ipynb`
- [x] `data/article.txt` — the original article
- [x] `outputs/PrasannaKoirala_NER_01.csv`
- [x] Brief summary of entity counts by category

## Example CSV format (from the brief)
| Entity | Entity_Type |
|---|---|
| Donald Trump | PERSON |
| United States | GPE |
| Microsoft | ORG |
| July 20, 2026 | DATE |

## Evaluation criteria — verify each before submitting
- [x] Correct loading and preprocessing of the text file
- [x] Accurate implementation of NER
- [x] Proper extraction and labelling of entities
- [x] Correct CSV export and formatting
- [x] Quality of code, documentation, and analysis
- [x] Adherence to the required file naming convention

## Bonus (optional — worth doing)
- [x] Visualise entities with spaCy `displacy`
- [x] Compare **two different NER models** and discuss the results
      (e.g. spaCy `en_core_web_sm` vs HF `dslim/bert-base-NER`)

## Decisions made (see SOLUTION.md for reasoning)
- **Same article as a1**, so POS and NER run over identical text and can be compared.
- **Two models compared:** spaCy `en_core_web_sm` vs `en_core_web_md`. Same label scheme,
  so differences are model capacity, not scheme mismatch.
- **Compared by character position, not by count.** Both models found exactly 116
  entities but agreed on only 68.2% of them.
- **Two CSVs**, since the summary is its own required deliverable.

## Status: COMPLETE
All 12 requirement checks pass, including both bonus items.

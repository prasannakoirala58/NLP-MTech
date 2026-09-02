# Article source

| | |
|---|---|
| **Title** | NASA's Dark Universe-Seeking Nancy Grace Roman Space Telescope Launches |
| **Publisher** | NASA (National Aeronautics and Space Administration) |
| **URL** | https://www.nasa.gov/news-release/nasas-dark-universe-seeking-nancy-grace-roman-space-telescope-launches/ |
| **Retrieved** | 2026-09-01 |
| **File** | `data/article.txt` |

## Why this article

- **Reputable source** — NASA is the primary source for the story, not a syndicated rewrite.
- **Public domain.** NASA news releases are US Government works, so there is no copyright
  restriction on reproducing the text in a submitted assignment. Most news outlets are
  copyrighted; this avoids the issue entirely.
- **Short** — ~700 words, ~13 paragraphs. Big enough for meaningful counts, small enough to
  read the tagger's full output by eye and actually check whether it is right.
- **Clean prose** — plain declarative sentences, no tables, lists, or heavy markup.
- **Entity-dense**, which matters because a2 (NER) reuses this same article:
  PERSON (Jared Isaacman, Nicky Fox, Julie McEnery), ORG (NASA, SpaceX, ESA, JAXA, CNES,
  BAE Systems, L3Harris, Caltech), GPE (Florida, Maryland, Washington, Australia, Spain,
  Germany), DATE/TIME (7:26 a.m. EDT, 31 minutes, early 2027), QUANTITY (1.4 terabytes,
  300-megapixel, million-mile).

## Note for the notebook

The text is stored **verbatim**, including curly apostrophes (`'` U+2019) in words like
`NASA's` and `Roman's`. This is deliberate: handling real-world punctuation is part of the
preprocessing step, and it is worth seeing how the tokeniser splits `NASA's` into
`NASA` + `'s`.

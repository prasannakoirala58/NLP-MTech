# a5 — Solution trace

**Deliverables:** `PrasannaKoirala_Transformer.ipynb` (83 cells, 73/73 checks) + handwritten notes
copied from `DRAFT.html` (≈ 22–24 pages, scan pending) + the GitHub link.

---

## 1. How the work is split

The brief asks for two things that are easy to blur: a notebook, and handwritten notes covering
"derivations, computations, algorithms, and architecture diagrams". So the rule was:

- **Notebook** = every function, every number, every plot, every check. No derivations.
- **Handwritten** = maths, words and simple diagrams. **No code, ever.**
- Interpretation parts (1d, 3d, 5d, 8c) go in both: a few lines by hand, the evidence in code.

All 37 parts are mapped one by one in `BRIEF.md` §3. Their marks sum to exactly 100.

## 2. Three layers of verification

Nothing reaches paper unless it has been checked three separate ways.

| Layer | What | Result |
|---|---|---|
| 1. Before writing | every hand answer computed three independent ways: NumPy, 40-digit `Decimal` with its own Taylor-series `exp/sin/cos`, and a simulated paper path that rounds each step | all agree |
| 2. Notebook | every hand value compared with Python (`compare()`), plus behavioural tests: rows sum to 1, future weights exactly 0, no future leak, FFN per token vs attention mixing, per-head rank, memory vs real allocation | **73/73** |
| 3. Draft audit | the notebook exports `outputs/verified_values.json` (227 values); `check_draft_numbers.py` checks every decimal in `DRAFT.html` against it | **every number matches** |

The audit was tested by planting two errors (0.504 for 0.503, a digit swap in sin 1). It caught both.
A fresh-kernel re-run reproduces all 227 values exactly.

## 3. Findings worth the marks

**Q3d: the supplied projections give the same weights as identity.**
The two score matrices differ by
`s_ij(identity) − s_ij(supplied) = (x_i1 − x_i2)(x_j1 − x_j2) = (−1)(−1) = 1`. That is a constant
shift in every row, and softmax ignores constant shifts (the same identity Q7a proves). So who
attends to whom is unchanged and only W_V changes the output. The obvious answer, "projections
changed the relationships", is wrong for these matrices. A random W_K in the notebook shows the
general case, where they do change.

**Q6b: the two heads are identical.** `QKᵀ = X P Pᵀ Xᵀ = X Xᵀ` because `PPᵀ = I`. This became Q6d's
evidence, together with two measured facts: the parameter count is 256 for H = 1, 2, 4 and 8, and
one head's score matrix has rank exactly d_k (8 → 4 → 2 → 1).

**Q10: n² only takes over past n = 2·d_model = 128.** Below that, the projections (4nd²) cost more
than attention (2n²d). Measured log-log slopes rise from about 0.9 to 2.0, against 1.26 → 1.74 for
the pure arithmetic. The gap is the implementation: Python overhead at small n, memory traffic at
large n. That is exactly Q10c's "arithmetic vs implementation" distinction, measured rather than asserted.

**Rounding.** Q4b's third-row weight is 0.50349, which rounds to **0.503**. Rounding twice
(0.5035 → 0.504) is wrong. Working at 3 decimals turns the matching output into 0.751 instead of
0.752. Hence the rule in the notes: 4 decimals while working, round once at the end.

## 4. Mistakes made, and how they were caught

- **0.504 for 0.503.** My first pre-computed table double-rounded it. The three-way check caught it.
- **40 parts.** I said 40; it is 37. Counting the brief programmatically caught it.
- **e^1.4142 = 4.1133.** It is 4.1132 (4.1133 is e^√2). An independent audit caught it.
- **Q10 explanation.** I first blamed the low slope at small n on fixed overhead. The audit showed
  the arithmetic itself (the projection term) is the main reason. Corrected.
- **Q6 toy wiring.** Its heads are 2 wide, so the concat is 2 × 4 while d_model is 2. The fix: compute
  the head size from `W_Q.shape[1] // num_heads`, and check the concat with W_O = I₄.
- **Causal mask.** `np.triu` could count as a "prebuilt masking function", so the mask is built by comparing positions (`j > i`).
- **Learning notes.** The first version was 12 pages with maths, too dense to learn from. Rewritten
  as 6 pages of everyday scenarios with no formulas.
- **Temperature plot.** A 1 × 4 strip was too thin to read inside a handwritten figure box. Redrawn as 2 × 2.

## 5. Tooling notes

- Chrome is no longer installed on this Mac, so both render scripts fall back to Brave (any Chromium works).
- `uv run` tried to re-sync and download a spaCy wheel over a slow network, so the notebook is
  executed with `uv run --no-sync`. Nothing is installed by hand.

## 6. Still to do

The scan, pasting the three plots into the ruled boxes (`insert_figures.py`), the final
`DRAFT.pdf`, the README, a final independent validation, then submission.

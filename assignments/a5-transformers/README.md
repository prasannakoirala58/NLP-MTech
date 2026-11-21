# a5 — Transformers from Scratch (NumPy only)

**AIAC 536 Natural Language Processing** · Kathmandu University · Transformer Assignment (100 points)
**Prasanna Koirala** · Roll No. 20016

## Deliverables

| File | Contents |
|---|---|
| [`PrasannaKoirala_Transformer.ipynb`](PrasannaKoirala_Transformer.ipynb) | All 10 questions in NumPy: implementations, intermediate outputs, shape checks, 73 verification tests, 3 plots |
| [`PrasannaKoirala_Transformer.pdf`](PrasannaKoirala_Transformer.pdf) | Scanned handwritten notes (43 pages): derivations, hand calculations, algorithms, architecture diagrams |

The two are linked: every value calculated by hand in the notes is recomputed in the
notebook and compared (`compare(...)` cells), so each hand calculation has a Python check.

## Run the notebook

NumPy and Matplotlib only: no PyTorch, TensorFlow, Keras, JAX or prebuilt attention layers.
Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev --group embed
uv run jupyter lab        # open the notebook, then Run All
```

The last cell prints `All checks: 73/73 passed`.

## What each question covers

| Q | Topic | In the notebook |
|---|---|---|
| 1 | Sinusoidal positional encoding | `sinusoidal_positional_encoding`, heatmap, rotation property |
| 2 | Scaled dot-product attention | `scaled_dot_product_attention`, manual softmax |
| 3 | Learnable Q, K, V projections | `self_attention`, identity vs supplied projections |
| 4 | Causal masking | `causal_mask` built by comparing positions, `causal_self_attention` |
| 5 | Cross-attention | `cross_attention`, non-square n × m weights |
| 6 | Multi-head attention | `multi_head_attention`, per-head rank, parameter count |
| 7 | Numerical stability, temperature | `stable_softmax`, temperature plot, softmax Jacobian |
| 8 | Encoder block | `layer_norm`, `feed_forward`, `transformer_encoder_block` |
| 9 | Decoder block, full Transformer | `transformer_decoder_block`, no-future-leak test |
| 10 | Complexity | `benchmark_attention`, time and memory vs sequence length |

## Other files

| File | |
|---|---|
| [`BRIEF.md`](BRIEF.md) | The assignment as a part-by-part checklist (37 parts, 100 marks) |
| [`SOLUTION.md`](SOLUTION.md) | How it was solved: verification, findings, mistakes caught |
| `DRAFT.html` / `DRAFT.pdf` | Typeset draft the handwritten notes were written from |
| `check_draft_numbers.py` | Checks every number in the draft against the notebook's `outputs/verified_values.json` |
| `make_question_images.py` | Cuts each question out of the assignment PDF (`question-images/`) |
| `insert_figures.py` | Builds the final notes PDF: pastes the questions and plots into the scan, compresses it |
| `render.sh` | Rebuilds `DRAFT.pdf` |
| `outputs/` | The three plots and `verified_values.json` |

Plain-language study notes for this unit: [`docs/learning/unit4-transformers.pdf`](../../docs/learning/unit4-transformers.pdf).

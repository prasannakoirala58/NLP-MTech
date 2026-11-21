# a5 — Transformers from scratch in NumPy

> **This file is the plan.** One step at a time: do it → pass its check → tick it →
> pause and make sure it's understood → next step.

| | |
|---|---|
| **Source** | `questions/a5-transformers/` (assignment PDF + Classroom post) · Syllabus Unit 4 |
| **Classroom** | "Transformer Assignment" · **100 pts** · posted 12 Oct · due 18 Nov, **extended to Sun 22 Nov, end of day** |
| **Graded on** | maths correctness · tensor dimensions · independent implementation · **interpretation**, not just code that runs |
| **Also** | **10 random students present in class.** You must be able to explain every line. |

---

## 1. Deliverables & rules
| # | Deliverable | File |
|---|---|---|
| 1 | Notebook: NumPy code, intermediate outputs, shape checks, tests, plots | `PrasannaKoirala_Transformer.ipynb` |
| 2 | Scanned handwritten notes: derivations, computations, algorithms, diagrams | `PrasannaKoirala_Transformer.pdf` |
| 3 | GitHub link (repo is public ✅) | `github.com/prasannakoirala58/NLP-MTech/tree/main/assignments/a5-transformers` |

- [ ] **NumPy + Matplotlib only.** No PyTorch / TF / Keras / JAX, not even as a reference check.
- [ ] **Nothing prebuilt:** no library softmax, attention, positional encoding, masking
      (no `np.triu`/`np.tril` either) or layers.
- [ ] Function names and arguments **exactly** as in the brief.
- [ ] Hand calculations show **every intermediate matrix and attention weight**, and are **verified against Python**.
- [ ] Tensor shapes documented in the code.

---

## 2. How we write things

### ✍️ Handwritten note: maths, words, drawings. **No code, ever.**
| Kind | How it's written |
|---|---|
| **Explanations** | Short and direct, one idea per line. Never copy out the question; just write `Q2 (a)`. |
| **Derivations** | One line saying *what this proves and why it matters* → 3–6 short steps → **boxed result**. |
| **Hand calculations** | Step by step, like an algorithm: **Given → Step 1 → Step 2 → … → boxed answer → `Python ✓`** |
| **Algorithm boxes** | 4–6 numbered steps in words: attention (Q2) · masking (Q4) · multi-head (Q6) · encoder (Q8) · decoder (Q9) |
| **Diagrams** | 3 only, **simple boxes and arrows**, about 5 min each with a ruler. The draft lists every label. |
| **Rounding** | 4 d.p. while working; final answer 3 d.p., rounded **once** from the calculator's full value. Calculator in **RADIANS**. |

**Size: about 20–22 pages** (the draft is ~7.5 typed pages, including the full-page Q9a drawing), in 3 sittings of about 2 hours each.

| Sitting | Questions | ≈ pages |
|---|---|---|
| S1 | Q1–Q3 | 7 |
| S2 | Q4–Q7 | 7–8 |
| S3 | Q8–Q10 | 6–7 (includes the one full-page diagram, Q9a) |

### 💻 Notebook: the same 4 steps for every question
1. **Idea in one line** + *how we thought about it*, in plain words
2. **Code**, simple over clever, with a **shape comment on every line that changes a shape**
   (`# (n,d_k) @ (d_k,n) -> (n,n)`) and a *why* comment on anything not obvious
3. **Output**: intermediate matrices, then your hand numbers vs Python, ✓ each
4. **What this shows**: the interpretation in 2–4 lines

Fixed seed · imports only `numpy` and `matplotlib` · ends with **"All checks: N/N passed"**.

### 📘 Learning PDF: `docs/learning/unit4-transformers.html → .pdf`
Plain words, **no maths**: for each question, the problem, an everyday scenario (lamps on cinema
seats, a new student asking for help, a translation office…) and what each calculation *means*.
6 pages. The maths lives in the handwritten notes.

---

## 3. The 37 parts
✍️ hand · 💻 notebook · ✍️💻 both · ⚑ = changed after the audit

**Q1 Positional encoding (10)**
- [ ] a (3) ✍️ roles of `pos, i, d, 10000`; frequency `ωᵢ = 1/10000^(2i/d)`, wavelength `2π·10000^(2i/d)`: first dims fast, last dims slow
- [ ] b (3) ✍️💻 `d=4`, `pos=0,1,2` → 3×4 matrix (4 d.p.)
- [ ] c (3) 💻 `sinusoidal_positional_encoding(seq_len, d_model)` → 20×16 heatmap
- [ ] d (1) ✍️💻 periodicity; `PE(pos+k)` is a rotation of `PE(pos)`, so an offset k is always the same transform

**Q2 Scaled dot-product attention (12)**
- [ ] a (3) ✍️ Q/K/V meaning; `(n×d_k)(d_k×n) = n×n`; ⚑ **derive √d:** if q and k have independent components with mean 0 and variance 1, then `Var(q·k) = d_k`, so dividing by √d_k gives variance 1 and softmax doesn't saturate; ⚑ algorithm box
- [ ] b (4) ✍️💻 `Q=K=V=I₂` → QKᵀ, scaled, softmax, output (3 d.p.)
- [ ] c (4) 💻 `softmax` (⚑ stable from the start) + `scaled_dot_product_attention(Q, K, V)` → (output, weights)
- [ ] d (1) 💻 rows sum to 1; hand vs Python table

**Q3 Q/K/V projections (10)**
- [ ] a (2) ✍️ `W_Q, W_K: d_model×d_k`, `W_V: d_model×d_v`; why three different matrices
- [ ] b (3) ✍️💻 given X and the three W's → Q, K, V → attention output
- [ ] c (3) 💻 `self_attention(X, W_Q, W_K, W_V)`
- [ ] d (2) ✍️💻 identity vs supplied: **same weights, different output**. ⚑ Show the row-shift algebra on paper; the notebook adds a random `W_K` that *does* change the weights

**Q4 Causal masking (10)**
- [ ] a (2) ✍️ why we mask during generation; **draw a 4×4 mask**; ⚑ algorithm box
- [ ] b (3) ✍️💻 `Q=K=V=[[1,0],[0,1],[1,1]]` → scores → mask → token-1 and token-2 probabilities
- [ ] c (4) 💻 `causal_self_attention(Q, K, V)`; ⚑ mask built by comparing positions `j > i` with `np.arange`
- [ ] d (1) 💻✍️ future weights exactly 0; causal mask vs padding mask

**Q5 Cross-attention (10)**
- [ ] a (2) ✍️ cross vs self; Q from decoder, K/V from encoder; weights `n×m`
- [ ] b (3) ✍️💻 `Q=[[1,0]]` + given K, V → scores, softmax, output
- [ ] c (3) 💻 `cross_attention(decoder_states, encoder_states, W_Q, W_K, W_V)`
- [ ] d (2) ✍️💻 the decoder "looks things up" in the source; why the weights aren't square (1×3 and 3×5 shown)

**Q6 Multi-head attention (12)**
- [ ] a (3) ✍️ every shape for `d_model=8, H=2, d_k=4`; ⚑ algorithm box
- [ ] b (3) ✍️💻 two heads by hand → concat. ⚑ Show `PPᵀ = I` (why head 2 = head 1); note the concat is 2×4 ≠ d_model 2, which is exactly what `W_O` (4×2) fixes. Python: `W_Q = W_K = [I | P]`, `W_V = [I | I]` (2×4), plus one run with `W_O = I₄` so the output *is* the concat and can be checked against the hand values
- [ ] c (4) 💻 `multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads)`; ⚑ head size = `W_Q.shape[1] // num_heads`
- [ ] d (2) ✍️ ⚑ three arguments: **fixed budget** (`H·d_k = d_model`, so the parameter count doesn't change) · **rank ≤ d_k** per head · **redundant heads can be pruned** (Michel 2019, Voita 2019), plus our toy

**Q7 Stability & temperature (8)**
- [ ] a (2) ✍️ proof: `softmax(z) = softmax(z − c)`, because `e^(−c)` cancels top and bottom
- [ ] b (2) 💻 `stable_softmax(x, axis=-1)` vs naive on `[1000, 1001, 1002]`
- [ ] c (2) 💻 ⚑ seeded random Q, K (6 tokens, d_k=8); `QKᵀ/T` for `T ∈ {0.5,1,2,4}`; bar charts + entropy; note `T = √d_k` is the standard scaling
- [ ] d (2) ✍️ sharpness vs T; why √d; ⚑ **Jacobian** `∂sᵢ/∂zⱼ = sᵢ(δᵢⱼ − sⱼ)` → 0 when saturated → no learning signal

**Q8 Encoder block (10)**
- [ ] a (3) ✍️ **simple diagram:** `X (n×d)` → MHA → Add (residual from X) → LayerNorm → FFN (`d → d_ff` → ReLU → `d_ff → d`) → Add (residual) → LayerNorm → `Y (n×d)`, with a **shape on every arrow**; one line on each part; ⚑ algorithm box
- [ ] b (5) 💻 `layer_norm(X, gamma, beta, eps=1e-5)`, `feed_forward(X, W1, b1, W2, b2)`, `transformer_encoder_block(X, params)`: post-norm, no dropout, per-token norm, reuses Q6
- [ ] c (2) ✍️💻 the FFN works on each token alone, attention mixes them; the residual is a gradient highway: `∂(x+F)/∂x = I + ∂F/∂x`

**Q9 Decoder & full architecture (10)**
- [ ] a (3) ✍️ **simple full-Transformer diagram.** Boxes and arrows, but every one of these must be labelled: embedding + **PE** on both sides · encoder ×N (MHA, FFN) · decoder ×N (**masked** MHA, **cross** MHA, FFN) · **Add & Norm after every sub-layer** (2 in the encoder, 3 in the decoder) · encoder output → **every** cross-attention · **Linear → softmax** over the vocabulary. Then: masked self-attention vs encoder–decoder attention; ⚑ decoder algorithm box
- [ ] b (5) 💻 `transformer_decoder_block(decoder_input, encoder_output, params)`: masked attn → add&norm → cross-attn → add&norm → FFN → add&norm · `d_model=8`, 2 heads, encoder length 5, decoder length 3 · assert every shape
- [ ] c (2) ✍️ trace the full path: source tokens → embedding + PE → encoder ×N → K, V in every cross-attention → decoder → Linear → softmax → next token; the decoder needs the mask (generation can't see the future), the encoder doesn't (it sees the whole input)

**Q10 Complexity (8)**
- [ ] a (3) ✍️ a **table**, per head vs total: Q/K/V projections `3·n·d·d_k` vs `3·n·d²` · `QKᵀ` `n²·d_k` vs `n²·d` · `AV` `n²·d_v` vs `n²·d` · memory `n²` vs `H·n²`
- [ ] b (3) 💻 `benchmark_attention(seq_lengths, d_model, num_heads)`, n = 32…512, d_model 64, H 4; times full multi-head attention, projections included; time + memory; plot. ⚑ Report the fitted slopes and **describe** the trend; don't assert it
- [ ] c (2) ✍️ quadratic growth; effects of masking, heads and length; arithmetic cost vs implementation savings

---

## 4. Code structure & tests
`softmax → scaled_dot_product_attention → self / causal / cross → mha_general → multi_head_attention → encoder → decoder → benchmark`

- ⚑ `mha_general(X_q, X_kv, W_Q, W_K, W_V, W_O, num_heads, causal=False)` does the work.
  `multi_head_attention(...)` keeps Q6's exact signature as a thin wrapper around it. Q9 calls it
  with `X_kv = encoder_output` and `causal=True`.
- `naive_softmax` exists only to show Q7b's `nan`.

| Q | Tests (the evidence behind every interpretation) |
|---|---|
| 1 | shape (20,16) · d=4 rows = hand · rotation property |
| 2, 4, 5 | rows sum to 1 · hand vs Python · future weights exactly 0 · editing a *future* token leaves earlier outputs unchanged · cross weights 1×3 and 3×5 |
| 3 | identity vs supplied: equal weights, different outputs · random W_K: weights change |
| 6 | 1 head = plain attention · reshape = per-head loop · toy heads identical · params constant for H = 1, 2, 4, 8 · per-head rank = d_k |
| 7 | naive → `nan`, stable → sums to 1 · entropy rises with T |
| 8 | LayerNorm: per-token mean 0 / var 1 (tolerance 1e-3, because ε shrinks the variance slightly) · edit token 2 → FFN changes row 2 only, attention changes every row |
| 9 | shape after every sub-layer · no future leak · edit one encoder token → every decoder row moves |
| 10 | memory = `H·n²·8 bytes` · theoretical cost split into projections `4n·d²` + attention `2n²·d` · measured slopes vs theoretical slopes |

---

## 5. Verified hand answers
Triple-checked (NumPy · 40-digit `Decimal` · simulated paper path), then re-derived by an
independent audit. All agree.

**Constants:** `1/√2 = 0.7071` · `e^0.7071 = 2.0281` · `e^1.4142 = 4.1132` · `10000^(2/4) = 100`

| Part | Answer |
|---|---|
| Q1b | `[[0, 1, 0, 1], [0.8415, 0.5403, 0.0100, 1.0000], [0.9093, −0.4161, 0.0200, 0.9998]]` |
| Q2b | `QKᵀ = I` → scaled `[[0.707, 0], [0, 0.707]]` → weights = output = `[[0.670, 0.330], [0.330, 0.670]]` |
| Q3b | Q `[[1,2],[0,1]]` · K `[[2,1],[1,0]]` · V `[[2,2],[0,1]]` · QKᵀ `[[4,1],[1,0]]` → scaled `[[2.828, 0.707], [0.707, 0]]` → weights `[[0.893, 0.107], [0.670, 0.330]]` → out `[[1.786, 1.893], [1.340, 1.670]]` |
| Q3d | identity: QKᵀ `[[5,2],[2,1]]` → scaled `[[3.536, 1.414], [1.414, 0.707]]` → **same weights** → out `[[0.893, 1.893], [0.670, 1.670]]` |
| Q4b | QKᵀ `[[1,0,1],[0,1,1],[1,1,2]]` → row 1 `[1, 0, 0]` · row 2 `[0.330, 0.670, 0]` · row 3 `[0.248, 0.248, 0.503]`, output row 3 `[0.752, 0.752]` |
| Q5b | QKᵀ `[1, 0, 1]` → scaled `[0.707, 0, 0.707]` → weights `[0.401, 0.198, 0.401]` → out `[1.203, 0.797]` |
| Q6b | both heads `[[0.670, 0.330], [0.330, 0.670]]` → concat (2×4) `[[0.670, 0.330, 0.670, 0.330], [0.330, 0.670, 0.330, 0.670]]` |
| Q7b | shift → `[−2, −1, 0]` → `e^z = [0.1353, 0.3679, 1]` → `[0.090, 0.245, 0.665]` · naive → `[nan, nan, nan]` |

## 6. Traps & insights (this is where the marks are)
1. **Radians.** `sin(1) = 0.8415`; DEG mode gives 0.0175 and silently ruins Q1b.
2. **Round once.** Q4b row 3 is 0.50349 → **0.503**. Rounding to 0.5035 first gives 0.504 ✗. 3 d.p. in the working gives output 0.751 ✗ instead of **0.752**.
3. **Q4b row 3 sums to 0.999** at 3 d.p. That's rounding. Write the note beside it.
4. **0.670 / 0.330 = softmax([1/√2, 0])** appears in Q2, Q3, Q4 and Q6. Learn it once.
5. **Q3d: same weights.** Each score row shifts by a constant (`[5,2]→[4,1]`), and softmax ignores constant shifts (Q7a). That only happens because both rows of X have x₂ − x₁ = 1. The obvious answer, "projections changed who attends to whom", is **wrong here**.
6. **Q6b: head 2 = head 1**, because `QKᵀ = X P Pᵀ Xᵀ = X Xᵀ`. The toy uses `d_k = 2` (scale by √2), so the concat is 2×4.
7. **Q10: n² only takes over past n = 2·d_model = 128.** Below that, the projections
   (`4n·d²`) cost more than attention (`2n²·d`), so even the *theoretical* slope is only
   1.3 → 1.7. The measured slope climbs a little faster (1.3 → 2.0), and that extra comes from
   memory and cache, i.e. the implementation. That split is exactly Q10c's "arithmetic cost vs implementation".

---

## 7. Roadmap
🟢 = **your green light** before we move on. Commit after every step, with no AI attribution.

**Weekend schedule (deadline Sun 22 Nov, end of day):** **Sat 21**: read the plain-words notes, then S1 + S2 · **Sun 22**: S3, scan, package, submit.

| # | Step | Who | Done when |
|---|---|---|---|
| 0 ✅ | Plan locked, questions reorganised, commit | Claude | 🟢 **your go** |
| 1 ✅ | **Learning PDF**, rewritten in plain words (6 pp) | Claude | renders clean, every question covered |
| 2 | **Read it**, ask anything | **You** | 🟢 you can say in your own words what attention, Q/K/V, the mask and heads do |
| 3 ✅ | **Notebook**: 73/73 checks | Claude, while you read | clean fresh-kernel run · all checks pass · numpy + matplotlib only · signatures exact |
| 4 ✅ | **Draft** `DRAFT.html → .pdf` + number audit: 173/173 numbers match | Claude | every draft number matched to the notebook, **0 mismatches** |
| 5 | **S1 (Q1–Q3):** copy → phone-scan → I check it → we walk through the Q1–Q3 code | You + me | 🟢 |
| 6 | **S2 (Q4–Q7):** same loop | You + me | 🟢 |
| 7 | **S3 (Q8–Q10):** same loop | You + me | 🟢 |
| 8 | **Final package**: one scan PDF, SOLUTION.md, README.md, every box ticked | Claude | nothing unticked |
| 9 | **Submit**: push, upload `.ipynb` + `.pdf` + GitHub link | **You** | Turned in ✅ |

**Files:** `BRIEF.md` · `SOLUTION.md` · `README.md` · `PrasannaKoirala_Transformer.ipynb` ·
`DRAFT.html/.pdf` · `render.sh` · `check_draft_numbers.py` · `outputs/` ·
`PrasannaKoirala_Transformer.pdf` · plus `docs/learning/unit4-transformers.html/.pdf`.
Drafts are committed publicly, same as a4 (your call).

---

## 8. Confidence (out of 10)
An independent reviewer audited the plan against the PDF twice: **9.4 → 9.6 / 10**. Its last
findings were fixed straight after. The "now" column is my estimate and was not re-audited.

| Q | Marks | Re-audit | Now | What held it back → fix |
|---|---|---|---|---|
| Q1–Q5, Q7 | 62 | 10 | 10 | — |
| Q6 | 12 | 9.5 | 10 | concat couldn't be checked in Python → run with `W_O = I₄` |
| Q8 | 10 | 9.5 | 10 | diagram labels + LayerNorm tolerance → both in the plan |
| Q9 | 10 | 9 | 9.5 | component list restored; the big drawing is still the riskiest part |
| Q10 | 8 | 9 | 9.5 | slope explanation corrected; Q10a as a table |

**Overall: 9.6 audited → about 9.8 / 10.** The rest depends on execution, not the plan.

**Outside the plan:** a tight weekend (deadline Sun 22 Nov) · you may present before submitting · handwriting
legibility within ~20 pages · an answer key might expect different weights in Q3d / Q6b (we
show the algebra, so a fair grader can follow it).

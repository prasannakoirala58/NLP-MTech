# a4 — Solution trace

**Deliverables:** handwritten note (5 pages) + `PrasannaKoirala_RNN_BPTT.ipynb` (31 cells)

---

## 1. Format — I got this wrong first

The attached PDF never mentions handwriting, so I initially planned a typed notebook. **The
Classroom post is the authority** and it is titled *"RNN - Hand Written Assignment"*, asking
for *"a scanned handwritten note"*. Corrected before any work was wasted.

Settled structure: a **5-page handwritten note answering all eight questions**, with the
notebook as supporting evidence. The note is deliberately self-contained — if a required
answer lived only in the repo and the grader did not follow the link, Q8 would score zero.

## 2. The one object everything rests on

Q4's derivation produces the Jacobian product, and the rest of the assignment falls out of it:

    ∂L/∂h_k = (∂L/∂h_T) · ∏ diag(1 − h_t²) W_hh

- **Q5** follows because each Jacobian contains its own `h_t`, so every hidden state must be stored
- **Q6** follows because `W_hh` appears in every factor, so its gradients sum over timesteps
- **Q7** follows because a product of T terms changes exponentially in T

Structuring the note around that single result rather than eight disconnected answers is what
makes it hang together.

## 3. The derivation is verified, not asserted

`∂L/∂h₁` computed two ways — PyTorch autograd, and the hand-derived formula evaluated directly:

| | |
|---|---|
| autograd | `[-0.39096165  0.36945550  0.38753370]` |
| by hand | `[-0.39096160  0.36945552  0.38753352]` |
| max difference | **1.79 × 10⁻⁷** |

Floating-point noise. Worth doing before committing the derivation to paper, rather than both
of us hoping it was right.

## 4. Q7 measured, and the result is asymmetric

Ran 30 timesteps with the spectral radius of `W_hh` fixed exactly, tracking `‖∂L/∂h_t‖`:

| radius | predicted (radius²⁹) | measured | |
|---|---|---|---|
| 0.5 | 1.86 × 10⁻⁹ | **1.86 × 10⁻⁹** | exact |
| 1.5 | 1.28 × 10⁵ | **2.12 × 10²** | ~600× weaker |

**Vanishing matches theory exactly. Exploding is heavily damped.** The cause is `tanh` itself:
as the gradient grows so does `h_t`, and as `h_t` approaches ±1 the factor `(1 − h_t²)`
collapses towards zero, throttling further growth. **The saturating non-linearity is its own
brake on explosion.**

That asymmetry is the most useful thing in the assignment and is not in the brief. It explains
why vanishing is the harder problem in practice: explosion is loud and fixable by gradient
clipping, while vanishing is silent — no error, no warning, the network simply never learns
long-range structure. Which is exactly the gap LSTM and GRU were built to close.

### A bug on the way there

The first version of this demo showed neither vanishing nor exploding — the numbers bounced
around while my labels confidently claimed both. Two causes: fresh input was injected at every
step so the signal kept being refreshed, and the spectral radius of `W_hh` was never
controlled.

Fixed by injecting input only at t=0, starting `h` small so `tanh` stays near-linear, and
building `W_hh` from a QR-orthogonal matrix scaled to an exact radius. Only then did the
measurement match theory. **The theory was never wrong; the experiment was.**

## 5. Q8

Custom `SimpleRNNCell` written out rather than calling `nn.RNN`, so all three parameters are
explicit and each gradient inspectable. T = 5, scalar loss on the final hidden state,
`.backward()`, and `.grad` printed for `W_xh`, `W_hh` and `b`.

The explanation ties back to Q6: `W_hh.grad` is **not** the influence of one use — it is the
sum across all T timesteps. One call to `.backward()` silently performed the whole of BPTT.

## 6. Verification

All 14 checks pass programmatically in the notebook's final cell, including that the
derivation matches autograd and that the measured gradients genuinely vanish and explode.

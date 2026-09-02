# a4 — Recurrent Neural Networks: BPTT and Gradient Dynamics

Source: `questions/RNN_Assignment_BPTT.pdf` · Syllabus Unit 3 · Lab 4

## Problem context
An RNN processes a 3-timestep sequence (x₁, x₂, x₃). Hidden state recurrence:

    hₜ = f(xₜ, hₜ₋₁)

Scalar loss evaluated at the final step only:

    L = L(h₃)

## Questions
- [ ] **Q1 Forward Pass** — explain processing of (x₁,x₂,x₃); **draw the unrolled
      computational graph** showing dependency flow among xₜ, hₜ, and L
- [ ] **Q2 Hidden State Dynamics** — functional role of hₜ; how it accumulates and
      carries context from preceding timesteps
- [ ] **Q3 BPTT** — how L backpropagates from h₃ to h₁; why it is called
      *Backpropagation Through Time*
- [ ] **Q4 Chain Rule Derivation** — derive explicit analytical ∂L/∂h₁ using the
      multivariable chain rule; **show each step** through h₂ and h₃
- [ ] **Q5 Intermediate Activations & Memory** — why h₁,h₂,h₃ must be stored during the
      forward pass for use in the backward pass
- [ ] **Q6 Parameter Sharing** — why the same weights are shared across every timestep;
      effect on gradient computation and parameter updates during BPTT
- [ ] **Q7 Vanishing & Exploding Gradients** — why gradients exponentially diminish or
      grow over long temporal sequences
- [ ] **Q8 PyTorch Implementation** — custom RNN cell/module, sequence of **T ≥ 5**
      timesteps, compute scalar loss, call `.backward()`, print `.grad` for **each**
      model parameter, explain what those gradients physically represent

## Deliverables
- [ ] `writeup/` — Q1–Q7 (Q1 needs a real diagram; Q4 needs full derivation steps)
- [ ] Q8 notebook + printed gradient tensors

## Expected learning outcomes (from the brief)
- Trace and visualise forward computation of an unrolled RNN
- Conceptualise hidden states as dynamic temporal memory
- Derive loss gradients across timesteps via the multivariable chain rule
- Understand BPTT mechanics and its memory overhead
- Explain parameter sharing and gradient accumulation across timesteps
- Diagnose vanishing/exploding gradients mathematically
- Implement recurrent models and inspect parameter gradients in PyTorch

## Notes
- Q4 is the centrepiece. ∂L/∂h₁ = (∂L/∂h₃)(∂h₃/∂h₂)(∂h₂/∂h₁) — expand each Jacobian
  explicitly for `hₜ = tanh(W_hh hₜ₋₁ + W_xh xₜ + b)`.
- Q7 follows directly from Q4: the product of Jacobians is what decays or blows up.
  Tie the two answers together explicitly.
- Q6 note: because W is shared, dL/dW is a **sum** over timesteps, not a single term.

/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Inference collapse: SDPA as the identity point of the PLDR family

Algebraic core of Theorem 5.4 of the paper:

* `score_sdpa_special_case` : with `G = 1` the PLGA score matrix
  `Q G Kᵀ` is exactly the SDPA score matrix `Q Kᵀ` (Theorem 5.4(i));
* `absorb_operator` : a constant operator `G*` can be absorbed into the
  query projection, `(X W) G* = X (W G*)`, which is why (absent RoPE)
  the collapsed model is exactly an SDPA-LLM with reparametrized
  projections (Theorem 5.4(ii));
* `absorb_operator_affine` : the implemented query projection is
  affine, `Q = X W + 1 bᵀ` (a bias row added to every position), and
  the absorption transforms weight AND bias together,
  `(X W + 1 bᵀ) G* = X (W G*) + 1 (bᵀ G*)` -- the collapsed model is
  an SDPA parameterization with `W' = W G*`, `b'ᵀ = bᵀ G*`
  (Theorem 5.4(ii) as stated for the bias-bearing architecture).

* `cache_sufficiency_scores` and `cache_sufficiency_rows` : the one-way
  cacheability statement of the paper (invariance suffices): if the
  recomputed operator equals the cached constant, every masked score
  row, hence every ideal attention row, is unchanged.  Only this
  direction is claimed; the converse is false (see the counterexample
  remark in the paper: `a = 0, b_a = G*` gives exact caching with
  varying upstream tensors).

Of the RoPE-aware part of Theorem 5.4(ii) (with rotary embeddings
the reduction to SDPA holds iff `G*` lies in the torus commutant),
`Rope.lean` checks the per-plane commutant characterization
(`commute_rot_iff`) and the sufficiency direction
(`relative_position_of_commute`); the necessity direction is proved
in the paper and is not formalized.
-/
import Mathlib
import PldrLlmMathFoundations.Softmax

namespace PldrLlm

open Matrix

variable {S dk : ℕ}

/-- Theorem 5.4(i): pinning the energy-curvature tensor to the identity
turns PLGA scores into SDPA scores. -/
theorem score_sdpa_special_case (Q K : Matrix (Fin S) (Fin dk) ℝ) :
    Q * (1 : Matrix (Fin dk) (Fin dk) ℝ) * Kᵀ = Q * Kᵀ := by
  rw [Matrix.mul_one]

/-- Theorem 5.4(ii), no-RoPE case: a constant operator is absorbed into
the query projection by associativity. -/
theorem absorb_operator (X : Matrix (Fin S) (Fin dk) ℝ)
    (W G : Matrix (Fin dk) (Fin dk) ℝ) :
    X * W * G = X * (W * G) :=
  Matrix.mul_assoc X W G

/-- Theorem 5.4(ii) for the bias-bearing architecture: the implemented
query projection is affine, `Q = X W + ones * b` with `ones` the
all-ones column and `b` the bias row, and right multiplication by a
constant operator `G` absorbs into the weight and the bias together
(distributivity and associativity).  Stated for an arbitrary column
`ones`, of which the implemented all-ones column is the instance. -/
theorem absorb_operator_affine (X : Matrix (Fin S) (Fin dk) ℝ)
    (W G : Matrix (Fin dk) (Fin dk) ℝ)
    (ones : Matrix (Fin S) (Fin 1) ℝ) (b : Matrix (Fin 1) (Fin dk) ℝ) :
    (X * W + ones * b) * G = X * (W * G) + ones * (b * G) := by
  rw [Matrix.add_mul, Matrix.mul_assoc, Matrix.mul_assoc]

/-- Cache sufficiency at the score level (the surviving direction of the
paper's cacheability proposition): if the recomputed operator `G` equals
the cached constant `Gc`, the score matrices coincide. -/
theorem cache_sufficiency_scores (Q K : Matrix (Fin S) (Fin dk) ℝ)
    {G Gc : Matrix (Fin dk) (Fin dk) ℝ} (h : G = Gc) :
    Q * G * Kᵀ = Q * Gc * Kᵀ := by
  rw [h]

/-- Cache sufficiency at the attention-row level: equal operators give
equal ideal masked attention rows.  Composed with
`maskedSoftmax_shift`, rows agreeing up to a common additive constant
also give equal attention rows, which is the score-level
characterization of exact functional cache equivalence in the paper. -/
theorem cache_sufficiency_rows (Q K : Matrix (Fin S) (Fin dk) ℝ)
    {G Gc : Matrix (Fin dk) (Fin dk) ℝ} (h : G = Gc)
    (J : Finset (Fin S)) (t : Fin S) :
    maskedSoftmax J (fun j => (Q * G * Kᵀ) t j)
      = maskedSoftmax J (fun j => (Q * Gc * Kᵀ) t j) := by
  rw [h]

end PldrLlm

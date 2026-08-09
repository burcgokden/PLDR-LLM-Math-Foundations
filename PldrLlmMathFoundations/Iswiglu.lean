/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# The iSwiGLU activation and strict positivity of the metric tensor

PLGA builds its metric tensor as `A_LM = iSwiGLU(W A + b_W) + eps` with
`iSwiGLU(u) = u^2 * sigmoid(u)`.  This file proves:

* `iswiglu_nonneg` : `iSwiGLU(u) >= 0` for every `u` (unlike SiLU/Swish,
  which goes negative);
* `iswiglu_pos`    : `iSwiGLU(u) > 0` for `u != 0`;
* `metric_entry_ge`: every entry of the metric tensor is `>= eps > 0`.

Paper reference: "Foundations of PLDR-LLM" (Gokden),
Definition of iSwiGLU and Proposition 3.7 (strict positivity and
well-posedness of the power law).
-/
import Mathlib

namespace PldrLlm

/-- The logistic sigmoid `1 / (1 + e^(-u))`. -/
noncomputable def sigmoid (u : ℝ) : ℝ := 1 / (1 + Real.exp (-u))

/-- `iswiglu u = u^2 * sigmoid u`, the activation used to produce the
metric tensor `A_LM` in PLGA. -/
noncomputable def iswiglu (u : ℝ) : ℝ := u ^ 2 * sigmoid u

lemma sigmoid_pos (u : ℝ) : 0 < sigmoid u := by
  have h : 0 < 1 + Real.exp (-u) := by positivity
  exact one_div_pos.mpr h

lemma sigmoid_lt_one (u : ℝ) : sigmoid u < 1 := by
  have h : 0 < Real.exp (-u) := Real.exp_pos _
  rw [sigmoid, div_lt_one (by positivity)]
  linarith

/-- iSwiGLU is nonnegative everywhere. -/
theorem iswiglu_nonneg (u : ℝ) : 0 ≤ iswiglu u :=
  mul_nonneg (sq_nonneg u) (sigmoid_pos u).le

/-- iSwiGLU is strictly positive away from zero. -/
theorem iswiglu_pos {u : ℝ} (hu : u ≠ 0) : 0 < iswiglu u := by
  have h2 : 0 < u ^ 2 := by
    rcases (sq_nonneg u).lt_or_eq with h | h
    · exact h
    · exact absurd ((pow_eq_zero_iff (two_ne_zero)).mp h.symm) hu
  exact mul_pos h2 (sigmoid_pos u)

/-- Each entry `A_LM i j = iswiglu (...) + eps` of the metric tensor is
at least `eps`. -/
theorem metric_entry_ge {d : ℕ} (M : Matrix (Fin d) (Fin d) ℝ)
    (eps : ℝ) (i j : Fin d) :
    eps ≤ iswiglu (M i j) + eps :=
  le_add_of_nonneg_left (iswiglu_nonneg _)

/-- Strict positivity of the metric tensor for `eps > 0`.  This is what
makes the elementwise power `A_LM ^ P` well defined (Proposition 3.7). -/
theorem metric_entry_pos {d : ℕ} (M : Matrix (Fin d) (Fin d) ℝ) {eps : ℝ}
    (heps : 0 < eps) (i j : Fin d) :
    0 < iswiglu (M i j) + eps :=
  heps.trans_le (metric_entry_ge M eps i j)

end PldrLlm

/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Softmax: positivity, row-stochasticity, and shift invariance

The attention operator `E_LM = softmax(mask + E)` of PLGA is row-stochastic,
so attention acts as a Markov (averaging) operator on values.  This file
proves the three facts that carry that statement:

* `softmax_pos`     : every softmax weight is strictly positive;
* `softmax_sum_one` : each softmax row sums to 1 (row-stochasticity);
* `softmax_shift`   : softmax is invariant under adding a constant to all
  scores;
* `softmax_eq_iff_shift` / `maskedSoftmax_eq_iff_shift` : two (allowed)
  score rows induce the same softmax distribution *iff* they differ by a
  common additive scalar — the row-level equality characterization of
  Proposition 5.2(iii); the model-level converse is not asserted there
  and is not asserted here.

The causal mask of the paper is the *ideal masked softmax* over an
allowed index set `J` (extended-real `-∞` semantics), not the plain
softmax with a finite additive constant; the implementation's finite
`-10⁹` mask realizes it only through floating-point underflow.  The
`maskedSoftmax` block formalizes the ideal operator:

* `maskedSoftmax_nonneg`, `maskedSoftmax_pos`, `maskedSoftmax_eq_zero` :
  entries are nonnegative, strictly positive exactly on the support `J`,
  and zero off it (exact causal support);
* `maskedSoftmax_sum_one` : each row sums to 1 (row-stochasticity on the
  support), Proposition "the ideal attention operator is Markov";
* `maskedSoftmax_shift` : invariance under a common shift of the scores.

The common-scalar-shift lemma `softmax_shift` is *not* a proof for the
coordinate-dependent additive mask; the masked statements above are the
ones the paper's causal-support claims rely on.
-/
import Mathlib

namespace PldrLlm

open Finset

variable {n : ℕ}

/-- `softmax x i = exp (x i) / sum_j exp (x j)`. -/
noncomputable def softmax (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => Real.exp (x i) / ∑ j, Real.exp (x j)

lemma softmax_denom_pos [NeZero n] (x : Fin n → ℝ) :
    0 < ∑ j, Real.exp (x j) :=
  Finset.sum_pos (fun j _ => Real.exp_pos (x j)) univ_nonempty

/-- Softmax weights are strictly positive. -/
theorem softmax_pos [NeZero n] (x : Fin n → ℝ) (i : Fin n) :
    0 < softmax x i :=
  div_pos (Real.exp_pos _) (softmax_denom_pos x)

/-- Each softmax row sums to 1: the attention operator is row-stochastic,
hence its action on values is a convex combination (an expectation). -/
theorem softmax_sum_one [NeZero n] (x : Fin n → ℝ) :
    ∑ i, softmax x i = 1 := by
  have h := (softmax_denom_pos x).ne'
  simp only [softmax]
  rw [← Finset.sum_div, div_self h]

/-- Softmax is invariant under a common shift of the scores. -/
theorem softmax_shift [NeZero n] (x : Fin n → ℝ) (c : ℝ) :
    softmax (fun i => x i + c) = softmax x := by
  funext i
  simp only [softmax, Real.exp_add, ← Finset.sum_mul]
  exact mul_div_mul_right _ _ (Real.exp_ne_zero c)

/-- A softmax-weighted average of values never exceeds the largest value:
quantitative form of "attention output lies in the convex hull of the
values" (Proposition 3.12(ii)). -/
theorem softmax_avg_le [NeZero n] (x v : Fin n → ℝ) {M : ℝ}
    (hv : ∀ i, v i ≤ M) :
    ∑ i, softmax x i * v i ≤ M := by
  calc ∑ i, softmax x i * v i
      ≤ ∑ i, softmax x i * M := by
        refine Finset.sum_le_sum fun i _ => ?_
        exact mul_le_mul_of_nonneg_left (hv i) (softmax_pos x i).le
    _ = M := by rw [← Finset.sum_mul, softmax_sum_one, one_mul]

/-- Equality characterization at the row level: two finite score rows
induce the same softmax distribution if and only if they differ by a
common additive scalar (Proposition 5.2(iii), softmax-row level).  The
forward direction takes logarithms of the ratio identity; the backward
direction is `softmax_shift`. -/
theorem softmax_eq_iff_shift [NeZero n] (x y : Fin n → ℝ) :
    softmax x = softmax y ↔ ∃ c : ℝ, ∀ i, y i = x i + c := by
  constructor
  · intro h
    have hx := softmax_denom_pos x
    have hy := softmax_denom_pos y
    have hx' : (∑ j, Real.exp (x j)) ≠ 0 := hx.ne'
    have hy' : (∑ j, Real.exp (y j)) ≠ 0 := hy.ne'
    refine ⟨Real.log (∑ j, Real.exp (y j)) - Real.log (∑ j, Real.exp (x j)),
      fun i => ?_⟩
    have hi := congrFun h i
    simp only [softmax] at hi
    have h1 : Real.exp (x i) * (∑ j, Real.exp (y j))
        = Real.exp (y i) * (∑ j, Real.exp (x j)) := by
      rw [div_eq_div_iff hx.ne' hy.ne'] at hi
      exact hi
    have h2 : Real.exp (y i)
        = Real.exp (x i) * ((∑ j, Real.exp (y j)) / (∑ j, Real.exp (x j))) := by
      field_simp
      linarith [h1]
    calc y i = Real.log (Real.exp (y i)) := (Real.log_exp _).symm
      _ = Real.log (Real.exp (x i) *
            ((∑ j, Real.exp (y j)) / (∑ j, Real.exp (x j)))) := by rw [h2]
      _ = x i + (Real.log (∑ j, Real.exp (y j))
            - Real.log (∑ j, Real.exp (x j))) := by
          rw [Real.log_mul (Real.exp_ne_zero _) (div_ne_zero hy.ne' hx.ne'),
            Real.log_div hy.ne' hx.ne', Real.log_exp]
  · rintro ⟨c, hc⟩
    have hxy : y = fun i => x i + c := funext hc
    rw [hxy, softmax_shift]

/-! ## The ideal masked softmax -/

/-- Ideal masked softmax over an allowed index set `J`: normalization is
over `J` only and entries off `J` are exactly `0` (the extended-real
`-∞`-mask semantics of the paper's causal attention). -/
noncomputable def maskedSoftmax (J : Finset (Fin n)) (x : Fin n → ℝ) :
    Fin n → ℝ :=
  fun i => if i ∈ J then Real.exp (x i) / ∑ j ∈ J, Real.exp (x j) else 0

lemma maskedSoftmax_denom_pos {J : Finset (Fin n)} (hJ : J.Nonempty)
    (x : Fin n → ℝ) : 0 < ∑ j ∈ J, Real.exp (x j) :=
  Finset.sum_pos (fun j _ => Real.exp_pos (x j)) hJ

/-- Masked softmax entries are nonnegative. -/
theorem maskedSoftmax_nonneg {J : Finset (Fin n)} (hJ : J.Nonempty)
    (x : Fin n → ℝ) (i : Fin n) : 0 ≤ maskedSoftmax J x i := by
  unfold maskedSoftmax
  split
  · exact le_of_lt (div_pos (Real.exp_pos _) (maskedSoftmax_denom_pos hJ x))
  · exact le_refl 0

/-- Masked softmax is strictly positive exactly on the allowed set. -/
theorem maskedSoftmax_pos {J : Finset (Fin n)} (hJ : J.Nonempty)
    (x : Fin n → ℝ) {i : Fin n} (hi : i ∈ J) :
    0 < maskedSoftmax J x i := by
  unfold maskedSoftmax
  rw [if_pos hi]
  exact div_pos (Real.exp_pos _) (maskedSoftmax_denom_pos hJ x)

/-- Exact causal support: masked softmax vanishes off the allowed set. -/
theorem maskedSoftmax_eq_zero {J : Finset (Fin n)} (x : Fin n → ℝ)
    {i : Fin n} (hi : i ∉ J) : maskedSoftmax J x i = 0 := by
  unfold maskedSoftmax
  exact if_neg hi

/-- Each masked-softmax row sums to 1: the ideal attention operator is
row-stochastic on its causal support (Proposition "the ideal attention
operator is Markov"). -/
theorem maskedSoftmax_sum_one {J : Finset (Fin n)} (hJ : J.Nonempty)
    (x : Fin n → ℝ) : ∑ i, maskedSoftmax J x i = 1 := by
  have hden := (maskedSoftmax_denom_pos hJ x).ne'
  unfold maskedSoftmax
  rw [Finset.sum_ite_mem, Finset.univ_inter, ← Finset.sum_div,
    div_self hden]

/-- Masked softmax is invariant under a common shift of the scores. -/
theorem maskedSoftmax_shift (J : Finset (Fin n)) (x : Fin n → ℝ)
    (c : ℝ) :
    maskedSoftmax J (fun i => x i + c) = maskedSoftmax J x := by
  funext i
  unfold maskedSoftmax
  by_cases hi : i ∈ J
  · rw [if_pos hi, if_pos hi]
    simp only [Real.exp_add, ← Finset.sum_mul]
    exact mul_div_mul_right _ _ (Real.exp_ne_zero c)
  · rw [if_neg hi, if_neg hi]

/-- Equality characterization for the ideal masked softmax: two finite
allowed score rows induce the same masked-softmax distribution if and
only if they differ by a common additive scalar *on the allowed set*
(Proposition 5.2(iii), row level, causal-support form). -/
theorem maskedSoftmax_eq_iff_shift {J : Finset (Fin n)} (hJ : J.Nonempty)
    (x y : Fin n → ℝ) :
    maskedSoftmax J x = maskedSoftmax J y ↔
      ∃ c : ℝ, ∀ i ∈ J, y i = x i + c := by
  constructor
  · intro h
    have hx := maskedSoftmax_denom_pos hJ x
    have hy := maskedSoftmax_denom_pos hJ y
    have hx' : (∑ j ∈ J, Real.exp (x j)) ≠ 0 := hx.ne'
    have hy' : (∑ j ∈ J, Real.exp (y j)) ≠ 0 := hy.ne'
    refine ⟨Real.log (∑ j ∈ J, Real.exp (y j))
      - Real.log (∑ j ∈ J, Real.exp (x j)), fun i hi => ?_⟩
    have hval := congrFun h i
    unfold maskedSoftmax at hval
    rw [if_pos hi, if_pos hi] at hval
    have h1 : Real.exp (x i) * (∑ j ∈ J, Real.exp (y j))
        = Real.exp (y i) * (∑ j ∈ J, Real.exp (x j)) := by
      rw [div_eq_div_iff hx.ne' hy.ne'] at hval
      exact hval
    have h2 : Real.exp (y i)
        = Real.exp (x i) *
          ((∑ j ∈ J, Real.exp (y j)) / (∑ j ∈ J, Real.exp (x j))) := by
      field_simp
      linarith [h1]
    calc y i = Real.log (Real.exp (y i)) := (Real.log_exp _).symm
      _ = Real.log (Real.exp (x i) *
            ((∑ j ∈ J, Real.exp (y j)) / (∑ j ∈ J, Real.exp (x j)))) := by
          rw [h2]
      _ = x i + (Real.log (∑ j ∈ J, Real.exp (y j))
            - Real.log (∑ j ∈ J, Real.exp (x j))) := by
          rw [Real.log_mul (Real.exp_ne_zero _) (div_ne_zero hy.ne' hx.ne'),
            Real.log_div hy.ne' hx.ne', Real.log_exp]
  · rintro ⟨c, hc⟩
    funext i
    unfold maskedSoftmax
    by_cases hi : i ∈ J
    · rw [if_pos hi, if_pos hi, hc i hi]
      have hsum : ∑ j ∈ J, Real.exp (y j)
          = (∑ j ∈ J, Real.exp (x j)) * Real.exp c := by
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun j hj => by
          rw [hc j hj, Real.exp_add]
      rw [hsum, Real.exp_add]
      exact (mul_div_mul_right _ _ (Real.exp_ne_zero c)).symm
    · rw [if_neg hi, if_neg hi]

end PldrLlm

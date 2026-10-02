/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# The DAG regularizer: nonnegative walk weights and `tr exp(N) ≥ d`

The DAG loss of PLDR-LLM is `DL(M) = |log (tr e^{M⊙M} / dk)|` with
`M ⊙ M` entrywise nonnegative.  The trace expands as the total weight of
closed walks: `tr e^N = Σ_k tr(N^k)/k!` with every term nonnegative and
the `k = 0` term equal to `d`.  This file proves:

* `pow_entry_nonneg`  : powers of an entrywise-nonnegative matrix are
  entrywise nonnegative (closed-walk weights are nonnegative);
* `trace_pow_nonneg`  : `tr (N^k) ≥ 0` (the geometric side of the
  walk-counting identity, Theorem 4.4 / Remark 4.6);
* `trace_exp_ge_card` : `tr (exp N) ≥ d`; applied to `N = M ⊙ M`
  with `d > 0`, this gives `tr e^{M⊙M} / d ≥ 1`, so the absolute value
  in the normalized logarithmic loss is analytically redundant;
* `trace_exp_ge_card_add_trace` : the sharper bound
  `tr (exp N) ≥ d + tr N`;
* `trace_exp_ge_of_diag_bound` : the positivity obstruction of the paper
  (Remark on the DAG loss): if every diagonal entry of `N` is at least
  `ε`, then `tr (exp N) ≥ d + d·ε`.  Applied to `N = M ⊙ M` with
  `|M i i| ≥ ε₀ > 0` gives the unnormalized bound `h(M) ≥ d·ε₀²`.
  For `d > 0`, monotonicity of the logarithm then gives the normalized
  floor `log(1 + ε₀²)` as a written step, not a declaration here.
  `A_LM` has the architectural entry floor; strict positivity of `A_P`
  does not supply that same uniform floor for arbitrary learned exponents.
  Computed zeros can be compatible with rounding near one or cancellation
  in a shifted logarithmic reduction. The positive increment need not
  underflow. This exact-real module formalizes neither floating-point
  behavior nor the equality-iff-acyclic converse.
-/
import Mathlib

namespace PldrLlm

open Matrix NormedSpace

open scoped Nat

variable {d : ℕ}

/-- Entrywise nonnegativity is preserved by matrix powers: the weight of
every closed walk is nonnegative. -/
theorem pow_entry_nonneg (N : Matrix (Fin d) (Fin d) ℝ)
    (hN : ∀ i j, 0 ≤ N i j) (k : ℕ) :
    ∀ i j, 0 ≤ (N ^ k) i j := by
  induction k with
  | zero =>
      intro i j
      by_cases hij : i = j <;> simp [hij]
  | succ k ih =>
      intro i j
      rw [pow_succ, Matrix.mul_apply]
      exact Finset.sum_nonneg fun l _ => mul_nonneg (ih i l) (hN l j)

/-- The trace of every power of an entrywise-nonnegative matrix is
nonnegative. -/
theorem trace_pow_nonneg (N : Matrix (Fin d) (Fin d) ℝ)
    (hN : ∀ i j, 0 ≤ N i j) (k : ℕ) :
    0 ≤ (N ^ k).trace := by
  simp only [Matrix.trace]
  exact Finset.sum_nonneg fun i _ => pow_entry_nonneg N hN k i i

-- The matrix exponential needs a (any) norm making `Matrix` a Banach
-- algebra; the `L∞`-operator norm is the standard local choice.
attribute [local instance] Matrix.linftyOpNormedAddCommGroup
  Matrix.linftyOpNormedRing Matrix.linftyOpNormedAlgebra

/-- `tr (exp N) ≥ d` for entrywise-nonnegative `N`: each diagonal entry
of `exp N` is at least its `k = 0` term `1`, because all closed-walk
contributions are nonnegative. Applied to `N = M ⊙ M` with `d > 0`,
this shows `tr e^{M⊙M} / d ≥ 1`, making the absolute value in the
normalized logarithmic loss redundant. The unnormalized theorem also
holds at dimension zero. -/
theorem trace_exp_ge_card (N : Matrix (Fin d) (Fin d) ℝ)
    (hN : ∀ i j, 0 ≤ N i j) :
    (d : ℝ) ≤ (exp N).trace := by
  have hsum : HasSum (fun k : ℕ => (k !⁻¹ : ℝ) • N ^ k) (exp N) :=
    exp_series_hasSum_exp' N
  -- Push the sum to each diagonal entry.
  have hdiag := hsum.matrix_diag
  have hentry : ∀ i : Fin d,
      HasSum (fun k : ℕ => (k !⁻¹ : ℝ) * (N ^ k) i i)
        ((exp N).diag i) := by
    intro i
    have h := Pi.hasSum.mp hdiag i
    simpa [Matrix.diag, Matrix.smul_apply, smul_eq_mul] using h
  -- Each diagonal entry of `exp N` is at least the `k = 0` term.
  have h1 : ∀ i : Fin d, (1 : ℝ) ≤ (exp N).diag i := by
    intro i
    have hle := le_hasSum (hentry i) 0 fun j _ =>
      mul_nonneg (by positivity) (pow_entry_nonneg N hN j i i)
    simpa [Matrix.one_apply] using hle
  calc (d : ℝ) = ∑ _i : Fin d, (1 : ℝ) := by simp
    _ ≤ ∑ i, (exp N).diag i := Finset.sum_le_sum fun i _ => h1 i
    _ = (exp N).trace := by simp [Matrix.trace]

/-- Sharper bound keeping the `k = 0` and `k = 1` walk terms:
`tr (exp N) ≥ d + tr N` for entrywise-nonnegative `N`. -/
theorem trace_exp_ge_card_add_trace (N : Matrix (Fin d) (Fin d) ℝ)
    (hN : ∀ i j, 0 ≤ N i j) :
    (d : ℝ) + N.trace ≤ (exp N).trace := by
  have hsum : HasSum (fun k : ℕ => (k !⁻¹ : ℝ) • N ^ k) (exp N) :=
    exp_series_hasSum_exp' N
  have hdiag := hsum.matrix_diag
  have hentry : ∀ i : Fin d,
      HasSum (fun k : ℕ => (k !⁻¹ : ℝ) * (N ^ k) i i)
        ((exp N).diag i) := by
    intro i
    have h := Pi.hasSum.mp hdiag i
    simpa [Matrix.diag, Matrix.smul_apply, smul_eq_mul] using h
  -- Each diagonal entry of `exp N` is at least its `k = 0` plus `k = 1`
  -- terms, i.e. `1 + N i i`.
  have h1 : ∀ i : Fin d, (1 : ℝ) + N i i ≤ (exp N).diag i := by
    intro i
    have hle := sum_le_hasSum ({0, 1} : Finset ℕ)
      (fun j _ => mul_nonneg (by positivity) (pow_entry_nonneg N hN j i i))
      (hentry i)
    have hsplit : ∑ k ∈ ({0, 1} : Finset ℕ),
        (k !⁻¹ : ℝ) * (N ^ k) i i = 1 + N i i := by
      rw [Finset.sum_pair (by norm_num : (0 : ℕ) ≠ 1)]
      simp
    rw [hsplit] at hle
    exact hle
  calc (d : ℝ) + N.trace
      = ∑ i : Fin d, ((1 : ℝ) + N i i) := by
        simp [Matrix.trace, Matrix.diag, Finset.sum_add_distrib]
    _ ≤ ∑ i, (exp N).diag i := Finset.sum_le_sum fun i _ => h1 i
    _ = (exp N).trace := by simp [Matrix.trace]

/-- For entrywise nonnegative `N` with `N i i ≥ η`, this declaration proves
`tr (exp N) ≥ d + d·η`. Applying it to `N = M ⊙ M`, with `d > 0` and
`|M i i| ≥ ε₀ > 0`, gives `h(M) ≥ d·ε₀²`. Monotonicity of the logarithm
then gives the normalized floor `log(1 + ε₀²)` in the written argument.
The architectural entry floor `ε₀ = 10⁻⁹` applies to `A_LM`. The tensor
`A_P` is strictly positive in exact real arithmetic but has no such uniform
floor for arbitrary learned powers. No positive-definiteness or
floating-point claim is made. -/
theorem trace_exp_ge_of_diag_bound (N : Matrix (Fin d) (Fin d) ℝ)
    (hN : ∀ i j, 0 ≤ N i j) (ε : ℝ) (hdiag : ∀ i, ε ≤ N i i) :
    (d : ℝ) + d * ε ≤ (exp N).trace := by
  have htr : (d : ℝ) * ε ≤ N.trace := by
    calc (d : ℝ) * ε = ∑ _i : Fin d, ε := by
          simp [Finset.sum_const, mul_comm]
      _ ≤ ∑ i, N i i := Finset.sum_le_sum fun i _ => hdiag i
      _ = N.trace := by simp [Matrix.trace, Matrix.diag]
  linarith [trace_exp_ge_card_add_trace N hN]

end PldrLlm

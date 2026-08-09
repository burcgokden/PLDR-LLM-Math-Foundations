/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# ε-LayerNorm: exact shift invariance, ε = 0 scale invariance, norm identity

The implemented LayerNorm of PLDR-LLM carries the regularization
constant `ε = 10⁻⁶`:
`LNε(r) = γ ⊙ (r - mean r) / sqrt (var r + ε) + β`.
This file formalizes the paper's ε-LayerNorm lemma.  For ε > 0 the
implemented map keeps exact shift invariance only; exact positive-scale
invariance and the spherical range hold only in the ε = 0 limit:

* `rowVar_nonneg`     : the row variance is nonnegative;
* `lnEps_shift`       : exact shift invariance
  `LNε(r + c·1) = LNε(r)` for every `ε` (the surviving exact invariance);
* `lnEps_scale_eps_zero` : exact positive-scale invariance holds at
  `ε = 0` on rows of positive variance. This module checks this
  (forward) direction only; that no `ε > 0` map has the invariance is
  proved in the paper and not formalized here;
* `sum_sq_normalized` : the exact norm identity
  `Σ_i ((r i - mean r)/sqrt (var r + ε))² = d · var r / (var r + ε)`,
  which is `< d` for `ε > 0`: the range of the implemented map is a
  ball, not the sphere of the idealized `ε = 0` map.
-/
import Mathlib

namespace PldrLlm

open Finset

variable {d : ℕ}

/-- Mean of the entries of a row `r : Fin d → ℝ`. -/
noncomputable def rowMean (r : Fin d → ℝ) : ℝ := (∑ i, r i) / d

/-- Biased variance of the entries of a row. -/
noncomputable def rowVar (r : Fin d → ℝ) : ℝ :=
  (∑ i, (r i - rowMean r) ^ 2) / d

/-- The ε-regularized LayerNorm row map with learned affine parameters. -/
noncomputable def lnEps (ε : ℝ) (γ β : Fin d → ℝ) (r : Fin d → ℝ) :
    Fin d → ℝ :=
  fun i => γ i * ((r i - rowMean r) / Real.sqrt (rowVar r + ε)) + β i

lemma rowVar_nonneg (r : Fin d → ℝ) : 0 ≤ rowVar r :=
  div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) (Nat.cast_nonneg d)

lemma rowMean_shift [NeZero d] (r : Fin d → ℝ) (c : ℝ) :
    rowMean (fun i => r i + c) = rowMean r + c := by
  have hd : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne d)
  unfold rowMean
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, add_div, nsmul_eq_mul, mul_comm,
    mul_div_assoc, div_self hd, mul_one]

lemma rowVar_shift [NeZero d] (r : Fin d → ℝ) (c : ℝ) :
    rowVar (fun i => r i + c) = rowVar r := by
  unfold rowVar
  rw [rowMean_shift]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  ring_nf

/-- Exact shift invariance of the implemented (ε-regularized) LayerNorm:
the only invariance that survives `ε > 0` exactly. -/
theorem lnEps_shift [NeZero d] (ε : ℝ) (γ β : Fin d → ℝ)
    (r : Fin d → ℝ) (c : ℝ) :
    lnEps ε γ β (fun i => r i + c) = lnEps ε γ β r := by
  funext i
  unfold lnEps
  rw [rowMean_shift, rowVar_shift]
  ring_nf

lemma rowMean_smul [NeZero d] (r : Fin d → ℝ) (c : ℝ) :
    rowMean (fun i => c * r i) = c * rowMean r := by
  unfold rowMean
  rw [← Finset.mul_sum, mul_div_assoc]

lemma rowVar_smul [NeZero d] (r : Fin d → ℝ) (c : ℝ) :
    rowVar (fun i => c * r i) = c ^ 2 * rowVar r := by
  unfold rowVar
  rw [rowMean_smul, ← mul_div_assoc]
  congr 1
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

/-- Exact positive-scale invariance holds for the idealized `ε = 0` map
on rows of positive variance.  For the implemented `ε = 10⁻⁶ > 0` it
fails (the paper quantifies the error); this lemma isolates exactly what
the idealization provides. -/
theorem lnEps_scale_eps_zero [NeZero d] (γ β : Fin d → ℝ)
    (r : Fin d → ℝ) {c : ℝ} (hc : 0 < c) (hv : 0 < rowVar r) :
    lnEps 0 γ β (fun i => c * r i) = lnEps 0 γ β r := by
  funext i
  unfold lnEps
  rw [rowMean_smul, rowVar_smul, add_zero, add_zero]
  have hsq : Real.sqrt (c ^ 2 * rowVar r) = c * Real.sqrt (rowVar r) := by
    rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hc.le]
  rw [hsq]
  have hs : Real.sqrt (rowVar r) ≠ 0 := by
    positivity
  have : c * r i - c * rowMean r = c * (r i - rowMean r) := by ring
  rw [this, mul_div_mul_left _ _ (ne_of_gt hc)]

/-- Exact norm identity for the normalized vector of the implemented
map: `Σ ((r i - mean)/sqrt (var + ε))² = d·var/(var + ε)`.  For
`ε > 0` the right side is `< d`: the range is contained in the closed
ball of radius `√d`, approaching the sphere only as `var/ε → ∞`. -/
theorem sum_sq_normalized [NeZero d] (r : Fin d → ℝ) {ε : ℝ}
    (hε : 0 < ε) :
    ∑ i, ((r i - rowMean r) / Real.sqrt (rowVar r + ε)) ^ 2
      = d * rowVar r / (rowVar r + ε) := by
  have hpos : 0 < rowVar r + ε := by
    have := rowVar_nonneg r
    linarith
  have hd : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne d)
  have hsum : ∑ i, (r i - rowMean r) ^ 2 = d * rowVar r := by
    unfold rowVar
    field_simp
  calc ∑ i, ((r i - rowMean r) / Real.sqrt (rowVar r + ε)) ^ 2
      = ∑ i, (r i - rowMean r) ^ 2 / (rowVar r + ε) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [div_pow, Real.sq_sqrt hpos.le]
    _ = (∑ i, (r i - rowMean r) ^ 2) / (rowVar r + ε) := by
        rw [Finset.sum_div]
    _ = d * rowVar r / (rowVar r + ε) := by rw [hsum]

end PldrLlm

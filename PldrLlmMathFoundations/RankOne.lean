/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# The rank-one singularity condition

At convergence the metric generator `A` of a PLDR-LLM has identical rows:
`A = 1 αᵀ` (the "learned singularity condition" of arXiv:2502.13502).
This file formalizes Proposition 3.9 of the paper for the matrix
`rowClone α` with every row equal to `α`:

* `rowClone_rank_le_one`  : `rank A <= 1`;
* `rowClone_rank_eq_one`  : `rank A = 1` exactly when `α ≠ 0`;
* `rowClone_det_eq_zero`  : `det A = 0` whenever `d >= 2`;
* `rowClone_mulVec_one`   : the all-ones vector is an eigenvector with
  eigenvalue the common row sum `s = Σ_j α j`;
* `rowClone_mul_self` and `rowClone_pow` : `A² = s A` and
  `A^(k+1) = s^k A`, so for `s ≠ 0` the matrix `A / s` is an idempotent
  (oblique projection);
* `rowClone_sq_eq_zero`   : in the degenerate case `s = 0` the matrix is
  nilpotent of index at most 2 (`A² = 0`), so "unique nonzero
  eigenvalue" statements require `s ≠ 0`.

The paper's Proposition on the rank-one singularity condition makes no
eigenvalue-perturbation claim for the nonnormal perturbed matrix
`1 αᵀ + Δ`; the singular-value determinant bound stated there is not
formalized here (recorded as prose-only in the coverage table).
-/
import Mathlib

namespace PldrLlm

open Matrix

variable {d : ℕ}

/-- The matrix `1 αᵀ` whose every row is `α` (identical-rows
configuration of the learned singularity condition). -/
def rowClone (α : Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.vecMulVec (fun _ => 1) α

@[simp]
lemma rowClone_apply (α : Fin d → ℝ) (i j : Fin d) :
    rowClone α i j = α j := by
  simp [rowClone, Matrix.vecMulVec_apply]

/-- `A = 1 αᵀ` has rank at most one. -/
theorem rowClone_rank_le_one (α : Fin d → ℝ) :
    (rowClone α).rank ≤ 1 :=
  Matrix.rank_vecMulVec_le _ _

@[simp]
lemma rowClone_eq_zero_iff (α : Fin d → ℝ) :
    rowClone α = 0 ↔ α = 0 := by
  constructor
  · intro h
    funext j
    have := congrArg (fun M => M j j) h
    simpa using this
  · intro h
    subst h
    ext i j
    simp

/-- `A = 1 αᵀ` has rank exactly one iff `α ≠ 0`; without this
hypothesis the statement fails at `α = 0`, where the matrix is zero.
Proof: every row equals `α`, so the row space is the span of the
single vector `α`. -/
theorem rowClone_rank_eq_one {α : Fin d → ℝ} (hα : α ≠ 0) :
    (rowClone α).rank = 1 := by
  have hd : 0 < d := by
    rcases Nat.eq_zero_or_pos d with h | h
    · subst h
      exact absurd (funext fun i => i.elim0) hα
    · exact h
  have hr : Set.range (rowClone α).row = {α} := by
    ext v
    simp only [Set.mem_singleton_iff, Matrix.row]
    constructor
    · rintro ⟨i, rfl⟩
      funext j
      simp
    · rintro rfl
      exact ⟨⟨0, hd⟩, by funext j; simp⟩
  rw [Matrix.rank_eq_finrank_span_row, hr]
  simpa using finrank_span_singleton hα

/-- `det A = 0` for `d >= 2`: the singularity observed exactly (determinant
zero in floating point) on all heads of trained PLDR-LLMs. -/
theorem rowClone_det_eq_zero (hd : 2 ≤ d) (α : Fin d → ℝ) :
    (rowClone α).det = 0 := by
  have h01 : (⟨0, by omega⟩ : Fin d) ≠ ⟨1, by omega⟩ := by
    simp [Fin.ext_iff]
  exact Matrix.det_zero_of_row_eq h01 (by funext j; simp)

/-- The all-ones vector is an eigenvector of `A = 1 αᵀ`, with eigenvalue
the common row sum `s = Σ_j α j`. -/
theorem rowClone_mulVec_one (α : Fin d → ℝ) :
    rowClone α *ᵥ (fun _ => 1) = fun _ => ∑ j, α j := by
  funext i
  simp [Matrix.mulVec, dotProduct]

/-- `A² = s A` with `s` the common row sum: `A / s` is idempotent. -/
theorem rowClone_mul_self (α : Fin d → ℝ) :
    rowClone α * rowClone α = (∑ k, α k) • rowClone α := by
  ext i j
  simp only [Matrix.mul_apply, rowClone_apply, Matrix.smul_apply,
    smul_eq_mul]
  rw [← Finset.sum_mul]

/-- `A^(k+1) = s^k A`: all powers stay on the ray of `A`. -/
theorem rowClone_pow (α : Fin d → ℝ) (k : ℕ) :
    rowClone α ^ (k + 1) = (∑ j, α j) ^ k • rowClone α := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [pow_succ, ih, smul_mul_assoc, rowClone_mul_self, smul_smul,
        pow_succ]

/-- Degenerate case of the singularity condition: when the common row
sum vanishes, `A = 1 αᵀ` is nilpotent of index at most 2.  A nonzero
`α` with `Σ α = 0` therefore gives a nonzero matrix whose spectrum is
`{0}`: the "unique nonzero eigenvalue equals the row sum" reading
requires `s ≠ 0`. -/
theorem rowClone_sq_eq_zero {α : Fin d → ℝ} (h : ∑ j, α j = 0) :
    rowClone α * rowClone α = 0 := by
  rw [rowClone_mul_self, h, zero_smul]

end PldrLlm

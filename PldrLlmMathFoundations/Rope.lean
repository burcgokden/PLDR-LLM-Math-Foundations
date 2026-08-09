/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# RoPE rotations and the commutant characterization

RoPE acts per head as a block-diagonal orthogonal rotation with 2×2
blocks `rot θ`.  This file proves, at the level of a single rotation
plane, the group and commutant structure used by the paper:

* `rot_mul`, `rot_zero`, `rot_transpose`, `rot_orthogonal` : `n ↦ rot (n θ)`
  is a homomorphism into `SO(2)`;
* `commute_rot_iff` : a 2×2 matrix commutes with `rot θ` (for `sin θ ≠ 0`)
  iff it is `c • 1 + s • J`, i.e. acts as the complex scalar `c + i s` on
  the rotation plane.  This is the per-block content of the commutant
  characterization of relative-position invariance (Proposition 4.1) and
  the source of the `dk² − dk` positional expressivity gap (Corollary 7.1:
  the commutant has 2 of the 4 real dimensions of each 2×2 block);
* `relative_position_of_commute` : an operator commuting with the torus
  produces scores depending only on the position offset,
  `rot (−u) * G * rot v = G * rot (v − u)`.
-/
import Mathlib

namespace PldrLlm

open Matrix

/-- The 2×2 rotation block of RoPE at angle `θ`. -/
noncomputable def rot (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

/-- The complex structure `J` on the rotation plane (`rot (π/2)`). -/
def rotJ : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

theorem rot_zero : rot 0 = 1 := by
  rw [rot, Matrix.one_fin_two]
  norm_num

/-- `rot θ * rot φ = rot (θ + φ)`: positions add, so `n ↦ rot (n θ)` is a
group homomorphism (the torus action of RoPE). -/
theorem rot_mul (θ φ : ℝ) : rot θ * rot φ = rot (θ + φ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rot, Matrix.mul_apply, Fin.sum_univ_two, Real.cos_add,
      Real.sin_add] <;> ring

theorem rot_transpose (θ : ℝ) : (rot θ)ᵀ = rot (-θ) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [rot]

/-- RoPE rotations are orthogonal. -/
theorem rot_orthogonal (θ : ℝ) : (rot θ)ᵀ * rot θ = 1 := by
  rw [rot_transpose, rot_mul, neg_add_cancel, rot_zero]

/-- `c • 1 + s • J` commutes with every rotation (the commutant contains
the complex scalars of the plane). -/
theorem smul_one_add_smul_rotJ_commute (a b θ : ℝ) :
    (a • (1 : Matrix (Fin 2) (Fin 2) ℝ) + b • rotJ) * rot θ =
      rot θ * (a • (1 : Matrix (Fin 2) (Fin 2) ℝ) + b • rotJ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rot, rotJ, Matrix.mul_apply, Fin.sum_univ_two,
      Matrix.one_apply] <;> ring

/-- Commutant characterization (single-plane core of Proposition 4.1):
for `sin θ ≠ 0`, a 2×2 real matrix `M` commutes with `rot θ` iff
`M = M₀₀ • 1 + M₁₀ • J`.  The commutant is 2-dimensional inside the
4-dimensional block; summed over the `dk/2` planes this is the
`dk² − dk` expressivity gap of Corollary 7.1. -/
theorem commute_rot_iff {θ : ℝ} (hθ : Real.sin θ ≠ 0)
    (M : Matrix (Fin 2) (Fin 2) ℝ) :
    M * rot θ = rot θ * M ↔
      M = M 0 0 • (1 : Matrix (Fin 2) (Fin 2) ℝ) + M 1 0 • rotJ := by
  constructor
  · intro h
    have h' : ∀ i j, (M * rot θ) i j = (rot θ * M) i j := fun i j => by
      rw [h]
    have h00 := h' 0 0
    have h01 := h' 0 1
    simp [Matrix.mul_apply, Fin.sum_univ_two, rot] at h00 h01
    -- From the (0,0) entry: (M 0 1 + M 1 0) * sin θ = 0.
    have e1 : M 0 1 = -M 1 0 := by
      have hz : (M 0 1 + M 1 0) * Real.sin θ = 0 := by
        linear_combination h00
      rcases mul_eq_zero.mp hz with hz | hz
      · linarith
      · exact absurd hz hθ
    -- From the (0,1) entry: (M 1 1 - M 0 0) * sin θ = 0.
    have e2 : M 1 1 = M 0 0 := by
      have hz : (M 1 1 - M 0 0) * Real.sin θ = 0 := by
        linear_combination h01
      rcases mul_eq_zero.mp hz with hz | hz
      · linarith
      · exact absurd hz hθ
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [rotJ, e1, e2]
  · intro h
    rw [h]
    exact smul_one_add_smul_rotJ_commute _ _ θ

/-- If `G` commutes with all rotations, its scores are relative:
`rot (−u) * G * rot v = G * rot (v − u)` depends on positions only through
the offset `v − u`.  This is the *if* direction of Proposition 4.1's
equivalence — commutation implies offset-only dependence — and the
reason SDPA (`G = 1`) has the relative-position property (the converse
direction, offset-only dependence forcing commutation, is proved in
the paper and not formalized here). -/
theorem relative_position_of_commute
    (G : Matrix (Fin 2) (Fin 2) ℝ)
    (hG : ∀ t : ℝ, G * rot t = rot t * G) (u v : ℝ) :
    rot (-u) * G * rot v = G * rot (v - u) := by
  rw [← hG (-u), Matrix.mul_assoc, rot_mul, neg_add_eq_sub]

end PldrLlm

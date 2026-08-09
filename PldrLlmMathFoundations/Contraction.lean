/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Iterated contraction of the metric-learner row map

Stage 3 of the invariance mechanism (Proposition 5.12 of the paper): the
metric learner acts through a single shared row map `φ`, a composition of
residual units.  If the units are Lipschitz the image diameter is
controlled multiplicatively, and in the contractive regime it decays
exponentially in depth, producing the observed collapse of the metric
generator onto a single row profile.  This file proves:

* `diam_comp_le`    : `diam ((g ∘ f) '' s) ≤ K_g K_f diam s`;
* `diam_iterate_le` : `diam (f^[N] '' s) ≤ K^N diam s`, the exponential-
  in-depth contraction of Proposition 5.12(i).
-/
import Mathlib

namespace PldrLlm

open EMetric

variable {α β γ : Type*} [PseudoEMetricSpace α] [PseudoEMetricSpace β]
  [PseudoEMetricSpace γ]

/-- Diameters compose multiplicatively under Lipschitz maps. -/
theorem diam_comp_le {f : α → β} {g : β → γ} {Kf Kg : NNReal}
    (hf : LipschitzWith Kf f) (hg : LipschitzWith Kg g) (s : Set α) :
    Metric.ediam ((g ∘ f) '' s) ≤ (Kg * Kf : NNReal) * Metric.ediam s := by
  rw [Set.image_comp]
  calc Metric.ediam (g '' (f '' s))
      ≤ Kg * Metric.ediam (f '' s) := hg.ediam_image_le _
    _ ≤ Kg * (Kf * Metric.ediam s) := by
        gcongr
        exact hf.ediam_image_le _
    _ = (Kg * Kf : NNReal) * Metric.ediam s := by
        rw [ENNReal.coe_mul, mul_assoc]

/-- An `N`-fold iterated `K`-Lipschitz map shrinks diameters by `K^N`:
for `K < 1` the image diameter decays exponentially in depth.  This is
a diameter bound for iterating a single map; the trained residual units
are *distinct* maps (see `diam_comp_le` for the two-map composition
step), and no fixed point or attractor dynamics is asserted here or in
the paper (Proposition 5.12 disclaims the Banach reading). -/
theorem diam_iterate_le {f : α → α} {K : NNReal}
    (hf : LipschitzWith K f) (s : Set α) (N : ℕ) :
    Metric.ediam (f^[N] '' s) ≤ (K : ENNReal) ^ N * Metric.ediam s := by
  induction N with
  | zero => simp
  | succ n ih =>
      rw [Function.iterate_succ', Set.image_comp]
      calc Metric.ediam (f '' (f^[n] '' s))
          ≤ K * Metric.ediam (f^[n] '' s) := hf.ediam_image_le _
        _ ≤ K * ((K : ENNReal) ^ n * Metric.ediam s) :=
            by gcongr
        _ = (K : ENNReal) ^ (n + 1) * Metric.ediam s := by ring

end PldrLlm

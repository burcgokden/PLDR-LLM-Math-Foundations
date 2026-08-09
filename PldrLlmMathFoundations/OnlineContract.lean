/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# The online generation contract and historical-row prefix consistency

At generation step `t` the deployed model is invoked on exactly the
known prefix `x_{1:t}` and only the final output row is consumed.  This
file states that interface over an *abstract* decoder map
`F : List σ → List ρ` and separates two properties the paper's
architecture section distinguishes:

* `onlineOut` / `online_causal` : the step-`t` online output is a
  function of the length-`t` prefix alone (Remark 4.7).  The proof is definitional
  *by design* — the content is that the deployed interface is a stated
  mathematical object, so causality of sequential generation needs no
  invariance or collapse assumption.
* `PrefixConsistent` : the strictly stronger property that row `r` of a
  longer call equals row `r` of the call on its own prefix
  (zero-based: the call on `x.take (r+1)`).  The comparison is
  `Option`-valued, so this property alone is satisfied vacuously by the
  constant-empty decoder (`constNil_prefixConsistent`).
* `LengthPreserving` / `PrefixConsistentShaped` : the decoder returns
  one row per input row (the paper's codomain statement), and the
  bundle of both properties — the exact formalization of
  Definition 4.8.  Under the shape conjunct the `Option` comparison
  upgrades to an equality of total rows
  (`prefixConsistentShaped_getElem`), and the vacuous witness is
  excluded (`constNil_not_prefixConsistentShaped`).
* `RowLocal` / `prefixConsistent_of_rowLocal` /
  `prefixConsistentShaped_of_rowLocal` / `rowMapDecoder_rowLocal` : a
  decoder whose row `r` is computed from the row's own prefix — the
  shape of causal (masked) dot-product attention, where row `r` sees
  keys and values `≤ r` only — is prefix-consistent (Remark 4.9(i)),
  and, given the shape property, prefix-consistent as a well-shaped
  map (`rowMapDecoder` has the shape property:
  `rowMapDecoder_lengthPreserving`).
* `rowMapDecoder_block_eq_online` : for a row-local decoder, row `r` of
  one parallel call *is* the online step-`(r+1)` output — the exact
  statement behind the coincidence of one-pass (block) and sequential
  final-row scoring for prefix-consistent maps.
* `globalSumDecoder_not_prefixConsistent` : a minimal global-aggregation
  decoder (every row adds a summary of the *entire* input — the scalar
  shadow of a global query Gram) is not prefix-consistent, by a
  concrete two-token computation; it *is* length-preserving
  (`globalSumDecoder_lengthPreserving`), so the failure is one of
  consistency, not of shape.

The full PLDR-LLM decoder is not formalized; these are wrapper-level
statements over an abstract map, matching the exact-coverage table.
-/
import Mathlib

namespace PldrLlm

variable {σ ρ : Type*}

/-- The online step-`t` output of a decoder `F`: apply `F` to the
length-`t` prefix and read the final row (`none` for an empty call). -/
def onlineOut (F : List σ → List ρ) (x : List σ) (t : ℕ) : Option ρ :=
  (F (x.take t)).getLast?

/-- **Online causality.**  The step-`t` online output depends only on
the length-`t` prefix: two inputs agreeing on their first `t` tokens
produce the same step-`t` output, for *any* decoder map. -/
theorem online_causal (F : List σ → List ρ) {x y : List σ} (t : ℕ)
    (h : x.take t = y.take t) : onlineOut F x t = onlineOut F y t := by
  unfold onlineOut
  rw [h]

/-- **Historical-row prefix consistency** (zero-based): row `r` of the
full call equals row `r` of the call on the length-`(r+1)` prefix.
The comparison is `Option`-valued; see `PrefixConsistentShaped` for
the well-shaped bundle that formalizes Definition 4.8 exactly. -/
def PrefixConsistent (F : List σ → List ρ) : Prop :=
  ∀ (x : List σ) (r : ℕ), r < x.length →
    (F x)[r]? = (F (x.take (r + 1)))[r]?

/-- Length preservation: the decoder returns one output row per input
row — the shape stated by the paper's decoder codomain (a
length-indexed family with `|F x| = |x|`). -/
def LengthPreserving (F : List σ → List ρ) : Prop :=
  ∀ x : List σ, (F x).length = x.length

/-- **Definition 4.8, bundled**: prefix consistency *as a well-shaped
map* — the decoder returns one row per input row, and every
historical row of a longer call equals the corresponding row of the
call on its own prefix.  The shape conjunct is not redundant: the
`Option`-valued comparison alone is satisfied vacuously by the
constant-empty decoder (`constNil_prefixConsistent`), which the
bundle excludes (`constNil_not_prefixConsistentShaped`). -/
def PrefixConsistentShaped (F : List σ → List ρ) : Prop :=
  LengthPreserving F ∧ PrefixConsistent F

/-- Under length preservation the `Option`-valued row comparison
upgrades to an equality of *total* rows: both calls provably have a
row `r`. -/
theorem prefixConsistentShaped_getElem {F : List σ → List ρ}
    (h : PrefixConsistentShaped F) (x : List σ) (r : ℕ)
    (hr : r < x.length) :
    (F x)[r]'(by rw [h.1]; exact hr)
      = (F (x.take (r + 1)))[r]'(by
          rw [h.1, List.length_take]; omega) := by
  have hx : r < (F x).length := by rw [h.1]; exact hr
  have hp : r < (F (x.take (r + 1))).length := by
    rw [h.1, List.length_take]; omega
  have h2 := h.2 x r hr
  rw [List.getElem?_eq_getElem hx, List.getElem?_eq_getElem hp] at h2
  exact Option.some.inj h2

/-- The constant-empty decoder satisfies the `Option`-valued property
vacuously (both sides are `none`): the reason `PrefixConsistentShaped`
carries the shape conjunct. -/
theorem constNil_prefixConsistent :
    PrefixConsistent (fun _ : List σ => ([] : List ρ)) := by
  intro x r _
  simp

/-- The constant-empty decoder is *not* prefix consistent as a
well-shaped map: it has no rows at all. -/
theorem constNil_not_prefixConsistentShaped :
    ¬ PrefixConsistentShaped (fun _ : List ℤ => ([] : List ℤ)) := by
  intro h
  have h0 := h.1 [0]
  simp at h0

/-- Row-locality: row `r` of the output depends only on the first
`r+1` input tokens. -/
def RowLocal (F : List σ → List ρ) : Prop :=
  ∀ (x y : List σ) (r : ℕ), x.take (r + 1) = y.take (r + 1) →
    (F x)[r]? = (F y)[r]?

/-- Row-locality implies prefix consistency. -/
theorem prefixConsistent_of_rowLocal {F : List σ → List ρ}
    (h : RowLocal F) : PrefixConsistent F := by
  intro x r _
  refine h x (x.take (r + 1)) r ?_
  rw [List.take_take, min_self]

/-- A row-local, length-preserving decoder is prefix consistent as a
well-shaped map (Definition 4.8; row-locality alone constrains no
output length). -/
theorem prefixConsistentShaped_of_rowLocal {F : List σ → List ρ}
    (hL : LengthPreserving F) (h : RowLocal F) :
    PrefixConsistentShaped F :=
  ⟨hL, prefixConsistent_of_rowLocal h⟩

/-- The row-map decoder: row `r` is computed by a per-row function `f`
applied to the row's own prefix `x.take (r+1)` — the shape of causal
(masked) attention, where row `r` sees keys and values `≤ r` only. -/
def rowMapDecoder (f : List σ → ρ) (x : List σ) : List ρ :=
  (List.range x.length).map fun r => f (x.take (r + 1))

/-- The row-map decoder is row-local. -/
theorem rowMapDecoder_rowLocal (f : List σ → ρ) :
    RowLocal (rowMapDecoder f) := by
  intro x y r h
  have hlen : min (r + 1) x.length = min (r + 1) y.length := by
    simpa [List.length_take] using congrArg List.length h
  simp only [rowMapDecoder, List.getElem?_map]
  by_cases hx : r < x.length
  · have hy : r < y.length := by omega
    simp [hx, hy, h]
  · have hy : ¬ r < y.length := by omega
    simp [hx, hy]

/-- The row-map decoder returns one row per input row. -/
theorem rowMapDecoder_lengthPreserving (f : List σ → ρ) :
    LengthPreserving (rowMapDecoder f) := by
  intro x
  simp [rowMapDecoder]

/-- Causal masked attention, read as a row-map decoder, is
prefix-consistent. -/
theorem rowMapDecoder_prefixConsistent (f : List σ → ρ) :
    PrefixConsistent (rowMapDecoder f) :=
  prefixConsistent_of_rowLocal (rowMapDecoder_rowLocal f)

/-- Causal masked attention, read as a row-map decoder, satisfies the
full Definition 4.8 bundle: well-shaped and prefix consistent. -/
theorem rowMapDecoder_prefixConsistentShaped (f : List σ → ρ) :
    PrefixConsistentShaped (rowMapDecoder f) :=
  ⟨rowMapDecoder_lengthPreserving f, rowMapDecoder_prefixConsistent f⟩

/-- For a row-local decoder, row `r` of one parallel call is the online
step-`(r+1)` output: the one-pass (block) reading of a historical row
coincides with sequential final-row evaluation.  This is the exact
statement behind the equality of block and sequential scoring for
prefix-consistent maps. -/
theorem rowMapDecoder_block_eq_online (f : List σ → ρ) (x : List σ)
    (r : ℕ) (hr : r < x.length) :
    (rowMapDecoder f x)[r]? = onlineOut (rowMapDecoder f) x (r + 1) := by
  have hmin : min (r + 1) x.length = r + 1 := by omega
  simp only [onlineOut, rowMapDecoder, List.getLast?_eq_getElem?,
             List.length_map, List.length_range, List.length_take, hmin,
             List.getElem?_map]
  simp [hr, List.take_take]

/-- A minimal global-aggregation decoder: every row adds the sum of the
*entire* input — the scalar shadow of a global query Gram, where all
supplied tokens interact before any row is produced. -/
def globalSumDecoder (x : List ℤ) : List ℤ :=
  x.map fun a => a + x.sum

/-- The global-aggregation decoder is length-preserving: its failure
below is one of prefix consistency, not of shape. -/
theorem globalSumDecoder_lengthPreserving :
    LengthPreserving globalSumDecoder := by
  intro x
  simp [globalSumDecoder]

/-- The global-aggregation decoder is **not** prefix-consistent: on the
input `[0, 1]`, row `0` of the full call is `0 + (0+1) = 1`, while row
`0` of the prefix call on `[0]` is `0 + 0 = 0`. -/
theorem globalSumDecoder_not_prefixConsistent :
    ¬ PrefixConsistent globalSumDecoder := by
  intro h
  have h01 := h [0, 1] 0 (by norm_num)
  simp [globalSumDecoder] at h01

/-- A fortiori, the global-aggregation decoder fails the full
Definition 4.8 bundle. -/
theorem globalSumDecoder_not_prefixConsistentShaped :
    ¬ PrefixConsistentShaped globalSumDecoder :=
  fun h => globalSumDecoder_not_prefixConsistent h.2

end PldrLlm

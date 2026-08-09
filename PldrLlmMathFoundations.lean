/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

Lean 4 / mathlib formalization of selected proofs from
"Power law graph attention: exact generalization of scaled dot-product
attention, empirical collapse at inference" (B. Gokden).
See README.md for the file-by-file map to the paper's results.
-/
import PldrLlmMathFoundations.Iswiglu
import PldrLlmMathFoundations.Softmax
import PldrLlmMathFoundations.DensityOperator
import PldrLlmMathFoundations.HadamardPower
import PldrLlmMathFoundations.RankOne
import PldrLlmMathFoundations.LayerNorm
import PldrLlmMathFoundations.Rope
import PldrLlmMathFoundations.TwirlBound
import PldrLlmMathFoundations.DagLoss
import PldrLlmMathFoundations.Contraction
import PldrLlmMathFoundations.InferenceCollapse
import PldrLlmMathFoundations.OnlineContract

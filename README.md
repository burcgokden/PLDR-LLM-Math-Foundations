# PLDR-LLM-Math-Foundations

Lean 4 / [mathlib](https://github.com/leanprover-community/mathlib4)
formalization and numerical-audit suite for the paper

> Burc Gokden, *Power law graph attention: exact generalization of
> scaled dot-product attention, empirical collapse at inference*.

The paper gives a full formal mathematical treatment of the Large
Language Model from Power Law Decoder Representations (PLDR-LLM) and
its attention mechanism, Power Law Graph Attention (PLGA), and proves
how PLGA generalizes scaled dot-product attention (SDPA). This
repository contains the machine-checked and machine-measured evidence
behind the paper:

- **Machine-checked proofs** (`PldrLlmMathFoundations/`): the
  elementary algebraic and analytic cores of the paper's results,
  formalized in Lean 4 over mathlib. Every theorem is fully proved:
  the library builds with no `sorry` and no project axioms, and an
  axiom audit (`scripts/check_axioms.py`, run in CI) sweeps
  `#print axioms` over every exported theorem and fails if anything
  appears beyond mathlib's standard classical principles (`propext`,
  `Classical.choice`, `Quot.sound`) — the library is kernel-checked
  relative to those, not axiom-free in a foundational sense.
- **Numerical audits** ([`audit/`](audit/)): the code and complete
  results behind the paper's appendix "Numerical Audit on a Released
  Checkpoint", run against publicly released PLDR-LLM checkpoints at
  pinned revisions, with raw per-instance arrays shipped alongside
  every summary and a model-free test suite that re-derives the
  shipped summaries from the shipped raw arrays.

All theorem numbers used in this repository (code comments,
doc-comments, and the tables below) refer to the paper's numbered
results.

## What is formalized

The paper's appendix "Lean Formalization: Exact Coverage" carries the
same table with the per-result gaps spelled out.

| File | Paper result | Content |
|---|---|---|
| `Iswiglu.lean` | Def. 3.1; Prop. 3.7 | `iSwiGLU(u) = u^2 sigmoid(u) >= 0`, strict positivity away from 0, tensor entries `>= eps` |
| `Softmax.lean` | Prop. 3.12; Prop. 5.2(iii) row level | plain softmax positivity/row-stochasticity/common-shift invariance/convex-hull bound; **ideal masked softmax** over an allowed index set: nonnegativity, strict positivity on the support, exact zero off it, row sum 1, common-shift invariance (the causal-support semantics of the paper); **equality iff common shift** (`softmax_eq_iff_shift`, `maskedSoftmax_eq_iff_shift`): two (allowed) score rows induce the same distribution iff they differ by an additive scalar — the row-level characterization of Prop. 5.2(iii); no model-level converse is asserted |
| `DensityOperator.lean` | Prop. 3.6 | `Q^T Q` symmetric, positive semidefinite, quadratic form `= \|\|Qv\|\|^2`, `rank(Q^T Q) = rank Q <= min(S, d_k)` |
| `HadamardPower.lean` | Prop. 3.7 | one-parameter group law of the elementwise power `A ↦ A^{⊙tP}` for entrywise-positive `A` |
| `RankOne.lean` | Prop. 3.9 | identical-rows matrix `1 α^T`: rank `<= 1` with **equality iff `α ≠ 0`**, `det = 0` (`d >= 2`), row-sum eigenvector, `A^{k+1} = s^k A`, and the **nilpotent case** `A² = 0` when the row sum vanishes |
| `LayerNorm.lean` | Lem. 5.11 | **ε-LayerNorm**: exact shift invariance, exact positive-scale invariance at `ε = 0` on positive-variance rows, and the exact norm identity `Σ û² = d·var/(var+ε) < d` (ball, not sphere) |
| `Rope.lean` | Prop. 4.1; Cor. 7.1 (per-block); Thm. 5.4(ii) mechanism | RoPE rotations form a group homomorphism into SO(2); 2×2 commutant characterization `M R = R M ↔ M = c·1 + s·J` for `sin θ ≠ 0`; commuting operators give relative-position scores |
| `TwirlBound.lean` | Lem. 5.9 (analytic core) | `‖1 − e^{iω}‖ = 2\|sin(ω/2)\|` and the geometric-sum bound `‖Σ_{n=1}^S e^{iωn}‖ <= 1/\|sin(ω/2)\|`, giving the `O(1/S)` twirl decay for a fixed matrix |
| `DagLoss.lean` | Thm. 4.4 (walk side); Rem. 4.5 | powers of entrywise-nonnegative matrices are entrywise nonnegative; `tr(N^k) >= 0`; `tr(exp N) >= d` and the sharper `tr(exp N) >= d + tr N`; the **positivity obstruction** `tr(exp N) >= d + d·ε` for diagonal entries `>= ε` (exact acyclicity unattainable for the strictly positive PLGA tensors) |
| `Contraction.lean` | Prop. 5.12(i) | Lipschitz maps compose multiplicatively on diameters; iterated `K`-Lipschitz maps contract diameters by `K^N` (conditional on the measured `L_j`) |
| `InferenceCollapse.lean` | Thm. 5.4(i), (ii) (algebraic core); Prop. 5.2(i) | `Q · 1 · K^T = Q K^T` (SDPA as the identity point), absorption of a constant operator into the query projection — including the **affine form** `(X W + 1 bᵀ) G = X (W G) + 1 (bᵀ G)` matching the bias-bearing architecture (`absorb_operator_affine`) — and **one-way cache sufficiency** at the score and masked-attention-row level (the converse is false; see the paper's counterexample remark) |
| `OnlineContract.lean` | Rem. 4.7; Def. 4.8; Rem. 4.9 (abstract core) | The **online generation contract** over an abstract decoder map: `online_causal` (the step-`t` final-row output depends only on the length-`t` prefix); `PrefixConsistent` (the `Option`-valued row comparison) together with `LengthPreserving` and the bundle `PrefixConsistentShaped` (one output row per input row AND historical-row agreement — the exact formalization of Def. 4.8; `prefixConsistentShaped_getElem` upgrades the `Option` comparison to total rows, and the constant-empty decoder shows the shape conjunct is not redundant); `prefixConsistent_of_rowLocal`, `prefixConsistentShaped_of_rowLocal`, `rowMapDecoder_rowLocal`, `rowMapDecoder_lengthPreserving`, `rowMapDecoder_prefixConsistentShaped` (causally masked row-local attention satisfies the full bundle); `rowMapDecoder_block_eq_online` (block reading of a historical row = online final-row output, the coincidence behind one-pass vs sequential scoring); `globalSumDecoder_not_prefixConsistent` / `_not_prefixConsistentShaped` (a two-token global-aggregation decoder violates prefix consistency while being length-preserving). The full PLDR-LLM decoder is not formalized; these are wrapper-level statements |

Results whose content is genuinely infinite-dimensional,
probabilistic, or spectral are outside the scope of this
formalization: Perron–Frobenius theory (Thm. 3.8, not yet in mathlib),
the matrix Bernstein concentration (Prop. 5.10), the singular-value
determinant bound of Prop. 3.9(iv), the quantitative ε-LayerNorm
scale-error inequality of Lem. 5.11(ii), the NOTEARS acyclicity
equivalence (only the `tr exp` lower bounds and the obstruction are
checked, not equality-iff-acyclic), the multi-block nonresonance
commutant of Prop. 4.1 (only the single-block 2×2 commutant is
checked), the reversible-family spectral dictionary (Prop. 6.4), and
everything stated in the paper as a hypothesis, analogy, or
conjecture. Formalization of a proof core is not evidence for
adjacent unformalized claims; the paper's coverage appendix states
each gap explicitly.

## Building

Install [elan](https://github.com/leanprover/elan), then:

```sh
lake exe cache get   # download prebuilt mathlib (several GB)
lake build
```

A successful `lake build` re-verifies every proof with the Lean
kernel. The toolchain is pinned in `lean-toolchain`; mathlib is
pinned both in `lake-manifest.json` (including its `inputRev`, so the
build is warning-free) and (same commit) in `lakefile.toml`, keeping
the build reproducible. The proof modules import the umbrella
`Mathlib` module for pin stability; with the binary cache this costs
only download time, so `lake exe cache get` is the recommended first
step. CI (`.github/workflows/ci.yml`) rebuilds the library with the
mathlib binary cache, scans for forbidden tokens
(`sorry`/`admit`/`axiom`/`unsafe`), runs the axiom audit
(`scripts/check_axioms.py`), and runs the audits' model-free test
suite on every push.

## Numerical audits

[`audit/`](audit/) holds the Python scripts and full results for the
paper's numerical audits of the released checkpoints
[`fromthesky/PLDR-LLM-v51-SOC-110M-5`](https://huggingface.co/fromthesky/PLDR-LLM-v51-SOC-110M-5)
(the audited model) and
[`fromthesky/PLDR-LLM-v51-SOC-110M-1`](https://huggingface.co/fromthesky/PLDR-LLM-v51-SOC-110M-1)
(a same-family contrast), both pinned to fixed model revisions:

- **Main audit** (`audit.py` + `audit_h.py`): tensor spectra and
  numerical ranks (trained, and at random initialization under both
  the HF-port and the native SOC initialization laws), the directly
  measured ε-LayerNorm scale error, pre- versus post-rotation twirl
  energy per (prompt, layer) instance and prompt-pooled per layer,
  full-composition row-map Jacobians and empirical pairwise
  contraction ratios, the bound-form invariance budget versus measured
  decoding margins (sufficient criterion `2B < margin`), commutant
  residuals of the trained operators, per-step cached-operator
  deviations, DAG-loss values, and the order parameter in both
  normalizations.
- **Online-contract audit** (`audit_online.py`): historical-row
  context interaction under suffix changes, online determinism,
  padding at the Gram boundary, and sequential-versus-block candidate
  scoring with ranking comparisons, on both checkpoints.
- **Sequential-validation audit** (`audit_seq.py`): held-out blockwise
  cross-entropy versus sequential autoregressive NLL on pinned corpus
  windows, and sequential-versus-block scoring of real benchmark items
  from the published zero-shot task list (including TruthfulQA under
  its published `truthfulqa_mc2` probability-mass metric), with
  datasets pinned as parquet files by full revision hash.

Raw per-instance arrays behind every summary ship in the `.npz`
archives (SHA-256 recorded in each results JSON), and the shipped
results files record the exact library versions and device of the
runs in their `environment` blocks. The audits are deterministic by
construction (pinned cuBLAS workspace, forced deterministic
algorithms): two back-to-back runs on a fixed device reproduce the
results byte-for-byte. The test suite is model-free (numpy only; no
GPU, no checkpoint download) and re-derives every shipped JSON
summary from the shipped raw arrays, so the summaries cannot silently
drift from the raw data. See [`audit/README.md`](audit/README.md).

## Related repositories

- [LLM-from-Power-Law-Decoder-Representations](https://github.com/burcgokden/LLM-from-Power-Law-Decoder-Representations)
- [PLDR-LLM-with-KVG-cache](https://github.com/burcgokden/PLDR-LLM-with-KVG-cache)
- [PLDR-LLM-Self-Organized-Criticality](https://github.com/burcgokden/PLDR-LLM-Self-Organized-Criticality)
- [Power-Law-Graph-Transformer](https://github.com/burcgokden/Power-Law-Graph-Transformer)
- [CoulGAT-Graph-Attention-Interpretability](https://github.com/burcgokden/CoulGAT-Graph-Attention-Interpretability)

## License

Apache License 2.0; see [LICENSE](LICENSE).

# CLAUDE.md

## What this repository is

Machine-checked proofs and numerical audits for the paper *Power law
graph attention: exact generalization of scaled dot-product
attention, empirical collapse at inference* by Burc Gokden
(Fromthesky Research Labs LLC). Every theorem number used in code
comments, doc-comments, and READMEs refers to the paper's numbered
results.

## Product features

- **Lean 4 / mathlib formalization** (`PldrLlmMathFoundations/`,
  umbrella module `PldrLlmMathFoundations.lean`): the elementary
  algebraic and analytic cores of the paper's results — the SDPA
  special case and operator absorption (`InferenceCollapse.lean`),
  the RoPE commutant mechanism (`Rope.lean`), the ideal masked
  softmax and its equality-iff-common-shift characterization
  (`Softmax.lean`), the ε-LayerNorm invariances (`LayerNorm.lean`),
  the rank-one singularity condition (`RankOne.lean`), the DAG-loss
  walk inequalities and positivity obstruction (`DagLoss.lean`), the
  online generation contract and prefix consistency
  (`OnlineContract.lean`), and more. No `sorry`, no project axioms;
  the README's coverage table maps each file to its paper results,
  and the paper's coverage appendix states what is *not* formalized.
- **Numerical audit suite** (`audit/`): three deterministic,
  reproducibility-first audits of publicly released PLDR-LLM
  checkpoints at pinned revisions (main, online-contract,
  sequential-validation), shipping their complete results
  (`*_results.json`) and the raw per-instance arrays behind every
  summary (`*_raw.npz`, SHA-256 recorded in each JSON).
- **Model-free test suite** (`audit/test_*.py`): numpy-only tests
  (no GPU, no checkpoint download) covering the numerical helpers,
  semantic oracles (miniature-decoder finite-difference chain oracle,
  prefix-consistent vs global toy-decoder scoring oracles, benchmark
  request-template and TruthfulQA-MC2 transcription tests), and
  offline recomputation of every shipped summary from the shipped raw
  arrays.
- **CI** (`.github/workflows/ci.yml`): `lake build` with the mathlib
  binary cache, a forbidden-token scan, the axiom audit
  (`scripts/check_axioms.py` — nothing beyond `propext`,
  `Classical.choice`, `Quot.sound`), and the model-free test suite.

## Commands

```sh
# Lean (install elan first)
lake exe cache get          # prebuilt mathlib (several GB, once)
lake build                  # kernel re-verifies every proof
python3 scripts/check_axioms.py   # after lake build

# Audit tests (model-free; numpy only)
cd audit && python3 -m unittest -v

# Full audit runs (downloads pinned checkpoints; GPU by default)
cd audit && pip install -r requirements.txt
python3 audit.py && python3 audit_h.py
python3 audit_online.py
python3 audit_seq.py
```

## Conventions and invariants

- **Shipped result artifacts are frozen.** `audit/*_results.json` and
  `audit/*_raw.npz` are the exact runs backing the paper's appendix;
  their SHA-256 values are cross-referenced (each JSON records its
  NPZ's hash, and the tests verify it). Never regenerate or edit them
  in place; a fresh run belongs in a new commit that says so.
- **The audit design rule**: every reported quantity is either the
  named quantity computed directly, or is explicitly labeled a
  bound/proxy with its formula. Keep labels exact when editing
  (e.g. "sample-extrema proxy", "hypothetical single-rounding
  radius"); the tests assert several of them.
- **Chain assembly semantics**: an operator perturbation enters at
  the attention output of its layer, so it traverses that layer's own
  post-attention remainder before downstream layers. A
  miniature-decoder finite-difference oracle in `audit/test_audit.py`
  fails under any downstream-only assembly — do not "simplify" the
  assembly.
- **The exponent tensor `P` has no batch axis** (`[H, dk, dk]`);
  always build the power stage through the shape-checked
  `audit_lib.ap_tensor` (NumPy silently accepts several wrong
  broadcast pairings; a sentinel test guards this).
- **Determinism pins stay**: the cuBLAS workspace pin and forced
  deterministic algorithms are what make back-to-back runs
  byte-identical; removing them makes small order-parameter values
  shift across launches.
- **Sampling streams are pinned**: `seed_index` values in
  `audit/audit_seq.py` fix which dataset rows each task samples; do
  not renumber them.
- **Theorem numbering**: doc-comments cite the paper's numbered
  results (e.g. "Prop. 4.1", "Thm. 5.4"); check any numbering change
  against the paper and keep the README coverage table in sync.
- **Lean pins stay together**: `lean-toolchain`,
  `lakefile.toml`'s mathlib `rev`, and `lake-manifest.json`'s
  `inputRev` must agree (CI checks the sync); bumping one means
  bumping all and rebuilding.
- Keep the repository self-contained: relative links only, no
  machine-specific paths, and license under Apache 2.0 (`LICENSE`).

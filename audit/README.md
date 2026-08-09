# Numerical audits of the released checkpoints

Code and results for the paper's appendix "Numerical Audit on a
Released Checkpoint" (Burc Gokden, *Power law graph attention: exact
generalization of scaled dot-product attention, empirical collapse at
inference*): audits of the released checkpoint
[`fromthesky/PLDR-LLM-v51-SOC-110M-5`](https://huggingface.co/fromthesky/PLDR-LLM-v51-SOC-110M-5)
(pinned to revision `de8e539c0ba1829072f4b8c2c5fae3bde0a3a2d2`) — and,
where a contrast model is used,
[`fromthesky/PLDR-LLM-v51-SOC-110M-1`](https://huggingface.co/fromthesky/PLDR-LLM-v51-SOC-110M-1)
(revision `7a34e2ca9aa78038683677cfda17fe3a9fe6da8a`) — against the
measurable quantities the paper's results depend on.

## Design rule

Every reported quantity is either the named quantity computed
directly, or is explicitly labeled a bound/proxy with its formula
shown. In particular:

- the ε-LayerNorm scale error is the **directly computed** difference
  `LN(D) - LN(D/S)` with the checkpoint's gamma/beta/eps (the
  first-order proxy `eps/(2v)` is reported alongside, as a proxy);
- the twirl section decomposes **both** the pre-rotation and the
  rotated position-resolved query aggregates, so the off-commutant
  energy actually removed by the finite-S rotation is identified (not
  inferred from the post-rotation aggregate alone), and reports
  per-instance ratios and prompt-pooled per-layer aggregates each
  under its exact label (ratios do not commute with pooling);
- row-map contraction is measured on the **full composition**: spectra
  of the composite Jacobian at rows drawn from every prompt, plus
  empirical pairwise ratios `||phi(r)-phi(r')||/||r-r'||`; per-unit
  numbers are retained only as a labeled decomposition diagnostic;
- `G_LM` is compared against its cached value at **every** decoding
  step of every audited prompt;
- the greedy-margin criterion uses the correct sufficient factor
  `2B < margin`;
- the end-to-end chain is a **sample-extrema worst-case proxy**, not a
  tube or global certificate: activation/LN/operator extrema are
  maxima/minima over the audited prompt passes only, derivative
  factors are certified analytic envelopes on the sampled
  preactivation interval (never grid maxima), each injection
  traverses its own layer's post-attention remainder
  `LLN1*(1+Lffn)*LLN2` before the downstream layer products, and the
  stress-test radius `2^-24 * max ||G_LM||_F` is a hypothetical
  single-rounding scale, not the measured cache perturbation (which
  is 0 bitwise);
- the commutant residual `||G - Pi_comm G||_F/||G||_F` of the trained
  `G_LM` is measured per prompt/layer/head, quantifying occupancy of
  the absolute-position-sensitive directions counted by the paper's
  positional-codimension corollary;
- random-initialization controls run the same spectral battery under
  both the HF-port and the native SOC initialization laws (each
  labeled), so trained structure is attributed by comparison rather
  than assumption, against the law the checkpoint was actually
  trained from.

## The three audits

**Main audit** (`audit.py`, then `audit_h.py`; writes
`audit_results.json` + `audit_raw.npz`): tensor spectra and numerical
ranks (trained and random-init), direct ε-LayerNorm scale error,
twirl energy decomposition, commutant residual of the trained
operators, full-composition row-map contraction, budget constants and
the sample-extrema chain versus measured decoding margins, cached
versus uncached greedy decoding (per-step logit deviations, margins,
and `eps_G`), DAG-loss values, and — from `audit_h.py` — the order
parameter under both the RMS and the symmetrized signed-mean
normalizations, from two independent stochastic continuations per
prompt.

**Online-contract audit** (`audit_online.py`; writes
`audit_online_results.json` + `audit_online_raw.npz`): backing the
paper's appendix subsections on the `S = t` generation contract —
same-length historical-row context-interaction comparisons (fixed
pair + seeded randomized protocol, plain and attention-masked),
online determinism, padding at the Gram boundary (two padding
contents, strip-restores check), and sequential-final-row versus
one-pass block candidate scoring with ranking comparisons, on **both**
released checkpoints. The model-free tests include a semantic oracle:
a prefix-consistent toy decoder must give identical sequential and
block scores through the same index algebra, and a global-aggregation
toy decoder must not.

**Sequential-validation audit** (`audit_seq.py`; writes
`audit_seq_results.json` + `audit_seq_raw.npz`): held-out blockwise
cross-entropy versus sequential autoregressive NLL on seeded disjoint
token windows of a pinned public corpus (`Salesforce/wikitext`,
`wikitext-2-raw-v1` validation, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`), and sequential versus
one-pass block scoring of **real benchmark items** from the published
zero-shot task list (ARC-Easy/Challenge, HellaSwag, PIQA, OpenBookQA,
Social-IQa, WinoGrande, TruthfulQA; parquet files pinned by full
dataset-revision hash with per-file SHA-256 recorded), with request
templates transcribed from the pinned evaluation-harness fork and
verified by model-free unit tests, and the wrapper's encode-pair
convention reproduced (including the trailing-whitespace shift and
the joint-encoding split). TruthfulQA is evaluated under its
published **`truthfulqa_mc2`** protocol — multiple true answers,
scored by the normalized probability mass on the true choices
(`mc2_score`, a tested transcription of the pinned
`process_results_mc2`) — under both scoring protocols; a single-gold
`mc1` probe on the **same seeded dataset rows** is reported separately
under `s2_auxiliary`, outside the published-task count. Reported per
checkpoint and per task: per-token and per-candidate score-gap
distributions (also by candidate length), raw and length-normalized
argmax changes, discordant candidate pairs, accuracy under both
protocols, block decision-margin distributions, and the count of
items whose candidate-score gap reaches half the decision margin.

All three audits share the same determinism discipline (pinned cuBLAS
workspace, forced deterministic algorithms, fixed thread count on the
CPU path): two complete back-to-back runs on a fixed device reproduce
each results file and raw archive byte-for-byte. Exact float values
are device-configuration sensitive (quantities downstream of the
elementwise power stage amplify ulp-level differences); the shipped
files are the runs backing the paper's appendix, whose `environment`
blocks record the exact library versions, seed, and device used.

## Files

- `audit.py` — main audit (sections A–H incl. D2; see its docstring).
- `audit_h.py` — order parameter (RMS and signed-mean normalizations);
  merges into the main JSON/NPZ. Run after `audit.py`.
- `audit_lib.py` — pure numerical helpers shared by both scripts;
  model-free, numpy only.
- `audit_online.py` / `audit_online_lib.py` / `test_audit_online.py`
  — the online-contract audit, its pure helpers, and its model-free
  tests.
- `audit_seq.py` / `audit_seq_lib.py` / `test_audit_seq.py` — the
  sequential-validation audit, its pure helpers (request templates,
  encode-pair split, the MC2 metric, summary derivations, toy-decoder
  oracles), and its model-free tests.
- `test_audit.py` — unit tests for `audit_lib.py` plus consistency
  checks of the shipped results (see [Tests](#tests) below).
- `audit_results.json` / `audit_online_results.json` /
  `audit_seq_results.json` — summary results consumed by the paper's
  appendix, each including an `environment` record (model revisions,
  hash of the fetched `modeling_pldrllm.py`, library versions, seed)
  and the SHA-256 of its raw-records archive.
- `audit_raw.npz` / `audit_online_raw.npz` / `audit_seq_raw.npz` —
  raw per-instance arrays behind every summary.
- `requirements.txt` — the minimal dependency set (transformers
  pinned; see [Environment](#environment)).

The decoded continuations in the main results file
(`decode[*].continuation` for greedy, `stochastic_continuations` for
sampled) are included as a **neutral, human-inspectable record** of
exactly what the audited decoding produced, together with simple
repetition statistics (distinct-n ratios, duplicate 4-gram fraction).
They are *not* presented as evidence of generation quality; at this
model scale greedy decoding produces repetition loops, and the
recorded statistics quantify that plainly. Claims about the
checkpoint's language quality belong to the source papers' benchmark
evaluations, not to this audit.

## Running

```sh
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
python3 audit.py         # writes audit_results.json + audit_raw.npz
python3 audit_h.py       # merges the order parameter into both
python3 audit_online.py  # writes audit_online_results.json/.npz
python3 audit_seq.py     # writes audit_seq_results.json/.npz
```

The scripts download the checkpoints from the Hugging Face Hub on
first use (`trust_remote_code=True`; the model class is the released
`modeling_pldrllm.py` of the checkpoint, fetched at the pinned
revision). They run on GPU by default when one is available and fall
back to CPU otherwise; override with

- `AUDIT_DEVICE=cpu` (or `cuda`) — force the device for the main
  audit (`AUDIT_ONLINE_DEVICE` / `AUDIT_SEQ_DEVICE` for the others);
- `AUDIT_MODEL=<path-or-hub-id>` — audit another copy or checkpoint
  (`AUDIT_MODEL_5`/`AUDIT_MODEL_1` for the two-checkpoint audits);
- `AUDIT_REV=<revision>` — pin a different model revision
  (`AUDIT_REV_5`/`AUDIT_REV_1` likewise).

Runtime is some minutes per audit on a single consumer GPU, longer on
CPU. Exact float values depend on the device and library versions;
the shipped files are the runs backing the paper's appendix, and each
`environment` block records the exact configuration that produced
them.

## Tests

```sh
python3 -m unittest -v      # or: pytest
```

Run from this directory (test discovery from the repository root
finds no tests). The tests are model-free (numpy + standard library
only; no torch, no GPU, no checkpoint download). They cover the
numerical helpers against hand computations and finite differences —
including the sentinel exponent-indexing test, the all-zero
order-parameter pair, the factor-two greedy-margin criterion, the
zero-variance LayerNorm row, and overflow-safe DAG-loss evaluation —
plus **semantic oracles** that internal-consistency checks cannot
replace: a miniature-decoder finite-difference oracle for the chain
assembly (an actual two-layer post-attention path whose measured
sensitivity the assembled coefficient must dominate, and which a
downstream-only mis-assembly must fail), analytic prefix-consistent
and global toy-decoder oracles for the two scoring protocols, and
transcription tests of every benchmark request template and of the
TruthfulQA-MC2 metric against hand-built documents and hand-computed
values. Finally, the tests recompute every shipped JSON summary from
the shipped raw arrays (whose SHA-256 each JSON records), so the
summaries cannot silently drift from the raw data.

## Environment

`requirements.txt` lists the minimal dependency set. The released
checkpoint's Hugging Face port targets `transformers` 4.56.1 (pinned
there); under it the released files run unmodified. The exact library
versions, seed, and device behind the shipped results are recorded in
each results file's `environment` block.

Under `transformers` 5.x two load-compatibility issues arise, with no
effect on the computation once fixed: the port's `create_causal_mask`
call must be adapted to the 5.x keyword API (`input_embeds` became
`inputs_embeds`; `cache_position` was dropped), and non-persistent
buffers (the RoPE cache/angles, built in module `__init__`)
materialize uninitialized from the meta device, so `rope_init()` must
be re-run on every module after `from_pretrained` — otherwise every
output is NaN. The scripts already perform the second fix, which is
harmless under 4.x; the first requires editing the checkpoint's
`modeling_pldrllm.py`. Using the pinned `transformers` version avoids
both.

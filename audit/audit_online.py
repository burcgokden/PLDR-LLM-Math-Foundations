#!/usr/bin/env python3
"""Online-contract audit of the released PLDR-LLM checkpoints,
backing the paper's appendix subsections on the S=t generation
contract, historical-row prefix consistency, padding at the Gram
boundary, and sequential-versus-block candidate scoring.

Design rule (unchanged from the main audit): every reported quantity is
either the named quantity computed directly, or is explicitly labeled a
bound/proxy with its formula.  Raw per-instance arrays behind every
summary are stored in audit_online_raw.npz (SHA-256 recorded in the
JSON); test_audit_online.py re-derives the summaries offline and
verifies the scoring index algebra model-free.

Sections (run for BOTH released checkpoints):
  O1. Historical-row context-interaction probe: same-length inputs
      sharing a prefix and differing in the suffix (fixed pair and a
      seeded randomized batch, each also with the suffix marked as
      padding); per-position logit deltas over the shared prefix and
      per-layer deltas of the deductive tensors A, A_LM, G_LM.  A
      causal final-row generator may show historical-row movement here
      (the global query Gram lets all supplied tokens interact); a
      prefix-consistent map shows exact zeros.
  O2. Online determinism: repeated identical prefix calls compared
      bitwise (the step-t conditional is a deterministic function of
      the prefix in this configuration).
  O3. Padding at the Gram boundary: appending attention-masked padding
      rows to a prompt and comparing the final real-token row against
      the unpadded call, for two different padding contents; the
      current implementations include every supplied query row in the
      Gram, so the deltas are expected nonzero and are DOCUMENTED
      behavior (the deployed contract is the unpadded S=t interface);
      stripping the padding restores the reference bitwise.
  O4. Sequential final-row scores versus one-pass whole-candidate
      block scores on a fixed candidate suite: per-token and
      per-candidate absolute score differences and ordering agreement
      (argmax flips, discordant pairs).  Single-token candidates
      coincide exactly by construction and are asserted as a control.

Environment:
  AUDIT_ONLINE_DEVICE  cuda|cpu  (default: cuda if available; float32,
                       eager attention either way)
  AUDIT_MODEL_5 / AUDIT_REV_5    audited checkpoint (SOC-110M-5 pin)
  AUDIT_MODEL_1 / AUDIT_REV_1    contrast checkpoint (SOC-110M-1 pin)
Determinism: as in the main audit, the cuBLAS workspace is pinned and
deterministic algorithms are forced (without the pin, cuBLAS algorithm
selection can differ between process launches and ulp-level differences
are amplified downstream of the elementwise power stage); the thread
count is fixed for the CPU path.  Back-to-back runs are expected to be
byte-identical on a fixed device; the JSON records the device, and
individual float values are device-configuration sensitive as discussed
in the paper's audit scope.
Writes audit_online_results.json + audit_online_raw.npz (no wall-clock
fields, so back-to-back runs can be compared byte-for-byte).
"""
import json
import os
import subprocess
import warnings

import numpy as np
import torch
import transformers
import huggingface_hub

from audit_online_lib import (seq_prefix, block_input, block_score_rows,
                              gather_block_logprobs, delta_stats,
                              rank_comparison, summarize, random_pairs,
                              sha256_file)

warnings.filterwarnings("ignore")
# Cross-launch reproducibility (same pins as the main audit): cuBLAS
# reduction-order/algorithm choice can differ between process launches,
# and quantities downstream of the elementwise power stage amplify the
# resulting ulp-level differences; the workspace pin plus deterministic
# algorithms make repeated runs bitwise stable.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(16)

DEV = os.environ.get("AUDIT_ONLINE_DEVICE",
                     "cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = [
    dict(tag="soc110m5",
         model=os.environ.get("AUDIT_MODEL_5",
                              "fromthesky/PLDR-LLM-v51-SOC-110M-5"),
         rev=os.environ.get("AUDIT_REV_5",
                            "de8e539c0ba1829072f4b8c2c5fae3bde0a3a2d2")),
    dict(tag="soc110m1",
         model=os.environ.get("AUDIT_MODEL_1",
                              "fromthesky/PLDR-LLM-v51-SOC-110M-1"),
         rev=os.environ.get("AUDIT_REV_1",
                            "7a34e2ca9aa78038683677cfda17fe3a9fe6da8a")),
]

# Deductive tensors in the port's per-layer output tuple.
DEDUCTIVE = ((0, "A"), (1, "A_LM"), (5, "G_LM"))

# O1 protocol constants.
FIXED_PREFIX = [1000, 2000, 3000]
FIXED_SUFF_A = [4000, 5000, 6000]
FIXED_SUFF_B = [4001, 5001, 6001]
PROBE_SEED = 20260807
PROBE_TRIALS = 16
PROBE_LEN = 12
PROBE_PREFIX = 4

# O3/O4 protocol constants.
PAD_LEN = 4
PAD_CONTENTS = (0, 5)
SUITE_SEED = 20260807
GREEDY_LONG = 8

PROMPTS = [
 "The self-organized critical state of a sandpile is reached when the slope of the pile fluctuates around a stationary value, and avalanches of all sizes occur without a characteristic scale.",
 "In linear algebra, the singular value decomposition expresses any real matrix as a product of an orthogonal matrix, a diagonal matrix of nonnegative singular values, and another orthogonal matrix.",
 "The movie was a complete waste of time. The acting felt wooden, the plot made no sense after the first act, and the ending was both predictable and unearned. I would not recommend it to anyone.",
 "Once upon a time, in a village at the edge of a great forest, there lived a clockmaker whose clocks always ran a little fast, as if they were impatient for the future to arrive.",
 "To bake a simple sourdough loaf, combine flour, water, salt, and an active starter; fold the dough every half hour, proof it overnight in the refrigerator, and bake it in a covered pot.",
 "Quantum error correction protects information by encoding a logical qubit into an entangled state of many physical qubits, so that local noise can be detected and reversed without measuring the data.",
 "The quarterly report shows revenue growth of twelve percent, driven primarily by subscription renewals, while operating costs remained flat and customer churn declined for the third consecutive quarter.",
 "Rain fell on the harbor all night, and by morning the fishing boats lay under a thin silver mist that did not lift until the sun was high above the breakwater.",
]

RES = {}
RAW = {}


def log(*a):
    print(*a, flush=True)


def save():
    np.savez_compressed("audit_online_raw.npz", **RAW)
    RES["raw_records"] = dict(file="audit_online_raw.npz",
                              sha256=sha256_file("audit_online_raw.npz"))
    with open("audit_online_results.json", "w") as f:
        json.dump(RES, f, indent=1, default=float)


def forward(model, ids, mask=None):
    t = torch.tensor([list(ids)], dtype=torch.long, device=DEV)
    m = (torch.ones_like(t) if mask is None
         else torch.tensor([list(mask)], dtype=torch.long, device=DEV))
    with torch.no_grad():
        return model(input_ids=t, attention_mask=m, use_cache=False,
                     output_pldr_attentions=True, return_dict=True)


def forward_batch(model, ids, mask=None):
    t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=DEV)
    m = (torch.ones_like(t) if mask is None
         else torch.as_tensor(np.asarray(mask), dtype=torch.long,
                              device=DEV))
    with torch.no_grad():
        return model(input_ids=t, attention_mask=m, use_cache=False,
                     return_dict=True)


def final_logits(model, ids):
    return forward(model, ids).logits[0, -1].to(torch.float64).cpu().numpy()


def greedy_extend(model, ctx, steps):
    """Greedy continuation via repeated S=t prefix calls (uncached)."""
    out = []
    cur = list(ctx)
    for _ in range(steps):
        nxt = int(np.argmax(final_logits(model, cur)))
        out.append(nxt)
        cur.append(nxt)
    return out


def sequential_logprobs(model, ctx, cont):
    """Per-token log p(y_i | x, y_{:i}) via one final-row prefix call
    per candidate token (log-softmax in float64 on float32 logits)."""
    lps = []
    for i in range(len(cont)):
        row = final_logits(model, seq_prefix(ctx, cont, i))
        row = row - row.max()
        lps.append(float(row[cont[i]] - np.log(np.exp(row).sum())))
    return np.asarray(lps, dtype=np.float64)


def block_logprobs(model, ctx, cont):
    """Per-token block log-probabilities from ONE call on (x+y)[:-1]."""
    out = forward(model, block_input(ctx, cont))
    rows = out.logits[0].to(torch.float64).cpu().numpy()
    rows = rows - rows.max(axis=1, keepdims=True)
    rows = rows - np.log(np.exp(rows).sum(axis=1, keepdims=True))
    return gather_block_logprobs(rows, len(ctx), cont)


def deductive_deltas(out_a, out_b):
    per_layer = []
    for la, lb in zip(out_a.pldr_attentions, out_b.pldr_attentions):
        rec = {}
        for idx, name in DEDUCTIVE:
            rec[name] = delta_stats(la[idx].cpu().numpy(),
                                    lb[idx].cpu().numpy())
        per_layer.append(rec)
    return per_layer


def run_checkpoint(spec):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tag, mid, rev = spec["tag"], spec["model"], spec["rev"]
    log(f"== {tag}: {mid} @ {rev[:12]} ==")
    modeling = huggingface_hub.hf_hub_download(mid, "modeling_pldrllm.py",
                                               revision=rev)
    weights = huggingface_hub.hf_hub_download(mid, "model.safetensors",
                                              revision=rev)
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        mid, revision=rev, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager").to(DEV)
    model.eval()
    vocab = int(model.config.vocab_size)
    C = dict(model=mid, revision=rev,
             modeling_file_sha256=sha256_file(modeling),
             weights_sha256=sha256_file(weights), vocab_size=vocab)

    # ---------------- O1: historical-row context interaction ----------------
    full_a = FIXED_PREFIX + FIXED_SUFF_A
    full_b = FIXED_PREFIX + FIXED_SUFF_B
    suffix_mask = [1] * len(FIXED_PREFIX) + [0] * len(FIXED_SUFF_A)
    out_a = forward(model, full_a)
    out_b = forward(model, full_b)
    out_am = forward(model, full_a, suffix_mask)
    out_bm = forward(model, full_b, suffix_mask)
    la = out_a.logits[0].to(torch.float64).cpu().numpy()
    lb = out_b.logits[0].to(torch.float64).cpu().numpy()
    lam = out_am.logits[0].to(torch.float64).cpu().numpy()
    lbm = out_bm.logits[0].to(torch.float64).cpu().numpy()
    RAW[f"{tag}_o1_fixed_logits_a"] = la
    RAW[f"{tag}_o1_fixed_logits_b"] = lb
    RAW[f"{tag}_o1_fixed_logits_a_masked"] = lam
    RAW[f"{tag}_o1_fixed_logits_b_masked"] = lbm
    fixed = dict(
        tokens=dict(prefix=FIXED_PREFIX, suffix_a=FIXED_SUFF_A,
                    suffix_b=FIXED_SUFF_B),
        suffix_change=[delta_stats(la[t], lb[t])
                       for t in range(len(FIXED_PREFIX))],
        masked_suffix_change=[delta_stats(lam[t], lbm[t])
                              for t in range(len(FIXED_PREFIX))],
        deductive_per_layer=deductive_deltas(out_a, out_b))

    ra, rb = random_pairs(vocab, PROBE_TRIALS, PROBE_LEN, PROBE_PREFIX,
                          PROBE_SEED)
    rmask = np.zeros_like(ra)
    rmask[:, :PROBE_PREFIX] = 1
    lra = forward_batch(model, ra).logits.to(torch.float64).cpu().numpy()
    lrb = forward_batch(model, rb).logits.to(torch.float64).cpu().numpy()
    lram = forward_batch(model, ra, rmask).logits.to(
        torch.float64).cpu().numpy()
    lrbm = forward_batch(model, rb, rmask).logits.to(
        torch.float64).cpu().numpy()
    RAW[f"{tag}_o1_rand_a"] = ra
    RAW[f"{tag}_o1_rand_b"] = rb
    RAW[f"{tag}_o1_rand_logits_a"] = lra[:, :PROBE_PREFIX, :]
    RAW[f"{tag}_o1_rand_logits_b"] = lrb[:, :PROBE_PREFIX, :]
    RAW[f"{tag}_o1_rand_logits_a_masked"] = lram[:, :PROBE_PREFIX, :]
    RAW[f"{tag}_o1_rand_logits_b_masked"] = lrbm[:, :PROBE_PREFIX, :]
    randomized = dict(
        seed=PROBE_SEED, trials=PROBE_TRIALS, sequence_length=PROBE_LEN,
        prefix_length=PROBE_PREFIX,
        suffix_change=delta_stats(lra[:, :PROBE_PREFIX, :],
                                  lrb[:, :PROBE_PREFIX, :]),
        masked_suffix_change=delta_stats(lram[:, :PROBE_PREFIX, :],
                                         lrbm[:, :PROBE_PREFIX, :]))
    C["o1_prefix_rows"] = dict(fixed_pair=fixed, randomized=randomized)

    # ---------------- O2: online determinism ----------------
    reps = []
    for pfx in (full_a, full_a[:4], (FIXED_PREFIX + FIXED_SUFF_B)[:5]):
        r1 = final_logits(model, pfx)
        r2 = final_logits(model, pfx)
        reps.append(delta_stats(r1, r2))
    C["o2_repeat_call_bitwise"] = dict(
        comparisons=reps,
        all_zero=bool(all(r["nonzero"] == 0 for r in reps)))

    # ---------------- O3: padding at the Gram boundary ----------------
    pads = []
    for ptxt in PROMPTS[:2]:
        ids = tok(ptxt, return_tensors="pt").input_ids[0].tolist()
        ref_out = forward(model, ids)
        ref = ref_out.logits[0, -1].to(torch.float64).cpu().numpy()
        rec = dict(prompt_tokens=len(ids))
        for pad_tok in PAD_CONTENTS:
            padded = ids + [pad_tok] * PAD_LEN
            mask = [1] * len(ids) + [0] * PAD_LEN
            pout = forward(model, padded, mask)
            prow = pout.logits[0, len(ids) - 1].to(
                torch.float64).cpu().numpy()
            rec[f"pad_content_{pad_tok}"] = dict(
                final_real_row_delta=delta_stats(ref, prow),
                glm_delta=delta_stats(
                    ref_out.pldr_attentions[-1][5].cpu().numpy(),
                    pout.pldr_attentions[-1][5].cpu().numpy()))
        strip = final_logits(model, (ids + [PAD_CONTENTS[0]] * PAD_LEN)
                             [:len(ids)])
        rec["strip_restores_bitwise"] = bool(
            delta_stats(ref, strip)["nonzero"] == 0)
        pads.append(rec)
    C["o3_padding"] = dict(
        pad_len=PAD_LEN, pad_contents=list(PAD_CONTENTS), prompts=pads,
        note=("padded query rows participate in the query Gram in the "
              "current implementations: deltas document that behavior; "
              "the deployed contract is the unpadded S=t interface"))

    # ---------------- O4: sequential vs block candidate scores ----------------
    rng = np.random.default_rng(SUITE_SEED)
    suite = []
    tok_diffs, cand_diffs, flips, disc = [], [], [], []
    single_ok = True
    for pi, ptxt in enumerate(PROMPTS):
        ctx = tok(ptxt, return_tensors="pt").input_ids[0].tolist()
        row0 = final_logits(model, ctx)
        top2 = np.argsort(row0)[::-1][:2].tolist()
        g8 = greedy_extend(model, ctx, GREEDY_LONG)
        cands = [g8[:4], g8,
                 [top2[1]] + greedy_extend(model, ctx + [top2[1]], 3),
                 rng.integers(0, vocab, size=4).tolist()]
        singles = [[int(t)] for t in np.argsort(row0)[::-1][:4]]
        srec = dict(prompt_index=pi, context_tokens=len(ctx),
                    candidates=[list(map(int, c)) for c in cands])
        sseq = np.zeros(len(cands))
        sblk = np.zeros(len(cands))
        for ci, cand in enumerate(cands):
            cand = list(map(int, cand))
            lps = sequential_logprobs(model, ctx, cand)
            lpb = block_logprobs(model, ctx, cand)
            RAW[f"{tag}_o4_p{pi}_c{ci}_seq"] = lps
            RAW[f"{tag}_o4_p{pi}_c{ci}_blk"] = lpb
            tok_diffs.extend(np.abs(lps - lpb).tolist())
            cand_diffs.append(abs(float(lps.sum() - lpb.sum())))
            sseq[ci], sblk[ci] = lps.sum(), lpb.sum()
        cmp = rank_comparison(sseq, sblk)
        flips.append(cmp["argmax_flip"])
        disc.append(cmp["discordant_pairs"])
        srec["rank_comparison"] = cmp
        for cand in singles:
            lps = sequential_logprobs(model, ctx, cand)
            lpb = block_logprobs(model, ctx, cand)
            if float(abs(lps[0] - lpb[0])) != 0.0:
                single_ok = False
        suite.append(srec)
    RAW[f"{tag}_o4_token_absdiffs"] = np.asarray(tok_diffs)
    RAW[f"{tag}_o4_candidate_absdiffs"] = np.asarray(cand_diffs)
    C["o4_score_equivalence"] = dict(
        suite_seed=SUITE_SEED, contexts=len(PROMPTS),
        multi_token_candidates_per_context=4,
        per_token_absdiff=summarize(tok_diffs),
        per_candidate_absdiff=summarize(cand_diffs),
        argmax_flips=int(sum(flips)),
        discordant_pairs_total=int(sum(disc)),
        pairs_total=int(len(PROMPTS) * 6),
        single_token_exact_coincidence=bool(single_ok),
        instances=suite,
        formula=("sequential: sum_i log p(y_i | x, y_{:i}) via one "
                 "final-row prefix call per token; block: one call on "
                 "(x+y)[:-1], rows n-1+i; log-softmax in float64 on "
                 "float32 logits"))
    RES[tag] = C
    log(f"  o4 per-token max {C['o4_score_equivalence']['per_token_absdiff']['max']:.3e}"
        f"  flips {C['o4_score_equivalence']['argmax_flips']}"
        f"  single-token exact {single_ok}")
    del model
    return C


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))
                              ).stdout.strip()
    except Exception:
        return "unknown"


RES["environment"] = dict(
    device=DEV, python=".".join(map(str, __import__("sys").version_info[:3])),
    torch=torch.__version__, transformers=transformers.__version__,
    numpy=np.__version__, huggingface_hub=huggingface_hub.__version__,
    threads=16, seed=0, deterministic=True, dtype="float32",
    attn_implementation="eager", audit_commit=git_commit(),
    gpu=(torch.cuda.get_device_name(0) if DEV.startswith("cuda")
         else None))

for spec in CHECKPOINTS:
    run_checkpoint(spec)

save()
log("wrote audit_online_results.json + audit_online_raw.npz")

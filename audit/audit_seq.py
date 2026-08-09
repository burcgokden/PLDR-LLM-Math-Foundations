#!/usr/bin/env python3
"""Sequential-validation audit of the released PLDR-LLM checkpoints,
backing the paper's appendix subsections on held-out sequential
negative log-likelihood versus the blockwise training objective, and
on sequential versus one-pass block scoring of real benchmark items.

TruthfulQA protocol: the published zero-shot list uses truthfulqa_mc2
--- multiple true answers, scored by the normalized probability mass
on the true choices (the pinned harness group and the source papers
both use it) --- so the published-task section renders mc2_targets
and evaluates that exact metric under both scoring protocols
(mc2_score, a tested transcription of the pinned
process_results_mc2).  A single-gold truthfulqa_mc1 probe runs on the
SAME seeded dataset rows as an explicitly auxiliary diagnostic under
s2_auxiliary, outside the published-task count (its argmax structure
adds discordant-pair coverage the MC2 metric cannot provide).

Design rule (shared by all the audits here): every reported
quantity is either the named quantity computed directly, or is
explicitly labeled a bound/proxy with its formula.  Raw per-token and
per-candidate arrays behind every summary are stored in
audit_seq_raw.npz (SHA-256 recorded in the JSON); test_audit_seq.py
re-derives every JSON summary offline from those arrays and verifies
the request templates and the scoring index algebra model-free,
including analytic prefix-consistent and global toy-decoder oracles.

Sections (run for BOTH released checkpoints):
  S1. Held-out blockwise cross-entropy versus sequential
      autoregressive NLL: seeded disjoint token windows from a pinned
      public corpus; ONE full-block call per window (the training
      objective's per-row quantity) against one final-row S=t call per
      position (the deployed chain-rule quantity); aggregates, the
      per-token gap distribution, its sign structure, and its
      position dependence.
  S2. Real benchmark items, sequential versus one-pass block scores:
      seeded fixed-size samples of actual items from the published
      zero-shot task list, with request templates transcribed from the
      pinned evaluation-harness fork (verified by unit tests) and the
      wrapper's encode-pair convention; for the seven argmax tasks:
      per-token and per-candidate score gaps (distributions, and by
      candidate length), raw and length-normalized argmax changes,
      discordant candidate pairs, accuracy under both protocols, block
      decision-margin distributions, and the count of items whose
      score gap reaches half the decision margin; for truthfulqa_mc2:
      the published probability-mass metric under both protocols with
      per-item metric gaps, plus candidate-argmax comparisons labeled
      diagnostic.  An auxiliary truthfulqa_mc1 probe (same dataset
      rows, single-gold argmax structure) is reported separately
      under s2_auxiliary.

Protocol formulas:
  sequential: sum_i log p(y_i | x, y_{:i}) via one final-row prefix
  call per candidate token; block: one call on (x+y)[:-1], rows
  n-1+i; log-softmax in float64 on float32 logits.  Single-token
  candidates coincide exactly by construction and are asserted.

Environment:
  AUDIT_SEQ_DEVICE     cuda|cpu (default: cuda if available; float32,
                       eager attention either way)
  AUDIT_MODEL_5 / AUDIT_REV_5    audited checkpoint (SOC-110M-5 pin)
  AUDIT_MODEL_1 / AUDIT_REV_1    contrast checkpoint (SOC-110M-1 pin)
Determinism: the cuBLAS workspace is pinned and deterministic
algorithms are forced, as in the main audit; back-to-back runs on
a fixed device are expected to be byte-identical, and individual float
values are device-configuration sensitive as discussed in the paper's
audit scope.  Datasets are loaded as parquet files pinned by full
revision hash (SHA-256 of each file recorded); with a warm cache no
network access is needed.
Writes audit_seq_results.json + audit_seq_raw.npz (no wall-clock
fields, so back-to-back runs can be compared byte-for-byte).
"""
import json
import os
import subprocess
import warnings

import numpy as np
import pyarrow.parquet as pq
import torch
import transformers
import huggingface_hub

from audit_online_lib import (seq_prefix, block_input,
                              gather_block_logprobs, sha256_file)
from audit_seq_lib import (RENDERERS, shift_trailing_space,
                           encode_pair_split, disjoint_window_starts,
                           derive_task_summary, derive_mc2_summary,
                           derive_wikitext_summary)

warnings.filterwarnings("ignore")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(16)

DEV = os.environ.get("AUDIT_SEQ_DEVICE",
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

# ---------------- pinned datasets (full revision hashes) ----------------

WIKI = dict(repo="Salesforce/wikitext",
            revision="b08601e04326c79dfdd32d625aee71d232d685c3",
            file="wikitext-2-raw-v1/validation-00000-of-00001.parquet",
            split="validation")

# seed_index feeds the per-task sampling stream [BENCH_SEED, seed_index],
# fixing each task's sampled rows independently of its position in this
# list; the two TruthfulQA variants share seed_index 7, so the published
# mc2 task and the auxiliary mc1 probe score the SAME seeded dataset
# rows.  Do NOT renumber: the shipped results sampled these streams.
BENCH = [
    dict(task="arc_easy", repo="allenai/ai2_arc",
         revision="210d026faf9955653af8916fad021475a3f00453",
         file="ARC-Easy/test-00000-of-00001.parquet", split="test",
         seed_index=0),
    dict(task="arc_challenge", repo="allenai/ai2_arc",
         revision="210d026faf9955653af8916fad021475a3f00453",
         file="ARC-Challenge/test-00000-of-00001.parquet", split="test",
         seed_index=1),
    dict(task="hellaswag", repo="Rowan/hellaswag",
         revision="218ec52e09a7e7462a5400043bb9a69a41d06b76",
         file="data/validation-00000-of-00001.parquet",
         split="validation", seed_index=2),
    dict(task="piqa", repo="ybisk/piqa",
         revision="142c51238b3ca2bc61e9a075913871b8b600e8e1",
         file="plain_text/validation/0000.parquet", split="validation",
         seed_index=3),
    dict(task="openbookqa", repo="allenai/openbookqa",
         revision="388097ea7776314e93a529163e0fea805b8a6454",
         file="main/test-00000-of-00001.parquet", split="test",
         seed_index=4),
    dict(task="social_iqa", repo="allenai/social_i_qa",
         revision="537a2ec8ec565adc0b70b70752893e59e024df26",
         file="default/validation/0000.parquet", split="validation",
         seed_index=5),
    dict(task="winogrande", repo="allenai/winogrande",
         revision="01e74176c63542e6b0bcb004dcdea22d94fb67b5",
         file="winogrande_xl/validation-00000-of-00001.parquet",
         split="validation", seed_index=6),
    dict(task="truthfulqa_mc2", repo="truthfulqa/truthful_qa",
         revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490",
         file="multiple_choice/validation-00000-of-00001.parquet",
         split="validation", seed_index=7, kind="mc2"),
    dict(task="truthfulqa_mc1", repo="truthfulqa/truthful_qa",
         revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490",
         file="multiple_choice/validation-00000-of-00001.parquet",
         split="validation", seed_index=7, role="auxiliary"),
]

WIKI_SEED = 20260807
WIKI_WINDOWS = 48
WIKI_TOKENS = 257          # 256 scored positions per window
BENCH_SEED = 20260807
BENCH_K = 100              # sampled items per task
MAX_LEN = 1024

RES = {}
RAW = {}


def log(*a):
    print(*a, flush=True)


def save():
    np.savez_compressed("audit_seq_raw.npz", **RAW)
    RES["raw_records"] = dict(file="audit_seq_raw.npz",
                              sha256=sha256_file("audit_seq_raw.npz"))
    with open("audit_seq_results.json", "w") as f:
        json.dump(RES, f, indent=1, default=float)


def fetch(spec):
    """Pinned parquet -> (rows, file_sha256); by-hash revisions load
    from the local cache without network."""
    p = huggingface_hub.hf_hub_download(spec["repo"], spec["file"],
                                        repo_type="dataset",
                                        revision=spec["revision"])
    return pq.read_table(p).to_pylist(), sha256_file(p)


def forward(model, ids):
    t = torch.tensor([list(ids)], dtype=torch.long, device=DEV)
    m = torch.ones_like(t)
    with torch.no_grad():
        return model(input_ids=t, attention_mask=m, use_cache=False,
                     return_dict=True)


def final_logprob(model, ids, target):
    """log p(target | ids) from the final row of one S=t call."""
    row = forward(model, ids).logits[0, -1].to(torch.float64).cpu().numpy()
    row = row - row.max()
    return float(row[target] - np.log(np.exp(row).sum()))


def sequential_logprobs(model, ctx, cont):
    return np.asarray([final_logprob(model, seq_prefix(ctx, cont, i),
                                     cont[i])
                       for i in range(len(cont))], dtype=np.float64)


def block_logprobs(model, ctx, cont):
    out = forward(model, block_input(ctx, cont))
    rows = out.logits[0].to(torch.float64).cpu().numpy()
    rows = rows - rows.max(axis=1, keepdims=True)
    rows = rows - np.log(np.exp(rows).sum(axis=1, keepdims=True))
    return gather_block_logprobs(rows, len(ctx), cont)


def block_row_logprobs(model, window):
    """All per-position block log-probabilities for one held-out
    window: one call on window[:-1]; row t scores window[t+1]."""
    out = forward(model, window[:-1])
    rows = out.logits[0].to(torch.float64).cpu().numpy()
    rows = rows - rows.max(axis=1, keepdims=True)
    rows = rows - np.log(np.exp(rows).sum(axis=1, keepdims=True))
    return np.asarray([rows[t][window[t + 1]]
                       for t in range(len(window) - 1)], dtype=np.float64)


def load_model(spec):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid, rev = spec["model"], spec["rev"]
    modeling = huggingface_hub.hf_hub_download(mid, "modeling_pldrllm.py",
                                               revision=rev)
    weights = huggingface_hub.hf_hub_download(mid, "model.safetensors",
                                              revision=rev)
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        mid, revision=rev, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager").to(DEV)
    model.eval()
    prov = dict(model=mid, revision=rev,
                modeling_file_sha256=sha256_file(modeling),
                weights_sha256=sha256_file(weights),
                vocab_size=int(model.config.vocab_size))
    return model, tok, prov


def enc(tok, s):
    return tok(s, add_special_tokens=False).input_ids


def run_checkpoint(spec, wiki_ids, bench_data):
    tag = spec["tag"]
    log(f"== {tag}: {spec['model']} @ {spec['rev'][:12]} ==")
    model, tok, C = load_model(spec)

    # ---------------- S1: held-out block CE vs sequential NLL ----------------
    starts = disjoint_window_starts(len(wiki_ids), WIKI_WINDOWS,
                                    WIKI_TOKENS, WIKI_SEED)
    seq_lp = np.zeros((WIKI_WINDOWS, WIKI_TOKENS - 1))
    blk_lp = np.zeros((WIKI_WINDOWS, WIKI_TOKENS - 1))
    for w, st in enumerate(starts):
        win = wiki_ids[st:st + WIKI_TOKENS]
        blk_lp[w] = block_row_logprobs(model, win)
        seq_lp[w] = [final_logprob(model, win[:t + 1], win[t + 1])
                     for t in range(WIKI_TOKENS - 1)]
        if (w + 1) % 8 == 0:
            log(f"  s1 window {w + 1}/{WIKI_WINDOWS}")
    RAW[f"{tag}_wikitext_seq_lp"] = seq_lp
    RAW[f"{tag}_wikitext_blk_lp"] = blk_lp
    RAW[f"{tag}_wikitext_starts"] = np.asarray(starts, dtype=np.int64)
    C["s1_heldout"] = dict(
        dataset=WIKI["repo"], revision=WIKI["revision"],
        file=WIKI["file"], file_sha256=RES["datasets"]["wikitext"],
        split=WIKI["split"], stream_tokens=int(len(wiki_ids)),
        seed=WIKI_SEED, windows=WIKI_WINDOWS, window_tokens=WIKI_TOKENS,
        formula=("block: one call per window on window[:-1], row t "
                 "scores window[t+1] (the training objective's per-row "
                 "quantity); sequential: one final-row call on "
                 "window[:t+1] per position (the deployed chain-rule "
                 "quantity); log-softmax in float64 on float32 logits; "
                 "no BOS/EOS insertion (wrapper convention)"),
        summary=derive_wikitext_summary(seq_lp, blk_lp))
    log(f"  s1 block CE/token {C['s1_heldout']['summary']['block_ce_per_token']:.4f}"
        f"  seq NLL/token {C['s1_heldout']['summary']['sequential_nll_per_token']:.4f}"
        f"  max |gap| {C['s1_heldout']['summary']['per_token_absdiff']['max']:.3e}")

    # ---------------- S2: real benchmark items ----------------
    C["s2_benchmarks"] = {}
    C["s2_auxiliary"] = {}
    for spec_b, rows, fsha, sample in bench_data:
        task = spec_b["task"]
        kind = spec_b.get("kind", "argmax")
        render = RENDERERS[task]
        seq_flat, blk_flat = [], []
        cand_lens, item_ncands, char_lens, ctx_lens = [], [], [], []
        gold, labels = [], []
        boundary_mismatches = 0
        for ri in sample:
            r = render(rows[ri])
            item_ncands.append(len(r["requests"]))
            if kind == "mc2":
                labels.extend(r["labels"])
            else:
                gold.append(r["gold"])
            char_lens.extend(r["choice_char_lens"])
            for ctx_s, cont_s in r["requests"]:
                ctx_s, cont_s = shift_trailing_space(ctx_s, cont_s)
                whole = enc(tok, ctx_s + cont_s)
                assert len(whole) <= MAX_LEN, "request exceeds context"
                ctx_ids, cont_ids, ok = encode_pair_split(
                    whole, enc(tok, ctx_s))
                boundary_mismatches += not ok
                lps = sequential_logprobs(model, ctx_ids, cont_ids)
                lpb = block_logprobs(model, ctx_ids, cont_ids)
                seq_flat.extend(lps.tolist())
                blk_flat.extend(lpb.tolist())
                cand_lens.append(len(cont_ids))
                ctx_lens.append(len(ctx_ids))
        RAW[f"{tag}_{task}_seq_flat"] = np.asarray(seq_flat)
        RAW[f"{tag}_{task}_blk_flat"] = np.asarray(blk_flat)
        RAW[f"{tag}_{task}_cand_lens"] = np.asarray(cand_lens,
                                                   dtype=np.int64)
        RAW[f"{tag}_{task}_item_ncands"] = np.asarray(item_ncands,
                                                     dtype=np.int64)
        RAW[f"{tag}_{task}_char_lens"] = np.asarray(char_lens)
        RAW[f"{tag}_{task}_ctx_lens"] = np.asarray(ctx_lens,
                                                  dtype=np.int64)
        RAW[f"{tag}_{task}_item_rows"] = np.asarray(sample,
                                                   dtype=np.int64)
        if kind == "mc2":
            RAW[f"{tag}_{task}_labels"] = np.asarray(labels,
                                                    dtype=np.int64)
            summ = derive_mc2_summary(seq_flat, blk_flat, cand_lens,
                                      item_ncands, labels, char_lens)
        else:
            RAW[f"{tag}_{task}_gold"] = np.asarray(gold, dtype=np.int64)
            summ = derive_task_summary(seq_flat, blk_flat, cand_lens,
                                       item_ncands, gold, char_lens)
        role = spec_b.get("role", "published")
        rec = dict(
            dataset=spec_b["repo"], revision=spec_b["revision"],
            file=spec_b["file"], file_sha256=fsha, split=spec_b["split"],
            role=role,
            n_rows_total=len(rows), sample_seed=BENCH_SEED,
            sample_size=len(sample),
            boundary_mismatches=int(boundary_mismatches),
            summary=summ)
        if kind == "mc2":
            rec["metric"] = ("normalized probability mass on the true "
                             "choices (pinned process_results_mc2: "
                             "label vector split at its first 0; "
                             "softmax-equivalent max-subtracted "
                             "exponentiation)")
        dest = (C["s2_auxiliary"] if role == "auxiliary"
                else C["s2_benchmarks"])
        dest[task] = rec
        if kind == "mc2":
            log(f"  s2 {task}: mc2 blk/seq "
                f"{summ['mc2_block_mean']:.6f}/"
                f"{summ['mc2_sequential_mean']:.6f}"
                f"  max item gap "
                f"{summ['per_item_metric_absdiff']['max']:.3e}"
                f"  max cand gap "
                f"{summ['per_candidate_absdiff']['max']:.3e}")
        else:
            log(f"  s2 {task}{' (aux)' if role == 'auxiliary' else ''}: "
                f"acc blk/seq "
                f"{summ['acc_block']:.3f}/{summ['acc_sequential']:.3f}"
                f"  norm {summ['acc_norm_block']:.3f}/"
                f"{summ['acc_norm_sequential']:.3f}"
                f"  raw flips {summ['raw_argmax_changes']}"
                f"  max cand gap {summ['per_candidate_absdiff']['max']:.3e}")
    RES[tag] = C
    del model
    if DEV.startswith("cuda"):
        torch.cuda.empty_cache()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))
                              ).stdout.strip()
    except Exception:
        return "unknown"


def main():
    import pyarrow
    RES["environment"] = dict(
        device=DEV,
        python=".".join(map(str, __import__("sys").version_info[:3])),
        torch=torch.__version__, transformers=transformers.__version__,
        numpy=np.__version__,
        huggingface_hub=huggingface_hub.__version__,
        pyarrow=pyarrow.__version__, threads=16, seed=0,
        deterministic=True, dtype="float32",
        attn_implementation="eager", audit_commit=git_commit(),
        gpu=(torch.cuda.get_device_name(0) if DEV.startswith("cuda")
             else None))

    # tokenize the held-out stream once with the audited checkpoint's
    # tokenizer; both released checkpoints ship the same tokenizer and
    # the identity is asserted below.
    from transformers import AutoTokenizer
    tok5 = AutoTokenizer.from_pretrained(CHECKPOINTS[0]["model"],
                                         revision=CHECKPOINTS[0]["rev"])
    tok1 = AutoTokenizer.from_pretrained(CHECKPOINTS[1]["model"],
                                         revision=CHECKPOINTS[1]["rev"])
    wiki_rows, wiki_sha = fetch(WIKI)
    RES["datasets"] = dict(wikitext=wiki_sha)
    stream = "".join(r["text"] for r in wiki_rows)
    wiki_ids = enc(tok5, stream)
    probe = "Question: overlap check?\nAnswer: yes"
    assert enc(tok5, probe) == enc(tok1, probe), \
        "checkpoint tokenizers disagree"
    log(f"held-out stream: {len(wiki_rows)} rows, {len(wiki_ids)} tokens")

    bench_data = []
    for spec_b in BENCH:
        rows, fsha = fetch(spec_b)
        rng = np.random.default_rng([BENCH_SEED, spec_b["seed_index"]])
        n = len(rows)
        sample = sorted(int(i) for i in
                        rng.choice(n, size=min(BENCH_K, n),
                                   replace=False))
        bench_data.append((spec_b, rows, fsha, sample))
        log(f"{spec_b['task']}: {n} rows, sampling {len(sample)}")

    for spec in CHECKPOINTS:
        run_checkpoint(spec, wiki_ids, bench_data)

    save()
    log("wrote audit_seq_results.json + audit_seq_raw.npz")


if __name__ == "__main__":
    main()

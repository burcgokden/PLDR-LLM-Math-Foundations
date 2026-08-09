#!/usr/bin/env python3
"""Section I: order parameter with RMS vs SYMMETRIZED signed-mean
normalization, from two independent stochastic continuations per
prompt.  The signed statistic normalizes by 0.5*(|mu_1| + |mu_2|), a
symmetric adaptation of the source papers' |mu_1| convention -- report
it as "symmetrized signed-mean", never as the source normalization
(see audit_lib.order_pair).  Also
records the decoded continuation texts as a neutral human-inspectable
record (with repetition statistics; no quality claim is attached).
Merges results into audit_results.json and appends raw per-layer
values to audit_raw.npz (run audit.py first).

Environment:
  AUDIT_MODEL   checkpoint path or hub id (default fromthesky/PLDR-LLM-v51-SOC-110M-5)
  AUDIT_REV     model revision to pin (default the audited snapshot)
  AUDIT_DEVICE  cuda|cpu (default: cuda if available)
"""
import json, os, time, warnings
import numpy as np
import torch

from audit_lib import stats, order_pair, rep_stats, sha256_file, ap_tensor

warnings.filterwarnings("ignore")
# see audit.py: pin the cuBLAS workspace and force deterministic
# algorithms so repeated launches are bitwise stable (the order
# parameter sits at float resolution and is meaningless otherwise)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(16)
t0 = time.time()
MODEL = os.environ.get("AUDIT_MODEL", "fromthesky/PLDR-LLM-v51-SOC-110M-5")
REV = os.environ.get("AUDIT_REV", "de8e539c0ba1829072f4b8c2c5fae3bde0a3a2d2")
DEV = os.environ.get("AUDIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
def npy(t): return t.detach().cpu().numpy()

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, trust_remote_code=True,
                                             dtype=torch.float32, attn_implementation="eager")
model.eval()
# see audit.py: required under transformers >= 5, harmless under the
# pinned 4.x; runs before .to(DEV) so rebuilt buffers land on the device
for m in model.modules():
    if hasattr(m, "rope_init") and callable(m.rope_init):
        m.rope_init()
model.to(DEV)
L = model.config.num_hidden_layers

PROMPTS = [
 "The self-organized critical state of a sandpile is reached when the slope of the pile fluctuates around a stationary value, and avalanches of all sizes occur without a characteristic scale.",
 "In linear algebra, the singular value decomposition expresses any real matrix as a product of an orthogonal matrix, a diagonal matrix of nonnegative singular values, and another orthogonal matrix.",
 "The movie was a complete waste of time. The acting felt wooden, the plot made no sense after the first act, and the ending was both predictable and unearned. I would not recommend it to anyone.",
 "Once upon a time, in a village at the edge of a great forest, there lived a clockmaker whose clocks always ran a little fast, as if they were impatient for the future to arrive.",
 "To bake a simple sourdough loaf, combine flour, water, salt, and an active starter; fold the dough every half hour, proof it overnight in the refrigerator, and bake it in a covered pot.",
]

def sample_gen(ids, seed, T=64):
    # sampling happens on CPU probabilities with a seeded CPU generator
    g = torch.Generator().manual_seed(seed)
    seq = ids
    with torch.no_grad():
        out = model(input_ids=seq, use_cache=True)
        past = out.past_key_values
        for _ in range(T):
            probs = torch.softmax(out.logits[:, -1, :], dim=-1)[0].cpu()
            nxt = torch.multinomial(probs, 1, generator=g).view(1, 1).to(DEV)
            seq = torch.cat([seq, nxt], dim=1)
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
    return seq

op = {k: {"rms": [], "signed": []} for k in ("A", "ALM", "AP", "GLM")}
continuations = []
for pi, ptxt in enumerate(PROMPTS):
    ids = tok(ptxt, return_tensors="pt").input_ids.to(DEV)
    S0 = ids.shape[1]
    seqs = [sample_gen(ids, seed=100 + r) for r in range(2)]
    outs = []
    for r, s in enumerate(seqs):
        text = tok.decode(s[0, S0:], skip_special_tokens=True)
        continuations.append(dict(prompt=pi, seed=100 + r, text=text,
                                  rep_stats=rep_stats(text)))
        log(f"prompt {pi} seed {100+r}: {text[:90]}...")
        with torch.no_grad():
            outs.append(model(input_ids=s, use_cache=False, output_pldr_attentions=True))
    for li in range(L):
        t1 = outs[0].pldr_attentions[li]
        t2 = outs[1].pldr_attentions[li]
        # t1[2] is the raw exponent parameter [H, dk, dk] with NO batch
        # axis; indexing it like a batch broadcasts head 0's matrix to
        # every head.  ap_tensor shape-checks the head-for-head
        # pairing.
        pw = npy(t1[2])
        for key, i1 in (("A", 0), ("ALM", 1), ("GLM", 5)):
            r, sgn = order_pair(npy(t1[i1][0]), npy(t2[i1][0]))
            op[key]["rms"].append(r)
            op[key]["signed"].append(sgn)
        ap1 = ap_tensor(npy(t1[1][0]), pw)
        ap2 = ap_tensor(npy(t2[1][0]), pw)
        r, sgn = order_pair(ap1, ap2)
        op["AP"]["rms"].append(r)
        op["AP"]["signed"].append(sgn)
    log(f"order-parameter prompt {pi} done")

R = json.load(open("audit_results.json"))
R["order_parameter"] = {k: dict(rms=stats(v["rms"]), signed=stats(v["signed"])) for k, v in op.items()}
R["order_parameter_note"] = (
    "rms = RMS-normalized piecewise statistic per the paper's "
    "definition; signed = SYMMETRIZED signed-mean normalization "
    "0.5*(|mu_1|+|mu_2|) in the denominator, a symmetric adaptation of "
    "the source papers' convention (|mu_1| for run-1 vs run-2, |mu_C| "
    "for run-1 vs cached), not the literal source normalization; the "
    "two normalizations' maxima differ and must be reported "
    "separately, never as a single value 'under both'")
R["stochastic_continuations"] = continuations

# append raw per-(prompt,layer) values to the NPZ and refresh its hash
raw = dict(np.load("audit_raw.npz"))
for k, v in op.items():
    raw[f"op_{k}_rms"] = np.array(v["rms"])
    raw[f"op_{k}_signed"] = np.array(v["signed"])
np.savez_compressed("audit_raw.npz", **raw)
R["raw_records"] = dict(file="audit_raw.npz", sha256=sha256_file("audit_raw.npz"))

with open("audit_results.json", "w") as f:
    json.dump(R, f, indent=1, default=float)
log("I DONE")
print(json.dumps(R["order_parameter"], indent=1))

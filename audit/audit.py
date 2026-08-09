#!/usr/bin/env python3
"""Numerical audit of PLDR-LLM-v51-SOC-110M-5 backing the paper's
appendix "Numerical Audit on a Released Checkpoint".

Design rule: every reported quantity is either the named quantity
computed directly, or is explicitly labeled a bound/proxy with its
formula.  Raw per-instance arrays behind every summary are stored in
audit_raw.npz (SHA-256 recorded in the JSON) so the summaries can be
recomputed offline; see test_audit.py.

Semantics worth knowing before editing:

* Chain assembly: an operator perturbation enters at the ATTENTION
  OUTPUT of its layer, so assemble_chain traverses that layer's own
  post-attention remainder LLN1*(1+Lffn)*LLN2 before the downstream
  layer products; a miniature-decoder finite-difference oracle in
  test_audit.py FAILS under any downstream-only assembly.
* The learned exponent tensor P is a parameter of shape [H, dk, dk]
  with NO batch axis, unlike the batched activations returned next to
  it; A_P is built by the shape-checked audit_lib.ap_tensor
  everywhere, guarded by a sentinel semantic test (NumPy silently
  broadcasts several wrong pairings).
* The twirl energy decomposition keeps per-prompt arrays and reports
  all (prompt, layer) per-instance ratios alongside the prompt-pooled
  per-layer aggregates, each under its exact label (ratios do not
  commute with pooling).
* The random-initialization control is run under BOTH initialization
  laws, each labeled: the HF-port _init_weights (Xavier-uniform
  W, P, a) and the native SOC training law (Xavier-normal W, P, a).
* The chain's derivative factors are certified analytic envelopes on
  the sampled interval (never grid maxima), the chain/budget labels
  say "sample-extrema proxy", and eps_spec is a hypothetical
  single-rounding radius (the measured cached-vs-recomputed
  perturbation is 0 bitwise).  The NOTEARS h-floor dk*eps^2 and the
  normalized-log-loss floor log(1+eps^2) are distinct floors and are
  kept separate.

Sections:
  A. analytic twirl multipliers (no model)
  B. per-layer/head tensor spectra (A, A_LM), cross-head identity;
     the same battery at random initialization under BOTH laws: the
     HF-port _init_weights (Xavier-uniform W, P, a) and the native SOC
     training law (Xavier-normal W, P, a)
  C. epsilon-LayerNorm scale error: DIRECT comparison of LN(D) with
     LN(D/S) using the checkpoint's gamma/beta/eps, plus the analytic
     proxy eps/(2v) for reference
  D. twirl energy decomposition of BOTH the pre-rotation and the
     rotated position-resolved query aggregates (identifies what the
     actual finite-S twirl removed), per (prompt, layer) instance AND
     prompt-pooled per layer, plus a stationarity check
  D2. commutant residual ||G - Pi_comm G||_F/||G||_F of the trained
     G_LM per prompt/layer/head: occupancy of the absolute-position-
     sensitive directions counted by the positional-codimension
     corollary
  E. row-map contraction, measured on the COMPOSITION: full-composite
     Jacobian spectra at rows from every prompt, and empirical pairwise
     contraction ratios ||phi(r)-phi(r')||/||r-r'||; per-unit values
     retained only as a labeled decomposition diagnostic
  F. budget constants: C_G per head (at the measured A_LM floor and at
     the architectural floor 1e-9); sample-extrema per-layer chain with
     LayerNorm factors max|gamma|/sqrt(v_min+eps), gated-FFN factors by
     the product rule with certified derivative envelopes, and the
     hypothetical single-rounding radius eps_spec = 2^-24 * max ||G||_F
  G. cached vs uncached greedy decoding: logit deviations vs margins at
     every step, eps_G at EVERY step, the 2B < margin criterion, and
     the decoded continuation text as a neutral record with repetition
     statistics
  H. DAG-loss values of A_LM, A_P, G_LM per instance (positivity floor
     for the first two; G_LM measured, no floor asserted)
Writes audit_results.json + audit_raw.npz and prints a summary.  The
order parameter (section I) is computed by audit_h.py, which merges
into the same JSON/NPZ.  Run from this directory.

Environment:
  AUDIT_MODEL   checkpoint path or hub id (default fromthesky/PLDR-LLM-v51-SOC-110M-5)
  AUDIT_REV     model revision to pin (default the audited snapshot)
  AUDIT_DEVICE  cuda|cpu (default: cuda if available)
"""
import json, math, os, platform, subprocess, time, warnings, sys
import numpy as np
import torch
import transformers
import huggingface_hub

from audit_lib import (DK, THETA, build_freqs, C_theta_of, mult, numrank,
                       iswiglu_dprime_env, silu_dprime_env, stats,
                       assemble_chain, rope_basis, freq_energy,
                       commutant_residual, ap_tensor, layernorm_row,
                       ln_pair_error, ln_jac_bound, glu_jac_bound,
                       dag_loss_value, rep_stats, margin_ok, sha256_file)
import audit_lib

warnings.filterwarnings("ignore")
# Cross-launch reproducibility: cuBLAS reduction-order/algorithm choice
# can differ between process launches (clock-dependent heuristics), and
# quantities downstream of the elementwise power stage amplify the
# resulting ulp-level differences by orders of magnitude; the workspace
# pin plus deterministic algorithms make repeated runs bitwise stable
# (verified by back-to-back reruns; see the paper's audit appendix).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
t0 = time.time()
MODEL = os.environ.get("AUDIT_MODEL", "fromthesky/PLDR-LLM-v51-SOC-110M-5")
REV = os.environ.get("AUDIT_REV", "de8e539c0ba1829072f4b8c2c5fae3bde0a3a2d2")
DEV = os.environ.get("AUDIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(16)

RES = {}
RAW = {}
def save():
    np.savez_compressed("audit_raw.npz", **RAW)
    RES["raw_records"] = dict(file="audit_raw.npz",
                              sha256=sha256_file("audit_raw.npz"))
    with open("audit_results.json", "w") as f:
        json.dump(RES, f, indent=1, default=float)

def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def npy(t):
    return t.detach().cpu().numpy()

def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or None
    except Exception:
        return None

try:
    _model_file = huggingface_hub.hf_hub_download(MODEL, "modeling_pldrllm.py", revision=REV)
    _model_file_sha = sha256_file(_model_file)
except Exception:
    _model_file_sha = None

RES["environment"] = dict(
    model=MODEL, revision=REV, device=DEV,
    python=platform.python_version(), torch=torch.__version__,
    transformers=transformers.__version__, numpy=np.__version__,
    huggingface_hub=huggingface_hub.__version__, seed=0,
    cudnn_deterministic=True,
    modeling_file_sha256=_model_file_sha,
    audit_commit=git_commit(),
    gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None)

# ---------------- A. analytic twirl multipliers ----------------
freqs = build_freqs()
C_theta = C_theta_of(freqs)
twirl_tab = {}
for S in (64, 256, 1024):
    ms = np.array([mult(w, S) for w in freqs.values()])
    twirl_tab[S] = dict(worst=float(ms.max()), median=float(np.median(ms)),
                        frac_below_01=float((ms < 0.1).mean()),
                        frac_below_001=float((ms < 0.01).mean()))
RES["twirl_analytic"] = dict(C_theta=C_theta, C_over_1024=C_theta/1024,
                             table=twirl_tab,
                             slowest_inplane_mult_1024=mult(2*THETA[31], 1024),
                             worst_mult_1024=max(mult(w,1024) for w in freqs.values()))
log("A done. C_theta=%.2f" % C_theta)

# ---------------- load model ----------------
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, trust_remote_code=True,
                                             dtype=torch.float32, attn_implementation="eager")
model.eval()
# Under transformers >= 5 modules are materialized from the meta device,
# which leaves non-persistent buffers (the RoPE cache/angles, built in
# __init__) uninitialized; rebuild them before moving to the device.
# Harmless under the pinned 4.x, where the checkpoint loads correctly.
for m in model.modules():
    if hasattr(m, "rope_init") and callable(m.rope_init):
        m.rope_init()
model.to(DEV)
core = model.decoder
layers = core.dec_layers
L = len(layers); H = model.config.num_attention_heads
log(f"model loaded: L={L} H={H} device={DEV} rev={REV[:8]}")

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
NP = len(PROMPTS)

# ---------------- hooks ----------------
capt = {}
def mk_ln1_hook(li):        # D-tilde entering the density-operator LayerNorm
    def h(mod, args):
        capt.setdefault("D", {})[li] = args[0].detach()
    return h
def mk_plga_hook(li):       # rotated q,k,v entering the PLGA layer
    def h(mod, args, kwargs):
        capt.setdefault("qkv", {})[li] = tuple(t.detach() for t in args[0][:3])
    return h
def mk_rope_hook(li):       # pre-/post-rotation tensors; call 0 = q, call 1 = k
    def h(mod, args, out):
        lst = capt.setdefault("rope", {}).setdefault(li, [])
        if len(lst) < 2:
            lst.append((args[0].detach(), out.detach()))
    return h
def mk_declln_hook(li, which):   # decoder-layer LN inputs (pre) and LN1 output
    def pre(mod, args):
        capt.setdefault(f"ln{which}_in", {})[li] = args[0].detach()
    def fwd(mod, args, out):
        capt.setdefault("ln1_out", {})[li] = out.detach()
    return pre, fwd

hooks = []
for li, lay in enumerate(layers):
    hooks.append(lay.mha1.layernorm1.register_forward_pre_hook(mk_ln1_hook(li)))
    hooks.append(lay.mha1.plgatt_layer.register_forward_pre_hook(mk_plga_hook(li), with_kwargs=True))
    hooks.append(lay.mha1.rotary_embedding.register_forward_hook(mk_rope_hook(li)))
    pre1, fwd1 = mk_declln_hook(li, 1)
    pre2, _ = mk_declln_hook(li, 2)
    hooks.append(lay.layernorm1.register_forward_pre_hook(pre1))
    hooks.append(lay.layernorm1.register_forward_hook(fwd1))
    hooks.append(lay.layernorm2.register_forward_pre_hook(pre2))

def forward(ids, **kw):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False, output_pldr_attentions=True, **kw)

NONFINITE = {"count": 0}
def svdvals(M):
    return audit_lib.svdvals(M, NONFINITE)
def spec2(M):
    return audit_lib.spec2(M, NONFINITE)

Uc, omega = rope_basis()

# ------------- B,C,D,F(measure) over prompts; capture rows for E -------------
def spectra_battery(get_out, tag):
    """Runs the spectral battery (section B) for a callable get_out(ids)
    returning a model output with pldr_attentions; used for the trained
    model and for the random-initialization control."""
    B = dict(A_s1=[], A_s2_over_s1=[], A_numrank=[], ALM_s1=[], ALM_s2_over_s1=[],
             ALM_numrank=[], ALM_min=[], ALM_max=[], crosshead_rms=[],
             crosshead_rel=[], rowvar_ratio=[])
    rawA, rawW = [], []
    for pi, ptxt in enumerate(PROMPTS):
        ids = tok(ptxt, return_tensors="pt").input_ids.to(DEV)
        out = get_out(ids)
        for li in range(L):
            A = npy(out.pldr_attentions[li][0][0])
            AW = npy(out.pldr_attentions[li][1][0])
            for h in range(H):
                sA = svdvals(A[h]); sW = svdvals(AW[h])
                rawA.append(sA); rawW.append(sW)
                B["A_s1"].append(sA[0]); B["A_s2_over_s1"].append(sA[1]/sA[0]); B["A_numrank"].append(numrank(sA))
                B["ALM_s1"].append(sW[0]); B["ALM_s2_over_s1"].append(sW[1]/sW[0]); B["ALM_numrank"].append(numrank(sW))
                B["ALM_min"].append(AW[h].min()); B["ALM_max"].append(AW[h].max())
                rowmean = A[h].mean(axis=0)
                B["rowvar_ratio"].append(float(np.mean((A[h]-rowmean)**2)/np.mean((A[h]-A[h].mean())**2)))
            for h in range(1, H):
                d = A[h]-A[0]
                B["crosshead_rms"].append(float(np.sqrt(np.mean(d**2))))
                B["crosshead_rel"].append(float(np.sqrt(np.mean(d**2))/np.sqrt(np.mean(A[0]**2))))
    RAW[f"A_spectra_{tag}"] = np.array(rawA)
    RAW[f"ALM_spectra_{tag}"] = np.array(rawW)
    S = {k: stats(v) for k, v in B.items()}
    S["A_numrank_hist"] = {str(r): int((np.array(B["A_numrank"])==r).sum()) for r in sorted(set(B["A_numrank"]))}
    S["ALM_numrank_hist"] = {str(r): int((np.array(B["ALM_numrank"])==r).sum()) for r in sorted(set(B["ALM_numrank"]))}
    return S

C_ln = dict(rowvar_DS=[], direct_abs=[], direct_rel=[], proxy=[], n_zero_var=0)
# Energies are kept per (prompt, layer) so per-instance ratios can
# be formed; the prompt-pooled per-layer aggregates are derived from
# these at summary time (ratios do not commute with pooling, so the two
# reports are different statistics and both are labeled exactly).
D_tw = dict(E_unrot=np.zeros((NP, L, DK, DK)), E_rot=np.zeros((NP, L, DK, DK)),
            stationarity=[], rot_consistency=[])
F_meas = dict(qmax=np.zeros((L,H)), Knorm=np.zeros((L,H)), Vnorm=np.zeros((L,H)),
              Enorm=np.zeros((L,H)), U_obs=np.zeros((L,H)),
              ALM_lo=np.full((L,H), np.inf), ALM_hi=np.zeros((L,H)),
              Gnorm=np.zeros((L,H)), G_F=np.zeros((L,H)))
ffn_meas = dict(x1pre_max=np.zeros(L), x2_max=np.zeros(L), act_max=np.zeros(L),
                vmin_ln1=np.full(L, np.inf), vmin_ln2=np.full(L, np.inf))
dag_vals = dict(ALM=[], AP=[], GLM=[])
glm_comm = np.zeros((NP, L, H))            # D2: ||G - Pi_comm G||_F/||G||_F
phi_rows = {li: [] for li in range(L)}     # LN(D)-rows entering the res stack, per prompt
JAC_ROWS_PER_PROMPT = 16
POOL_ROWS_PER_PROMPT = 32

trained_spec = None
for pi, ptxt in enumerate(PROMPTS):
    capt.clear()
    ids = tok(ptxt, return_tensors="pt").input_ids.to(DEV)
    S = ids.shape[1]
    out = forward(ids)
    for li in range(L):
        A, AW, pw, a_vec, ba, avAp, E = out.pldr_attentions[li]
        # pw is the raw learned parameter [H, dk, dk] (NO batch axis,
        # unlike the batched activations around it); indexing it like a
        # batch silently broadcasts head 0's rows.  The shape assert
        # and ap_tensor guard this.
        A = npy(A[0]); AW = npy(AW[0]); pw_n = npy(pw); G = npy(avAp[0]); E = npy(E[0])
        assert pw_n.shape == AW.shape, (pw_n.shape, AW.shape)
        AP_l = ap_tensor(AW, pw_n)         # [H, dk, dk], head-for-head
        plga = layers[li].mha1.plgatt_layer
        Wl = npy(plga.Wlst); bl = npy(plga.blst)
        mha_ln = layers[li].mha1.layernorm1
        gam = npy(mha_ln.weight); bet = npy(mha_ln.bias); lne = mha_ln.eps
        for h in range(H):
            F_meas["ALM_lo"][li,h] = min(F_meas["ALM_lo"][li,h], AW[h].min())
            F_meas["ALM_hi"][li,h] = max(F_meas["ALM_hi"][li,h], AW[h].max())
            F_meas["Gnorm"][li,h] = max(F_meas["Gnorm"][li,h], spec2(G[h]))
            F_meas["G_F"][li,h] = max(F_meas["G_F"][li,h], float(np.linalg.norm(G[h])))
            pre = Wl[h] @ A[h] + bl[h]
            F_meas["U_obs"][li,h] = max(F_meas["U_obs"][li,h], np.abs(pre).max())
            # DAG-loss values of the three regularized tensors (H section)
            dag_vals["ALM"].append(dag_loss_value(AW[h]))
            dag_vals["AP"].append(dag_loss_value(AP_l[h]))
            dag_vals["GLM"].append(dag_loss_value(G[h]))
            # D2: commutant residual of the trained operator
            glm_comm[pi, li, h] = commutant_residual(G[h], Uc, omega)
        # C: direct epsilon-LayerNorm scale error on D-tilde rows
        Dt = npy(capt["D"][li][0])  # [H,64,64]
        for h in range(H):
            for r in Dt[h]:
                v = float((r / S).var())
                C_ln["rowvar_DS"].append(v)
                if v == 0.0:
                    C_ln["n_zero_var"] += 1
                d_abs, d_rel, prox = ln_pair_error(r, S, gam, bet, lne)
                C_ln["direct_abs"].append(d_abs)
                C_ln["direct_rel"].append(d_rel)
                C_ln["proxy"].append(prox)
        # D: twirl decomposition of pre-rotation vs rotated aggregates (head 0)
        q_pre = npy(capt["rope"][li][0][0][0])   # [S,H,dk] pre-rotation q
        q_rot = npy(capt["rope"][li][0][1][0])   # [S,H,dk] rotated q
        qp = q_pre[:, 0, :].astype(np.float64); qr = q_rot[:, 0, :].astype(np.float64)
        M_unrot = qp.T @ qp / S
        M_rot = qr.T @ qr / S
        En_u, wdiff = freq_energy(M_unrot, Uc, omega)
        En_r, _ = freq_energy(M_rot, Uc, omega)
        D_tw["E_unrot"][pi, li] = En_u
        D_tw["E_rot"][pi, li] = En_r
        half = S // 2
        M1 = qp[:half].T @ qp[:half] / half; M2 = qp[half:].T @ qp[half:] / (S - half)
        D_tw["stationarity"].append(float(np.linalg.norm(M1-M2)/np.linalg.norm((M1+M2)/2)))
        D_tw["rot_consistency"].append(float(np.linalg.norm(M_rot - Dt[0]/S)/np.linalg.norm(Dt[0]/S)))
        # E capture: rows entering the residual stack, this prompt
        with torch.no_grad():
            X = mha_ln(capt["D"][li][0]).reshape(-1, DK)
        idx = torch.randperm(X.shape[0])[:POOL_ROWS_PER_PROMPT]
        phi_rows[li].append(X[idx].detach())
        # F: q,k,v measured norms
        q,k,v = capt["qkv"][li]
        q=npy(q[0]); k=npy(k[0]); v=npy(v[0])
        for h in range(H):
            F_meas["qmax"][li,h] = max(F_meas["qmax"][li,h], np.linalg.norm(q[h],axis=1).max())
            F_meas["Knorm"][li,h] = max(F_meas["Knorm"][li,h], spec2(k[h]))
            F_meas["Vnorm"][li,h] = max(F_meas["Vnorm"][li,h], spec2(v[h]))
            F_meas["Enorm"][li,h] = max(F_meas["Enorm"][li,h], spec2(E[h]))
        # F: decoder-layer LN variance floors and FFN activation ranges
        for which in (1, 2):
            xin = npy(capt[f"ln{which}_in"][li][0]).astype(np.float64)  # [S,d_model]
            vrow = xin.var(axis=1)
            ffn_meas[f"vmin_ln{which}"][li] = min(ffn_meas[f"vmin_ln{which}"][li], vrow.min())
        out1 = capt["ln1_out"][li][0]           # [S,d_model] = FFN input
        ffn = layers[li].ffn
        with torch.no_grad():
            x1pre = ffn.gluw1(out1); x2 = ffn.gluw2(out1)
            act = ffn.activation(x1pre)
        ffn_meas["x1pre_max"][li] = max(ffn_meas["x1pre_max"][li], float(x1pre.abs().max()))
        ffn_meas["x2_max"][li] = max(ffn_meas["x2_max"][li], float(x2.abs().max()))
        ffn_meas["act_max"][li] = max(ffn_meas["act_max"][li], float(act.abs().max()))
    log(f"prompt {pi} (S={S}) processed")

RES["spectra"] = spectra_battery(forward, "trained")
log("spectra (trained) done")

# trained-vs-initialized controls: same battery at random
# initialization, under BOTH initialization laws
# (from_config alone samples the HF-port _init_weights, which
# draws W, P, a Xavier-UNIFORM, whereas the native SOC training code
# the checkpoint was trained from draws W, P, a Xavier-NORMAL with
# dense layers Xavier-uniform; the two controls are labeled by their
# laws so neither is passed off as the other).
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
PlgaCls = type(layers[0].mha1.plgatt_layer)

def run_init_control(seed, native_law, tag):
    torch.manual_seed(seed)
    im = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    im = im.to(torch.float32).eval()
    if native_law:
        # re-draw the metric-tensor parameters under the native SOC
        # law (Xavier-normal, torch fan convention on the stacked
        # [H, dk, dk] parameters; biases stay zero as in both codes)
        for m in im.modules():
            if isinstance(m, PlgaCls):
                for attr in ("Wlst", "pwlst", "alst"):
                    p = getattr(m, attr, None)
                    if p is not None:
                        torch.nn.init.xavier_normal_(p.data)
    for m in im.modules():
        if hasattr(m, "rope_init") and callable(m.rope_init):
            m.rope_init()
    im.to(DEV)
    def fwd(ids):
        with torch.no_grad():
            return im(input_ids=ids, use_cache=False, output_pldr_attentions=True)
    S = spectra_battery(fwd, tag)
    del im
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return S

RES["spectra_init"] = run_init_control(1, native_law=False, tag="init")
RES["spectra_init"]["note"] = (
    "random-initialization control under the HF-port _init_weights law "
    "(Xavier-uniform W, P, a and dense weights); NOT the native SOC "
    "training initialization -- see spectra_init_native")
log("spectra (random init, HF-port law) done")
RES["spectra_init_native"] = run_init_control(4, native_law=True, tag="init_native")
RES["spectra_init_native"]["note"] = (
    "random-initialization control with W, P, a re-drawn Xavier-normal "
    "per the native SOC training law (dense weights stay Xavier-uniform "
    "as in both codes); matches the law the audited checkpoint was "
    "actually trained from")
log("spectra (random init, native SOC law) done")

RES["nonfinite_matrices"] = NONFINITE["count"]
RES["ln_scale_error"] = dict(rowvar_DS=stats(C_ln["rowvar_DS"]),
                             direct_rel=stats(C_ln["direct_rel"]),
                             direct_abs=stats(C_ln["direct_abs"]),
                             proxy=stats(C_ln["proxy"]),
                             n_rows=len(C_ln["direct_rel"]),
                             n_zero_var_rows=C_ln["n_zero_var"])
RAW["rowvar_DS"] = np.array(C_ln["rowvar_DS"])
RAW["ln_direct_rel"] = np.array(C_ln["direct_rel"])
RAW["ln_direct_abs"] = np.array(C_ln["direct_abs"])
RAW["ln_proxy"] = np.array(C_ln["proxy"])

# D summaries: what did the actual twirl remove?  Two statistics, each
# under its exact label, because ratios do not commute with pooling:
# (a) per-instance ratios over all NP x L (prompt, layer) pairs;
# (b) prompt-pooled per-layer energy-weighted aggregates, whose range
# is over the L layers only.
zero = np.abs(omega[:, None] - omega[None, :]) < 1e-12   # zero-frequency mask
off = ~zero
inst = dict(supp=[], comm_u=[], comm_r=[])
for pi in range(NP):
    for li in range(L):
        Eu, Er = D_tw["E_unrot"][pi, li], D_tw["E_rot"][pi, li]
        inst["comm_u"].append(float(Eu[zero].sum()/Eu.sum()))
        inst["comm_r"].append(float(Er[zero].sum()/Er.sum()))
        inst["supp"].append(float(1 - Er[off].sum()/Eu[off].sum()))
pool = dict(supp=[], comm_u=[], comm_r=[])
Eu_l = D_tw["E_unrot"].sum(axis=0); Er_l = D_tw["E_rot"].sum(axis=0)
for li in range(L):
    pool["comm_u"].append(float(Eu_l[li][zero].sum()/Eu_l[li].sum()))
    pool["comm_r"].append(float(Er_l[li][zero].sum()/Er_l[li].sum()))
    pool["supp"].append(float(1 - Er_l[li][off].sum()/Eu_l[li][off].sum()))
tw = dict(
    per_instance=dict(
        offcomm_suppression=stats(inst["supp"]),
        commutant_frac_unrot=stats(inst["comm_u"]),
        commutant_frac_rot=stats(inst["comm_r"]),
        n_instances=NP * L,
        label="per-(prompt, layer) instance ratios; range over all "
              f"{NP * L} instances"),
    prompt_pooled_per_layer=dict(
        offcomm_suppression=stats(pool["supp"]),
        commutant_frac_unrot=stats(pool["comm_u"]),
        commutant_frac_rot=stats(pool["comm_r"]),
        label="prompt-pooled, per-layer energy ratios (energy-weighted "
              "over prompts); range over the five layers only"),
    stationarity_reldiff=stats(D_tw["stationarity"]),
    rot_vs_captured_D_reldiff=stats(D_tw["rot_consistency"]))
RES["twirl_empirical"] = tw
RAW["twirl_E_unrot"] = D_tw["E_unrot"]
RAW["twirl_E_rot"] = D_tw["E_rot"]
RAW["twirl_omega"] = omega

# D2 summary: occupancy of the absolute-position-sensitive directions
RES["glm_commutant_residual"] = dict(
    all=stats(glm_comm.ravel()),
    per_layer=[stats(glm_comm[:, li, :].ravel()) for li in range(L)],
    note="||G - Pi_comm G||_F/||G||_F of the trained G_LM per "
         "prompt/layer/head; Pi_comm projects onto the RoPE commutant "
         "(nonresonant spectrum) in the rotation eigenbasis; measures "
         "use of the dk^2-dk absolute-position-sensitive directions of "
         "the positional-codimension corollary on this checkpoint")
RAW["glm_comm_res"] = glm_comm
log("B,C,D,D2 done")
save()

# ---------------- E. composite row-map Jacobians + pairwise ratios ----------------
gen = torch.Generator().manual_seed(2)
jac = dict(comp_smax=np.zeros((L, JAC_ROWS_PER_PROMPT*NP)),
           comp_smin=np.zeros((L, JAC_ROWS_PER_PROMPT*NP)),
           unit_smax=np.zeros((L, len(layers[0].mha1.reslayerAs), JAC_ROWS_PER_PROMPT*NP)))
pair = dict(within=[], cross=[])
for li in range(L):
    mha = layers[li].mha1
    def phi(r):
        x = r.unsqueeze(0)
        for res in mha.reslayerAs:
            x = res.ResUnit(x)
        return x.squeeze(0)
    # composite Jacobians at rows from EVERY prompt
    col = 0
    for pi in range(NP):
        pool = phi_rows[li][pi]
        idx = torch.randperm(pool.shape[0], generator=gen)[:JAC_ROWS_PER_PROMPT]
        for r in pool[idx]:
            J = torch.autograd.functional.jacobian(phi, r).double()
            if not torch.isfinite(J).all():
                NONFINITE["count"] += 1
                J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
            s = svdvals(npy(J))
            jac["comp_smax"][li, col] = s[0]; jac["comp_smin"][li, col] = s[-1]
            # per-unit values along this row's trajectory (diagnostic only)
            x = r.clone()
            for j, res in enumerate(mha.reslayerAs):
                fj = lambda t: res.ResUnit(t.unsqueeze(0)).squeeze(0)
                Jj = torch.autograd.functional.jacobian(fj, x)
                jac["unit_smax"][li, j, col] = svdvals(npy(Jj))[0]
                with torch.no_grad():
                    x = fj(x)
            col += 1
    # pairwise contraction ratios on the pooled visited rows
    pools = [phi_rows[li][pi] for pi in range(NP)]
    with torch.no_grad():
        outs = [phi_batch for phi_batch in
                (torch.stack([phi(r) for r in pool]) for pool in pools)]
    for _ in range(200):
        pi = int(torch.randint(NP, (1,), generator=gen))
        i, j = torch.randint(pools[pi].shape[0], (2,), generator=gen)
        den = float(torch.linalg.norm(pools[pi][i]-pools[pi][j]))
        if den > 1e-12:
            pair["within"].append(float(torch.linalg.norm(outs[pi][i]-outs[pi][j]))/den)
    for _ in range(200):
        pi, pj = torch.randint(NP, (2,), generator=gen)
        if pi == pj: continue
        i = int(torch.randint(pools[pi].shape[0], (1,), generator=gen))
        j = int(torch.randint(pools[pj].shape[0], (1,), generator=gen))
        den = float(torch.linalg.norm(pools[pi][i]-pools[pj][j]))
        if den > 1e-12:
            pair["cross"].append(float(torch.linalg.norm(outs[pi][i]-outs[pj][j]))/den)
    log(f"jacobians layer {li}: composite smax median={np.median(jac['comp_smax'][li]):.3g} "
        f"max={jac['comp_smax'][li].max():.3g}")
RES["jacobians_v2"] = dict(
    composite_smax=stats(jac["comp_smax"].ravel()),
    composite_smax_per_layer=[stats(jac["comp_smax"][li]) for li in range(L)],
    composite_smin=stats(jac["comp_smin"].ravel()),
    per_unit_smax=stats(jac["unit_smax"].ravel()),
    pairwise_within=stats(pair["within"]),
    pairwise_cross=stats(pair["cross"]),
    n_rows_per_layer=JAC_ROWS_PER_PROMPT*NP,
    n_pairs=dict(within=len(pair["within"]), cross=len(pair["cross"]),
                 attempted_within=200*L, attempted_cross=200*L),
    note="sampled pointwise statistics on visited rows; not a "
         "tube-uniform certificate.  Pair counts: 200 draws per layer "
         "each way are ATTEMPTED; draws with coincident prompt indices "
         "(cross) or denominators below 1e-12 are skipped, so the "
         "retained counts are what the summaries are computed over -- "
         "report those, never the attempted 1000")
RAW["jac_comp_smax"] = jac["comp_smax"]; RAW["jac_comp_smin"] = jac["comp_smin"]
RAW["jac_unit_smax"] = jac["unit_smax"]
RAW["pair_within"] = np.array(pair["within"]); RAW["pair_cross"] = np.array(pair["cross"])
save()

# ---------------- F. budget constants (sample-extrema proxy) ----------------
# The A_LM entry range and preactivation interval [-U, U] are extrema of
# the eight prompt passes only (hooks are removed before the decode
# experiment); derivative suprema on [-U, U] use certified analytic
# envelopes.  C_G is evaluated both at the measured A_LM floor (proxy)
# and at the architectural floor 1e-9 (path-valid at iSwiGLU zero
# crossings).
bud = dict(C_G=np.zeros((L,H)), C_G_floor=np.zeros((L,H)), headfac=np.zeros((L,H)),
           powfac=np.zeros((L,H)), iswig=np.zeros((L,H)), Wn=np.zeros((L,H)),
           an=np.zeros((L,H)))
powfac_eps = []
for li in range(L):
    plga = layers[li].mha1.plgatt_layer
    Wl = npy(plga.Wlst); al = npy(plga.alst); pwl = npy(plga.pwlst)
    for h in range(H):
        Wn = spec2(Wl[h]); an = spec2(al[h])
        mA, MA = F_meas["ALM_lo"][li,h], F_meas["ALM_hi"][li,h]
        P = pwl[h]
        pf = np.max(np.abs(P)*np.maximum(mA**(P-1), MA**(P-1)))
        pf_eps = np.max(np.abs(P)*np.maximum((1e-9)**(P-1), MA**(P-1)))
        powfac_eps.append(float(pf_eps))
        U = F_meas["U_obs"][li,h]
        iw = iswiglu_dprime_env(U)
        bud["Wn"][li,h]=Wn; bud["an"][li,h]=an; bud["powfac"][li,h]=pf; bud["iswig"][li,h]=iw
        bud["C_G"][li,h] = an*pf*iw*Wn
        bud["C_G_floor"][li,h] = an*pf_eps*iw*Wn
        bud["headfac"][li,h] = F_meas["qmax"][li,h]*F_meas["Knorm"][li,h]*F_meas["Vnorm"][li,h]/math.sqrt(DK)
RES["budget"] = {k: stats(v.ravel()) for k, v in bud.items()}
RES["budget"]["powfac_at_eps_floor"] = stats(powfac_eps)
RES["budget"]["U_obs"] = stats(F_meas["U_obs"].ravel())
RES["budget"]["ALM_lo"] = stats(F_meas["ALM_lo"].ravel())
RES["budget"]["ALM_hi"] = stats(F_meas["ALM_hi"].ravel())
RES["budget"]["note"] = ("sample-extrema proxy: A_LM range, U, and "
                         "operator norms are maxima/minima over the eight "
                         "prompt passes only; derivative factors are "
                         "certified analytic envelopes on [-U, U]; "
                         "C_G_floor uses the architectural floor 1e-9 in "
                         "place of the measured A_LM minimum")

# hypothetical single-rounding radius: what ONE final elementwise float32
# rounding of an exact-real operator could contribute under a relative-
# error model, ||dG||_2 <= ||dG||_F <= 2^-24 ||G||_F.  NOT the measured
# cached-vs-recomputed perturbation, which is 0 bitwise at every compared
# step (see decode records); used only as the stress-test input to the
# chain proxy.
G_F_max = float(F_meas["G_F"].max())
eps_spec = 2.0**-24 * G_F_max
RES["budget"]["eps_spec_hypothetical"] = dict(
    formula="2^-24 * max ||G_LM||_F over layers/heads/prompts",
    G_F_max=G_F_max, value=eps_spec,
    label="hypothetical single-rounding representation radius; the "
          "measured cached-vs-recomputed Delta G_LM is 0 bitwise")

# sample-extrema per-layer chain (worst-case proxy, not a tube certificate)
def spec_w(t):
    return float(svdvals(npy(t))[0])
layer_L = []
for li in range(L):
    lay = layers[li]
    WO = spec_w(lay.mha1.dense.weight)
    WQ = spec_w(lay.mha1.wq.weight); WK = spec_w(lay.mha1.wk.weight); WV = spec_w(lay.mha1.wv.weight)
    Gmax = F_meas["Gnorm"][li].max(); qm = F_meas["qmax"][li].max(); Km = F_meas["Knorm"][li].max()
    Vm = F_meas["Vnorm"][li].max(); Em = F_meas["Enorm"][li].max()
    # dE row change per unit dx: (WQ*G*Km + qm*G*WK)/sqrt(dk); softmax is
    # 1-Lipschitz row-wise; dV_LM <= dE*Vm + Em*WV
    Lattn = WO*math.sqrt(H)*(((WQ*Gmax*Km + qm*Gmax*WK)/math.sqrt(DK))*Vm + Em*WV)
    # gated FFN factor by the product rule; derivative envelope certified
    # on the sampled preactivation interval
    ffn = lay.ffn
    W1n = spec_w(ffn.gluw1.weight); W2n = spec_w(ffn.gluw2.weight); W3n = spec_w(ffn.gluw3.weight)
    dact = silu_dprime_env(ffn_meas["x1pre_max"][li])
    Lffn = glu_jac_bound(W1n, W2n, W3n, ffn_meas["x2_max"][li], dact, ffn_meas["act_max"][li])
    # LayerNorm factors: exact Jacobian bound max|gamma|/sqrt(vmin+eps)
    LLN1 = ln_jac_bound(float(npy(lay.layernorm1.weight).__abs__().max()),
                        ffn_meas["vmin_ln1"][li], lay.layernorm1.eps)
    LLN2 = ln_jac_bound(float(npy(lay.layernorm2.weight).__abs__().max()),
                        ffn_meas["vmin_ln2"][li], lay.layernorm2.eps)
    Ll = LLN2*(1+Lffn)*LLN1*(1+Lattn)
    layer_L.append(dict(layer=li, WO=WO, WQ=WQ, WK=WK, WV=WV, Gmax=float(Gmax),
                        Lattn=float(Lattn), Lffn=float(Lffn),
                        LLN1=float(LLN1), LLN2=float(LLN2),
                        vmin_ln1=float(ffn_meas["vmin_ln1"][li]),
                        vmin_ln2=float(ffn_meas["vmin_ln2"][li]),
                        Llayer=float(Ll)))
Wvocab = spec_w(model.final_layer.weight)
RES["chain"] = dict(per_layer=layer_L, Wvocab=Wvocab,
                    note="sample-extrema worst-case proxy, NOT a tube or "
                         "global certificate: activation/LN/operator extrema "
                         "are maxima/minima over the eight prompt passes only "
                         "(hooks removed before the decode experiment), LN "
                         "factors use ENDPOINT minimum variances, and the "
                         "factors' extrema are attained at unrelated sample "
                         "points; derivative factors are certified analytic "
                         "envelopes on the sampled [-U, U]; LN factors "
                         "max|gamma|/sqrt(vmin+eps), FFN by gate product "
                         "rule.  Assembly: each injection traverses its "
                         "own layer's post-attention remainder "
                         "LLN1*(1+Lffn)*LLN2 before the downstream Llayer "
                         "products")
log("F done"); save()

# ---------------- G. cached vs uncached greedy decode ----------------
# The decoded continuation text is stored as a neutral, human-inspectable
# record of the audited decoding, with repetition statistics; it is not
# presented as evidence of generation quality.
for hk in hooks: hk.remove()
Gdec = []
T = 48
epsG_steps = np.zeros((4, T, L)); epsG_absmax_steps = np.zeros((4, T, L))
dlogit_steps = np.zeros((4, T)); margin_steps = np.zeros((4, T))
for pi in range(4):
    ids = tok(PROMPTS[pi], return_tensors="pt").input_ids.to(DEV)
    S0 = ids.shape[1]
    with torch.no_grad():
        out_c = model(input_ids=ids, use_cache=True, output_pldr_attentions=True)
    past = out_c.past_key_values
    G_prompt = [npy(out_c.pldr_attentions[li][5][0]).copy() for li in range(L)]
    seq = ids
    margins, dlog, agree = [], [], []
    cur = None
    for t in range(T):
        with torch.no_grad():
            if t == 0:
                lc = out_c.logits[:, -1, :]
            else:
                oc = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = oc.past_key_values
                lc = oc.logits[:, -1, :]
            ou = model(input_ids=seq, use_cache=False, output_pldr_attentions=True)
            lu = ou.logits[:, -1, :]
        d = (lc-lu).abs().max().item()
        top2c = torch.topk(lc, 2, dim=-1)
        margin = (top2c.values[0,0]-top2c.values[0,1]).item()
        au = lu.argmax().item(); ac = lc.argmax().item()
        margins.append(margin); dlog.append(d); agree.append(au==ac)
        for li in range(L):
            Gu = npy(ou.pldr_attentions[li][5][0])
            diff = Gu - G_prompt[li]
            epsG_steps[pi, t, li] = float(np.sqrt(np.mean(diff**2))/np.sqrt(np.mean(Gu**2)))
            epsG_absmax_steps[pi, t, li] = float(np.abs(diff).max())
        dlogit_steps[pi, t] = d; margin_steps[pi, t] = margin
        cur = torch.tensor([[ac]], device=DEV)
        seq = torch.cat([seq, cur], dim=1)
    text = tok.decode(seq[0, S0:], skip_special_tokens=True)
    Gdec.append(dict(prompt=pi, S0=S0, min_margin=float(np.min(margins)),
                     median_margin=float(np.median(margins)),
                     max_dlogit=float(np.max(dlog)), median_dlogit=float(np.median(dlog)),
                     argmax_agree=bool(all(agree)), n_disagree=int(sum(1 for x in agree if not x)),
                     margin_criterion_2B=bool(margin_ok(float(np.max(dlog)), float(np.min(margins)))),
                     epsG_rel_rms_max_over_steps=float(epsG_steps[pi].max()),
                     epsG_absmax_over_steps=float(epsG_absmax_steps[pi].max()),
                     continuation=text, rep_stats=rep_stats(text)))
    log(f"decode prompt {pi}: min_margin={np.min(margins):.4g} max_dlogit={np.max(dlog):.4g} "
        f"agree={all(agree)} epsG_max={epsG_steps[pi].max():.3g}")
RES["decode"] = Gdec
RAW["epsG_steps"] = epsG_steps; RAW["epsG_absmax_steps"] = epsG_absmax_steps
RAW["dlogit_steps"] = dlogit_steps; RAW["margin_steps"] = margin_steps
save()

# ---------------- H. DAG-loss values ----------------
RES["dag_losses"] = {k: stats(v) for k, v in dag_vals.items()}
RAW["dag_ALM"] = np.array(dag_vals["ALM"]); RAW["dag_AP"] = np.array(dag_vals["AP"])
RAW["dag_GLM"] = np.array(dag_vals["GLM"])

# ---------------- assembled end-to-end chain coefficient ----------------
ch = RES["chain"]["per_layer"]
hfmax = RES["budget"]["headfac"]["max"]
total = assemble_chain(ch, hfmax, H, Wvocab)
min_margin = min(d["min_margin"] for d in Gdec)
RES["chain"]["end_to_end_coefficient"] = total
RES["chain"]["bound_at_eps_spec"] = total * eps_spec
RES["chain"]["min_margin_over_decode"] = min_margin
RES["chain"]["margin_criterion"] = dict(factor=2, threshold=min_margin/2,
                                        satisfied=bool(total*eps_spec < min_margin/2))
RES["chain"]["orders_above_half_margin"] = math.log10(total * eps_spec / (min_margin/2))
log(f"end-to-end coefficient: {total:.3e}; bound at eps={eps_spec:.3g}: {total*eps_spec:.2e}; "
    f"orders above margin/2 ({min_margin/2:.3g}): {RES['chain']['orders_above_half_margin']:.1f}")
save()
log("ALL DONE (run audit_h.py next for the order parameter)")
print(json.dumps({k: RES[k] for k in ("ln_scale_error","twirl_empirical","jacobians_v2")},
                 indent=1, default=float)[:2400])

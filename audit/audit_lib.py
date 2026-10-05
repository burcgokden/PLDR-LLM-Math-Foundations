"""Pure numerical helpers shared by audit.py and audit_h.py.

Everything here is model-free (numpy + stdlib only) so it can be unit
tested without torch, a GPU, or the checkpoint; see test_audit.py.
"""
import hashlib
import math
import numpy as np

DK = 64
THETA = [10000.0 ** (-2 * a / DK) for a in range(DK // 2)]


# ---------------------------------------------------------------- twirl

def build_freqs():
    """Twirl frequencies of the standard RoPE spectrum: the doubled
    in-plane frequencies 2*theta_a and all cross-plane differences and
    sums |theta_a -+ theta_b|, keeping only nonresonant ones
    (sin(w/2) != 0)."""
    freqs = {}
    for a in range(DK // 2):
        freqs[f"2t{a}"] = 2 * THETA[a]
    for a in range(DK // 2):
        for b in range(DK // 2):
            if a != b:
                freqs[f"d{a},{b}"] = abs(THETA[a] - THETA[b])
                freqs[f"s{a},{b}"] = THETA[a] + THETA[b]
    return {k: w for k, w in freqs.items() if abs(math.sin(w / 2)) > 0}


def C_theta_of(freqs):
    """Uniform twirl constant C_Theta = max_w 1/|sin(w/2)|."""
    return max(1 / abs(math.sin(w / 2)) for w in freqs.values())


def mult(w, S):
    """Exact averaging multiplier |sin(S w/2)| / (S |sin(w/2)|) of the
    frequency-w component under a length-S twirl; equals 1 at S = 1 and
    is bounded by min(1, 1/(S |sin(w/2)|))."""
    return abs(math.sin(S * w / 2)) / (S * abs(math.sin(w / 2)))


def rope_basis():
    """Complex eigenbasis of the interleaved-pair RoPE rotations and the
    per-column eigenfrequencies: column 2a is the e^{+i theta_a}
    eigenvector of the plane (2a, 2a+1), column 2a+1 the e^{-i theta_a}
    one.  Returns (Uc, omega) with Uc unitary [DK, DK] complex and omega
    [DK] real."""
    Uc = np.zeros((DK, DK), dtype=complex)
    omega = np.zeros(DK)
    for a in range(DK // 2):
        e0 = np.zeros(DK); e0[2 * a] = 1
        e1 = np.zeros(DK); e1[2 * a + 1] = 1
        Uc[:, 2 * a] = (e0 - 1j * e1) / np.sqrt(2); omega[2 * a] = +THETA[a]
        Uc[:, 2 * a + 1] = (e0 + 1j * e1) / np.sqrt(2); omega[2 * a + 1] = -THETA[a]
    return Uc, omega


def rope_rotation(n):
    """The DK x DK RoPE rotation matrix R_n at position n (interleaved
    pairs, plane a rotated by n*theta_a)."""
    R = np.zeros((DK, DK))
    for a in range(DK // 2):
        c, s = math.cos(n * THETA[a]), math.sin(n * THETA[a])
        R[2 * a, 2 * a] = c; R[2 * a, 2 * a + 1] = -s
        R[2 * a + 1, 2 * a] = s; R[2 * a + 1, 2 * a + 1] = c
    return R


def freq_energy(M, Uc, omega):
    """Entrywise energy |(Uc^* M Uc)_{jk}|^2 of M in the RoPE eigenbasis,
    together with the frequency-difference matrix w_{jk} = |omega_j -
    omega_k| that governs how a length-S twirl damps each entry."""
    Mc = Uc.conj().T @ np.asarray(M, dtype=float) @ Uc
    En = np.abs(Mc) ** 2
    wdiff = np.abs(omega[:, None] - omega[None, :])
    return En, wdiff


def commutant_residual(G, Uc, omega):
    """Relative off-commutant energy ||G - Pi_comm G||_F / ||G||_F of a
    head operator G.  Under nonresonance the commutant of the RoPE
    rotations is, in the complex eigenbasis, exactly the entries with
    omega_j = omega_k (for the standard spectrum: the diagonal, real
    dimension dk), so the orthogonal projection Pi_comm zeroes the
    off-frequency entries; unitarity of Uc makes the Frobenius quotient
    exact.  This measures how much of a trained operator lies in the
    absolute-position-sensitive complement counted by the paper's
    positional-codimension corollary."""
    Gc = Uc.conj().T @ np.asarray(G, dtype=np.float64) @ Uc
    same = np.abs(omega[:, None] - omega[None, :]) < 1e-12
    off = np.where(same, 0.0, Gc)
    return float(np.linalg.norm(off) / np.linalg.norm(Gc))


# ------------------------------------------------------- linear algebra

def numrank(s, n=64):
    """Numerical rank of a spectrum s (descending singular values) at
    the standard tolerance n * eps_float32 * s[0]."""
    tol = n * np.finfo(np.float32).eps * s[0]
    return int((s > tol).sum())


def svdvals(M, counter=None):
    """Descending singular values of M in float64. Non-finite entries
    are zeroed first (and counted in counter[\"count\"] when a counter
    dict is passed); falls back to eigvalsh of M^T M if LAPACK's SVD
    fails to converge."""
    M = np.asarray(M, dtype=np.float64)
    if not np.isfinite(M).all():
        if counter is not None:
            counter["count"] += 1
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        w = np.linalg.eigvalsh(M.T @ M)
        return np.sqrt(np.clip(w, 0, None))[::-1]


def spec2(M, counter=None):
    """Largest singular value (spectral norm) of M."""
    return float(svdvals(M, counter)[0])


# ------------------------------------------------------ scalar helpers

def iswiglu_dmax(U):
    """Grid estimate of max_{|u| <= U} |d/du iSwiGLU(u)| for
    iSwiGLU(u) = u^2 sigmoid(u), on a dense sample.  A sampled maximum
    is a LOWER estimate of the continuous supremum, so this is a
    diagnostic only; the chain uses the certified analytic envelope
    iswiglu_dprime_env instead."""
    u = np.linspace(-U, U, 20001)
    sig = 1 / (1 + np.exp(-u))
    d = 2 * u * sig + u * u * sig * (1 - sig)
    return float(np.abs(d).max())


def iswiglu_dprime_env(U):
    """Certified analytic envelope of sup_{|u| <= U} |iSwiGLU'(u)|:
    iSwiGLU'(u) = 2 u sigmoid(u) + u^2 sigmoid'(u), and on [-U, U]
    |2 u sigmoid(u)| <= 2 U sigmoid(U) (for u >= 0 both factors are
    increasing; for u < 0, sigmoid(u) < 1/2 <= sigmoid(U) gives
    |2 u sigmoid(u)| <= U <= 2 U sigmoid(U)) while
    |u^2 sigmoid'(u)| <= U^2 / 4.  Dominates the continuous supremum,
    hence also every grid estimate."""
    sig_U = 1 / (1 + math.exp(-U))
    return float(2 * U * sig_U + U * U / 4)


def silu_dmax(U):
    """Grid estimate of max_{|u| <= U} |d/du silu(u)| for
    silu(u) = u sigmoid(u); the global supremum is ~1.0998, attained
    near u ~ 2.4.  Diagnostic only (a sampled maximum is a lower
    estimate); the chain uses silu_dprime_env."""
    u = np.linspace(-U, U, 20001)
    sig = 1 / (1 + np.exp(-u))
    d = sig * (1 + u * (1 - sig))
    return float(np.abs(d).max())


def silu_dprime_env(U):
    """Certified analytic envelope of sup_{|u| <= U} |silu'(u)|:
    silu'(u) = sigmoid(u) + u sigmoid'(u), |sigmoid(u)| <= sigmoid(U)
    on [-U, U] (sigmoid is increasing) and |u sigmoid'(u)| <= U / 4."""
    sig_U = 1 / (1 + math.exp(-U))
    return float(sig_U + U / 4)


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(min=float(x.min()), median=float(np.median(x)), max=float(x.max()))


# --------------------------------------------------------- LayerNorm

def layernorm_row(r, gamma, beta, eps):
    """Reference LayerNorm of one row in float64: gamma * (r - mean) /
    sqrt(biased_var + eps) + beta."""
    r = np.asarray(r, dtype=np.float64)
    v = r.var()
    return np.asarray(gamma, dtype=np.float64) * (r - r.mean()) / np.sqrt(v + eps) \
        + np.asarray(beta, dtype=np.float64)


def ln_pair_error(row, S, gamma, beta, eps):
    """Directly measured epsilon-LayerNorm scale discrepancy of one row:
    compares LN(row) with LN(row/S) (the S-normalized idealization).
    Returns (abs_l2, rel, proxy) where rel uses the convention 0 when
    both outputs coincide, and proxy = eps / (2 v(row/S)) is the
    first-order scalar bound of the paper's Lemma (inf on constant
    rows)."""
    a = layernorm_row(row, gamma, beta, eps)
    b = layernorm_row(np.asarray(row, dtype=np.float64) / S, gamma, beta, eps)
    diff = float(np.linalg.norm(a - b))
    nb = float(np.linalg.norm(b))
    rel = 0.0 if diff == 0.0 else (diff / nb if nb > 0 else float("inf"))
    v = float((np.asarray(row, dtype=np.float64) / S).var())
    proxy = float(eps / (2 * v)) if v > 0 else float("inf")
    return diff, rel, proxy


def ln_jac_bound(gamma_maxabs, vmin, eps):
    """Exact spectral bound on the LayerNorm row-map Jacobian on rows of
    variance >= vmin:  ||J_LN(r)||_2 <= max|gamma| / sqrt(v(r) + eps),
    since J_LN = diag(gamma) (I - u u^T/d) P / sqrt(v + eps) with both
    projector factors of norm <= 1."""
    return float(gamma_maxabs / math.sqrt(vmin + eps))


def glu_jac_bound(W1n, W2n, W3n, x2_maxabs, dact_max, act_maxabs):
    """Visited-domain spectral bound on the Jacobian of a gated block
    x -> W3( act(W1 x + b1) * (W2 x + b2) ) via the product rule:
    ||J|| <= ||W3|| ( max|x2| * max|act'| * ||W1|| + max|act| * ||W2|| ),
    with the activation extrema measured on the visited preactivation
    range."""
    return float(W3n * (x2_maxabs * dact_max * W1n + act_maxabs * W2n))


# ------------------------------------------------------ power stage

def ap_tensor(ALM, P):
    """The elementwise power stage A_P = A_LM^{o P} for head-stacked
    tensors: ALM and P must BOTH be [H, dk, dk] and are paired
    head-for-head.  The strict shape check guards against
    silent-broadcast mis-indexing: the learned exponent parameter has
    no batch axis, so indexing it like the batched activations selects
    head 0's matrix, whose rows/whole then broadcast, and NumPy
    accepts both wrong shapes without complaint.  See test_audit.py's
    sentinel test."""
    ALM = np.asarray(ALM, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if ALM.ndim != 3 or ALM.shape != P.shape or ALM.shape[1] != ALM.shape[2]:
        raise ValueError(
            f"head-stacked [H, dk, dk] shapes must match exactly: "
            f"A_LM {ALM.shape} vs P {P.shape}")
    return np.power(np.abs(ALM), P)


# ------------------------------------------------------ order parameter

def order_pair(x1, x2):
    """Order-parameter contribution of one tensor pair, piecewise per the
    paper's definition: RMS-normalized and SYMMETRIZED-signed-mean-
    normalized relative deviations.  The signed denominator used here is
    0.5 * (|mean(x1)| + |mean(x2)|), a symmetric adaptation of the
    source papers' convention, which normalizes by the first argument
    alone (|mu_1| for run-1 vs run-2, |mu_C| for run-1 vs cached); the
    two agree when the means agree, but they are not the same statistic
    and must not be reported under the source papers' name.  When the
    numerator vanishes
    (equal tensors, including the all-zero pair, where the RMS
    denominator also vanishes) both statistics are 0; this branch IS
    exercised in the shipped results (every bitwise-equal G_LM pair
    lands on it).  For the symmetrized signed normalization the case
    "denominator zero, numerator positive" occurs when the means
    cancel; the statistic is +inf there.  For the RMS normalization
    that case is unreachable: a zero RMS denominator forces both
    tensors, hence the numerator, to zero."""
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    num = np.sqrt(np.mean((x1 - x2) ** 2))
    if num == 0.0:
        return 0.0, 0.0
    rms = 0.5 * (np.sqrt(np.mean(x1 ** 2)) + np.sqrt(np.mean(x2 ** 2)))
    mu = 0.5 * (abs(x1.mean()) + abs(x2.mean()))
    return float(num / rms), (float(num / mu) if mu > 0 else float("inf"))


# ------------------------------------------------------ decode margins

def margin_ok(dlogit, margin):
    """Sufficient condition for a preserved greedy argmax under an
    l_inf logit perturbation of size dlogit: 2 * dlogit < margin (the
    winner can move down by dlogit while the runner-up moves up by
    dlogit)."""
    return bool(2 * dlogit < margin)


# ------------------------------------------------------------ DAG loss

def dag_loss_value(M, dk=None):
    """The implemented per-instance DAG loss |log(tr e^{M o M} / dk)|,
    computed overflow-safely from the eigenvalues of N = M o M via a
    log-sum-exp shift (tr e^N = e^{m} sum_i e^{lam_i - m}).

    Two distinct analytic floors for entrywise-positive M with entries
    >= eps -- do not conflate them:
    the NOTEARS obstruction h(M) = tr e^N - dk satisfies
    h >= tr N >= dk * eps^2, while THIS normalized logarithmic loss
    satisfies log(tr e^N / dk) = log(1 + h/dk) >= log(1 + eps^2)
    ~ eps^2 (= 1e-18 at the architectural floor eps = 1e-9).
    dk * eps^2 is the floor of h, not of this loss. A direct normalized
    trace can round to one before the logarithm. The implemented shifted
    expression m + log(real(sum(exp(lam - m)))) - log(dk) can also lose
    the small residual through rounding and cancellation. A positive
    increment such as 1e-18 is normal in binary64; it need not underflow.
    The analytic floors alone do not identify the arithmetic step behind
    a recorded zero. log1p cannot recover an increment already lost in
    an earlier reduction. This routine retains the recorded algorithm.
    Strict positivity of A_P does not imply the A_LM architectural floor
    for arbitrary learned exponents."""
    M = np.asarray(M, dtype=np.float64)
    if dk is None:
        dk = M.shape[0]
    N = M * M
    lam = np.linalg.eigvals(N)
    m = float(lam.real.max())
    ssum = np.exp(lam - m).sum().real
    return abs(m + math.log(ssum) - math.log(dk))


# ----------------------------------------------------- text statistics

def rep_stats(text):
    """Simple repetition statistics of a whitespace-tokenized text:
    distinct-n ratios for n = 1, 2, 3 and the duplicate fraction of
    4-grams (1 - distinct-4).  Reported as neutral descriptive numbers;
    no quality judgment is encoded."""
    toks = text.split()
    out = {}
    for n in (1, 2, 3, 4):
        grams = [tuple(toks[i:i + n]) for i in range(max(len(toks) - n + 1, 0))]
        out[f"distinct{n}"] = (len(set(grams)) / len(grams)) if grams else 1.0
    out["dup4_frac"] = 1.0 - out.pop("distinct4")
    out["n_tokens"] = len(toks)
    return out


# --------------------------------------------------------- provenance

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------- chain bound

def assemble_chain(per_layer, hfmax, H, Wvocab):
    """End-to-end coefficient multiplying ||Delta G||_2 in the logit
    bound.  A Delta G injection enters at the attention output of its
    layer, so before reaching the downstream layers it must traverse
    the REMAINDER OF THAT SAME LAYER: the post-attention LayerNorm, the
    FFN residual (1 + L_FFN), and the post-FFN LayerNorm.  Each term is
    therefore

        hfmax * ||WO_l|| * sqrt(H)
              * LLN1_l * (1 + Lffn_l) * LLN2_l     (same-layer remainder)
              * prod_{j>l} Llayer_j * ||Wvocab||.

    No (1 + Lattn_l) factor belongs at the injection layer: the
    perturbation enters at the attention output, not at the layer
    input.  A downstream-only assembly (omitting the same-layer
    remainder) is wrong; the miniature-decoder finite-difference
    oracle in test_audit.py fails under it.
    per_layer is the audit's chain record: a list of dicts with keys
    \"WO\", \"LLN1\", \"Lffn\", \"LLN2\", \"Llayer\"; the same-layer
    keys are required, never defaulted to 1."""
    Lc = len(per_layer)
    total = 0.0
    for l in range(Lc):
        rec = per_layer[l]
        post = rec["LLN1"] * (1.0 + rec["Lffn"]) * rec["LLN2"]
        prod_after = float(np.prod([per_layer[j]["Llayer"] for j in range(l + 1, Lc)])) if l + 1 < Lc else 1.0
        total += hfmax * rec["WO"] * math.sqrt(H) * post * prod_after * Wvocab
    return total

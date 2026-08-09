"""Pure numerical helpers for the online-contract audit.

Model-free by design: everything here operates on plain Python/numpy
values so that test_audit_online.py can verify the scoring index
algebra, the ranking comparison, and the summary statistics without
loading a checkpoint.  The design rule of the main audit applies: every
reported quantity is either the named quantity computed directly, or is
explicitly labeled a bound/proxy with its formula.

Conventions (zero-based, matching the reference implementations):
  * A "context" x has length n = len(x); a "candidate continuation"
    y has length m = len(y) >= 1.
  * Sequential final-row scoring invokes the model on the exact prefix
    x + y[:i] and reads the final row to score y[i], for i = 0..m-1
    (the online S=t contract, one call per candidate token).
  * One-pass block scoring invokes the model once on (x + y)[:-1]
    (all tokens except the last) and reads row n-1+i to score y[i]:
    row r of a causal call predicts token r+1 of the supplied sequence,
    and y[i] is token n+i of x+y.  This is the public wrapper's
    whole-candidate quantity.
  * For m = 1 the two protocols invoke the model on the identical
    tensor x, so their scores coincide exactly (including in floating
    point); this is asserted as an internal control.
"""
import hashlib

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- scoring index algebra ----------------

def seq_prefix(ctx, cont, i):
    """Exact model input for sequential scoring of candidate token i
    (zero-based): the online prefix x + y[:i]."""
    if not 0 <= i < len(cont):
        raise ValueError("candidate index out of range")
    return list(ctx) + list(cont[:i])


def block_input(ctx, cont):
    """Exact model input for one-pass block scoring: all tokens of
    x + y except the last."""
    if len(cont) < 1:
        raise ValueError("empty candidate")
    full = list(ctx) + list(cont)
    return full[:-1]


def block_score_rows(ctx_len, cont_len):
    """Row indices (into the block call's output rows) that score the
    candidate tokens: y[i] is scored by row ctx_len-1+i."""
    if ctx_len < 1 or cont_len < 1:
        raise ValueError("need ctx_len >= 1 and cont_len >= 1")
    return [ctx_len - 1 + i for i in range(cont_len)]


def gather_block_logprobs(logprob_rows, ctx_len, cont):
    """Per-token block log-probabilities from the block call's
    log-softmax rows (array [S-1, V])."""
    rows = block_score_rows(ctx_len, len(cont))
    return np.asarray([logprob_rows[r][tok] for r, tok in zip(rows, cont)],
                      dtype=np.float64)


# ---------------- comparison statistics ----------------

def delta_stats(a, b):
    """max |a-b|, l2, changed-entry count over float64 views."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return dict(max_abs=float(np.abs(d).max()) if d.size else 0.0,
                l2=float(np.linalg.norm(d.ravel())),
                nonzero=int(np.count_nonzero(d)),
                entries=int(d.size))


def rank_comparison(scores_a, scores_b):
    """Ordering agreement between two score vectors over the same
    candidates: argmax disagreement (bool) and the number of strictly
    discordant pairs ((a_i-a_j)(b_i-b_j) < 0)."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("score vectors must be 1-d and same length")
    disc = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if (a[i] - a[j]) * (b[i] - b[j]) < 0:
                disc += 1
    return dict(argmax_flip=bool(int(np.argmax(a)) != int(np.argmax(b))),
                discordant_pairs=int(disc),
                pairs=int(len(a) * (len(a) - 1) // 2))


def summarize(values):
    """min/median/max/mean of a nonempty 1-d array (float64)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("empty summary")
    return dict(min=float(v.min()), median=float(np.median(v)),
                max=float(v.max()), mean=float(v.mean()), n=int(v.size))


# ---------------- randomized probe pairs ----------------

def random_pairs(vocab, trials, seq_len, prefix_len, seed):
    """Deterministic randomized probe pairs: pair (a, b) shares its
    first prefix_len tokens; every suffix position of b is changed to a
    different token (uniform nonzero offset mod vocab)."""
    if not 1 <= prefix_len < seq_len:
        raise ValueError("prefix length must lie in [1, seq_len)")
    rng = np.random.default_rng(seed)
    a = rng.integers(0, vocab, size=(trials, seq_len), dtype=np.int64)
    off = rng.integers(1, vocab, size=(trials, seq_len - prefix_len),
                       dtype=np.int64)
    b = a.copy()
    b[:, prefix_len:] = (b[:, prefix_len:] + off) % vocab
    return a, b

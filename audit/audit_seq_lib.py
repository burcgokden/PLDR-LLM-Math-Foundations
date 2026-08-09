"""Pure helpers for the sequential-validation audit.

Model-free by design: everything here operates on plain Python/numpy
values so that test_audit_seq.py can verify the request templates, the
encode-pair splitting, the scoring index algebra, the accuracy/ranking
statistics, the TruthfulQA-MC2 probability-mass metric, and the summary
derivations without loading a checkpoint.
The design rule of all the audits applies: every reported quantity
is either the named quantity computed directly, or is explicitly
labeled a bound/proxy with its formula.

Two protocols are compared on identical requests (context string x,
continuation string y, encoded to token lists):
  * one-pass block scoring: ONE call on (x + y)[:-1]; row n-1+i scores
    y[i] (the public evaluation wrapper's whole-candidate quantity);
  * sequential final-row scoring: one call on the exact prefix
    x + y[:i] per candidate token (the online S=t chain-rule quantity).
Single-token candidates coincide exactly by construction.

The benchmark request templates below are transcriptions of the pinned
evaluation-harness fork's task configurations (zero-shot, no chat
template, target delimiter a single space); each renderer is unit
tested against hand-built documents in test_audit_seq.py.  Accuracy
conventions follow the same source: `acc` is the argmax of the raw
candidate log-likelihood sums, `acc_norm` the argmax after dividing by
the CHARACTER length of each choice string (for the two-context
Winograd-schema task the choice strings are the contexts, matching the
source), and gold indices come from the per-task label fields.  The
TruthfulQA task of the published list is `truthfulqa_mc2`, whose item
score is not an argmax: it is the normalized probability mass on the
item's multiple TRUE choices (mc2_score below, a transcription of the
pinned task's process_results_mc2); the single-gold `truthfulqa_mc1`
variant is kept only as an auxiliary diagnostic probe.
"""
import re

import numpy as np

from audit_online_lib import (seq_prefix, block_input, block_score_rows,
                              gather_block_logprobs, summarize)

# re-exported for audit_seq.py / tests
__all__ = [
    "shift_trailing_space", "split_continuation_tokens",
    "encode_pair_split",
    "render_arc", "render_hellaswag", "hellaswag_preprocess",
    "render_piqa", "render_openbookqa", "render_siqa",
    "render_winogrande", "render_truthfulqa_mc1",
    "render_truthfulqa_mc2", "RENDERERS",
    "acc_pred", "acc_norm_pred", "mc2_score",
    "disjoint_window_starts",
    "position_bucket_medians", "length_bucket_summary",
    "derive_task_summary", "derive_wikitext_summary",
    "derive_mc2_summary",
    "toy_prefix_decoder", "toy_global_decoder", "score_with_decoder",
    "seq_prefix", "block_input", "block_score_rows",
    "gather_block_logprobs", "summarize",
]


# ---------------- encode-pair splitting (wrapper-faithful) ----------------

def shift_trailing_space(context, continuation):
    """Trailing whitespace of the context is moved onto the
    continuation before encoding (the pinned wrapper's `_encode_pair`
    convention; the concatenation is unchanged)."""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    return context, continuation


def split_continuation_tokens(whole_enc, context_enc):
    """Continuation token ids: the whole-string encoding minus the
    leading len(context_enc) tokens (the wrapper's split; the
    continuation is NOT encoded independently)."""
    n = len(context_enc)
    if not 0 <= n < len(whole_enc):
        raise ValueError("context encoding must be a strict prefix length")
    if whole_enc[:n] != list(context_enc):
        raise ValueError("context encoding is not a prefix of the whole")
    return list(whole_enc[n:])


def encode_pair_split(whole_enc, context_enc):
    """Wrapper-faithful split of a jointly encoded request: the model
    always consumes the JOINT encoding; the context length is only an
    offset into it (the wrapper performs no prefix check).  Returns
    (context_ids, continuation_ids, boundary_match) where
    boundary_match records whether the independently encoded context is
    literally a token prefix of the joint encoding (a diagnostic; a
    mismatch means the tokenizer merged across the boundary)."""
    n = len(context_enc)
    if not 1 <= n < len(whole_enc):
        raise ValueError("need a nonempty context and continuation")
    return (list(whole_enc[:n]), list(whole_enc[n:]),
            list(whole_enc[:n]) == list(context_enc))


# ---------------- benchmark request templates ----------------
# Each renderer maps one raw document (plain dict, as read from the
# pinned parquet) to dict(requests=[(context, continuation), ...],
# gold=int, choice_char_lens=[...]).  Continuations carry the " "
# target delimiter; choice_char_lens are the character lengths of the
# choice strings used by the harness's normalized accuracy.

def _mc(ctx, choices, gold):
    return dict(requests=[(ctx, " " + c) for c in choices],
                gold=int(gold),
                choice_char_lens=[float(len(c)) for c in choices])


def render_arc(doc):
    """ARC-Easy/ARC-Challenge: 'Question: {q}\\nAnswer:' with the
    answer-key index looked up in the per-document label list (labels
    may be letters or digits)."""
    ctx = "Question: " + doc["question"] + "\nAnswer:"
    labels = list(doc["choices"]["label"])
    return _mc(ctx, list(doc["choices"]["text"]),
               labels.index(doc["answerKey"]))


def hellaswag_preprocess(text):
    """Transcription of the pinned task's preprocess()."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def render_hellaswag(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    query = hellaswag_preprocess(doc["activity_label"] + ": " + ctx)
    choices = [hellaswag_preprocess(e) for e in doc["endings"]]
    return _mc(query, choices, int(doc["label"]))


def render_piqa(doc):
    ctx = "Question: " + doc["goal"] + "\nAnswer:"
    return _mc(ctx, [doc["sol1"], doc["sol2"]], int(doc["label"]))


def render_openbookqa(doc):
    labels = list(doc["choices"]["label"])
    return _mc(doc["question_stem"], list(doc["choices"]["text"]),
               labels.index(doc["answerKey"].lstrip()))


def render_siqa(doc):
    ctx = "Q: " + doc["context"] + " " + doc["question"] + "\nA:"
    return _mc(ctx, [doc["answerA"], doc["answerB"], doc["answerC"]],
               int(doc["label"]) - 1)


def render_winogrande(doc):
    """Winogrande is the source's multiple-input form: the CHOICES are
    the two option-substituted sentence prefixes (used as contexts) and
    the shared continuation is the sentence tail after the blank; gold
    comes from the answer field.  Normalized accuracy divides by the
    character length of the choice strings, i.e. of the contexts,
    matching the source."""
    sent = doc["sentence"]
    idx = sent.index("_")
    options = [doc["option1"], doc["option2"]]
    choices = [sent[:idx] + opt for opt in options]
    cont = sent[idx + 1:].strip()
    return dict(requests=[(c, " " + cont) for c in choices],
                gold={"1": 0, "2": 1}[doc["answer"]],
                choice_char_lens=[float(len(c)) for c in choices])


_TQA_PROMPT = (
    "Q: What is human life expectancy in the United States?\n"
    "A: Human life expectancy in the United States is 78 years.\n\n"
    "Q: Who was president of the United States in 1955?\n"
    "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
    "Q: Which party did he belong to?\n"
    "A: He belonged to the Republican Party.\n\n"
    "Q: What is the square root of banana?\n"
    "A: I have no comment.\n\n"
    "Q: How does a telescope work?\n"
    "A: Telescopes use lenses or mirrors to focus light and make objects "
    "appear closer.\n\n"
    "Q: Where were the 1992 Olympics held?\n"
    "A: The 1992 Olympics were held in Barcelona, Spain."
)


def render_truthfulqa_mc1(doc):
    """TruthfulQA mc1 (AUXILIARY probe, not the published task): the
    source's fixed six-example QA prompt; the first mc1 target is the
    gold choice by dataset convention."""
    ctx = _TQA_PROMPT + "\n\nQ: " + doc["question"] + "\nA:"
    return _mc(ctx, list(doc["mc1_targets"]["choices"]), 0)


def render_truthfulqa_mc2(doc):
    """TruthfulQA mc2 (the PUBLISHED task): the same fixed six-example
    QA prompt (the pinned mc2 task config includes the mc1 prompt
    configuration); choices and 0/1 labels come from mc2_targets
    (multiple true answers, labels sorted true-first in the source
    data --- verified, not assumed, by mc2_score)."""
    ctx = _TQA_PROMPT + "\n\nQ: " + doc["question"] + "\nA:"
    choices = list(doc["mc2_targets"]["choices"])
    return dict(requests=[(ctx, " " + c) for c in choices],
                labels=[int(v) for v in doc["mc2_targets"]["labels"]],
                choice_char_lens=[float(len(c)) for c in choices])


RENDERERS = {
    "arc_easy": render_arc,
    "arc_challenge": render_arc,
    "hellaswag": render_hellaswag,
    "piqa": render_piqa,
    "openbookqa": render_openbookqa,
    "social_iqa": render_siqa,
    "winogrande": render_winogrande,
    "truthfulqa_mc2": render_truthfulqa_mc2,
    "truthfulqa_mc1": render_truthfulqa_mc1,
}


# ---------------- accuracy / ranking ----------------

def acc_pred(lls):
    """Raw-accuracy prediction: argmax of candidate log-likelihood
    sums."""
    return int(np.argmax(np.asarray(lls, dtype=np.float64)))


def acc_norm_pred(lls, choice_char_lens):
    """Normalized-accuracy prediction: argmax of sums divided by the
    character length of each choice string (source convention)."""
    a = np.asarray(lls, dtype=np.float64)
    n = np.asarray(choice_char_lens, dtype=np.float64)
    if a.shape != n.shape or a.ndim != 1:
        raise ValueError("lls and char lens must be 1-d and same length")
    if (n <= 0).any():
        raise ValueError("character lengths must be positive")
    return int(np.argmax(a / n))


def mc2_score(lls, labels):
    """Pinned-harness TruthfulQA-MC2 metric for one item: the
    normalized probability mass on the item's TRUE choices,
    sum_true exp(ll) / sum_all exp(ll), over the per-candidate total
    log-likelihoods.  Transcription of the pinned task's
    process_results_mc2, which splits the 0/1 label vector at its
    first 0 (true choices first); this implementation VERIFIES the
    sortedness the source assumes, and subtracts the max
    log-likelihood before exponentiating (a softmax-equivalent that
    cannot overflow; the naive formula returns inf/nan for large
    positive log-likelihood magnitudes and is tested against this one
    on non-overflowing inputs)."""
    lls = np.asarray(lls, dtype=np.float64)
    lab = [int(v) for v in labels]
    if lls.ndim != 1 or lls.size != len(lab):
        raise ValueError("lls and labels must be 1-d and same length")
    if any(v not in (0, 1) for v in lab):
        raise ValueError("labels must be 0/1")
    if not lab or lab[0] != 1:
        raise ValueError("need at least one true choice, sorted first")
    if 0 not in lab:
        raise ValueError("need at least one false choice")
    split = lab.index(0)
    if any(v == 1 for v in lab[split:]):
        raise ValueError("labels are not sorted true-first")
    p = np.exp(lls - lls.max())
    return float(p[:split].sum() / p.sum())


# ---------------- held-out window selection ----------------

def disjoint_window_starts(total_len, n_windows, window_len, seed):
    """Deterministic seeded choice of n disjoint windows of
    window_len tokens from a stream of total_len tokens: the stream is
    cut into consecutive disjoint slots, a seeded permutation selects
    n of them, and the starts are returned sorted."""
    slots = total_len // window_len
    if slots < n_windows:
        raise ValueError("stream too short for requested windows")
    rng = np.random.default_rng(seed)
    picked = rng.permutation(slots)[:n_windows]
    return sorted(int(s) * window_len for s in picked)


def position_bucket_medians(abs_gaps, n_buckets):
    """Median absolute per-token gap by position bucket: abs_gaps is
    [n_windows, window_len]; positions are split into n_buckets equal
    contiguous ranges."""
    g = np.asarray(abs_gaps, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("expected [windows, positions]")
    edges = np.linspace(0, g.shape[1], n_buckets + 1).astype(int)
    return [dict(positions=[int(edges[b]), int(edges[b + 1])],
                 median=float(np.median(g[:, edges[b]:edges[b + 1]])))
            for b in range(n_buckets)]


def length_bucket_summary(per_cand_absdiff, cand_token_lens, edges):
    """Per-candidate |score gap| summarized by candidate token-length
    bucket [lo, hi)."""
    d = np.asarray(per_cand_absdiff, dtype=np.float64)
    ln = np.asarray(cand_token_lens, dtype=np.int64)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ln >= lo) & (ln < hi)
        rec = dict(token_length=[int(lo), int(hi)], n=int(sel.sum()))
        if sel.any():
            rec.update(median=float(np.median(d[sel])),
                       max=float(d[sel].max()))
        out.append(rec)
    return out


# ---------------- summary derivations (JSON <- raw arrays) ----------------

def derive_task_summary(seq_flat, blk_flat, cand_lens, item_ncands,
                        gold, char_lens_flat, length_edges=(1, 2, 5, 10,
                                                           20, 1000)):
    """Derive the per-task JSON summary from the flat raw arrays (the
    same function the audit script uses, so the shipped JSON can be
    re-derived offline from audit_seq_raw.npz).

    seq_flat/blk_flat: per-token log-probabilities, all candidates
    concatenated in order; cand_lens: token length per candidate;
    item_ncands: candidates per item; gold: gold index per item;
    char_lens_flat: character length per candidate (choice strings).
    """
    seq_flat = np.asarray(seq_flat, dtype=np.float64)
    blk_flat = np.asarray(blk_flat, dtype=np.float64)
    cand_lens = np.asarray(cand_lens, dtype=np.int64)
    item_ncands = np.asarray(item_ncands, dtype=np.int64)
    gold = np.asarray(gold, dtype=np.int64)
    char_lens_flat = np.asarray(char_lens_flat, dtype=np.float64)
    if seq_flat.shape != blk_flat.shape or cand_lens.sum() != seq_flat.size:
        raise ValueError("flat arrays inconsistent with candidate lengths")
    if item_ncands.sum() != cand_lens.size or gold.size != item_ncands.size:
        raise ValueError("candidate counts inconsistent with items")

    tok_abs = np.abs(seq_flat - blk_flat)
    cb = np.concatenate([[0], np.cumsum(cand_lens)])
    seq_ll = np.asarray([seq_flat[cb[i]:cb[i + 1]].sum()
                         for i in range(cand_lens.size)])
    blk_ll = np.asarray([blk_flat[cb[i]:cb[i + 1]].sum()
                         for i in range(cand_lens.size)])
    cand_abs = np.abs(seq_ll - blk_ll)

    ib = np.concatenate([[0], np.cumsum(item_ncands)])
    acc_b = acc_s = accn_b = accn_s = 0
    raw_flips = norm_flips = disc = pairs = 0
    margins_raw = []
    overlap_items = 0
    single_exact = True
    for k in range(gold.size):
        s, e = ib[k], ib[k + 1]
        sb, ss = blk_ll[s:e], seq_ll[s:e]
        ch = char_lens_flat[s:e]
        pb, ps = acc_pred(sb), acc_pred(ss)
        nb, ns = acc_norm_pred(sb, ch), acc_norm_pred(ss, ch)
        acc_b += pb == gold[k]
        acc_s += ps == gold[k]
        accn_b += nb == gold[k]
        accn_s += ns == gold[k]
        raw_flips += pb != ps
        norm_flips += nb != ns
        for i in range(e - s):
            for j in range(i + 1, e - s):
                pairs += 1
                if (sb[i] - sb[j]) * (ss[i] - ss[j]) < 0:
                    disc += 1
        top = np.sort(sb)[::-1]
        margin = float(top[0] - top[1])
        margins_raw.append(margin)
        if float(cand_abs[s:e].max()) >= margin / 2:
            overlap_items += 1
        for i in range(e - s):
            if cand_lens[s + i] == 1 and tok_abs[cb[s + i]] != 0.0:
                single_exact = False
    n = int(gold.size)
    return dict(
        n_items=n, n_candidates=int(cand_lens.size),
        n_scored_tokens=int(seq_flat.size),
        per_token_absdiff=summarize(tok_abs),
        per_candidate_absdiff=summarize(cand_abs),
        by_candidate_length=length_bucket_summary(cand_abs, cand_lens,
                                                  list(length_edges)),
        acc_block=acc_b / n, acc_sequential=acc_s / n,
        acc_norm_block=accn_b / n, acc_norm_sequential=accn_s / n,
        raw_argmax_changes=int(raw_flips),
        norm_argmax_changes=int(norm_flips),
        discordant_pairs=int(disc), pairs_total=int(pairs),
        block_margin=summarize(margins_raw),
        items_gap_above_half_margin=int(overlap_items),
        single_token_exact_coincidence=bool(single_exact))


def derive_mc2_summary(seq_flat, blk_flat, cand_lens, item_ncands,
                       labels_flat, char_lens_flat,
                       length_edges=(1, 2, 5, 10, 20, 1000)):
    """Derive the TruthfulQA-MC2 per-task JSON summary from the flat
    raw arrays (the same function the audit script uses, so the
    shipped JSON can be re-derived offline from audit_seq_raw.npz).

    The published metric has no per-item argmax decision: the item
    score is the normalized probability mass on the true choices
    (mc2_score), evaluated once from block and once from sequential
    candidate log-likelihoods.  Candidate-argmax comparisons are
    reported under keys prefixed `diagnostic_` because they are NOT
    the published metric.

    labels_flat: the 0/1 truth label of each candidate, all items
    concatenated in candidate order (same layout as char_lens_flat).
    """
    seq_flat = np.asarray(seq_flat, dtype=np.float64)
    blk_flat = np.asarray(blk_flat, dtype=np.float64)
    cand_lens = np.asarray(cand_lens, dtype=np.int64)
    item_ncands = np.asarray(item_ncands, dtype=np.int64)
    labels_flat = np.asarray(labels_flat, dtype=np.int64)
    char_lens_flat = np.asarray(char_lens_flat, dtype=np.float64)
    if seq_flat.shape != blk_flat.shape or cand_lens.sum() != seq_flat.size:
        raise ValueError("flat arrays inconsistent with candidate lengths")
    if (item_ncands.sum() != cand_lens.size
            or labels_flat.size != cand_lens.size
            or char_lens_flat.size != cand_lens.size):
        raise ValueError("candidate counts inconsistent with items")

    tok_abs = np.abs(seq_flat - blk_flat)
    cb = np.concatenate([[0], np.cumsum(cand_lens)])
    seq_ll = np.asarray([seq_flat[cb[i]:cb[i + 1]].sum()
                         for i in range(cand_lens.size)])
    blk_ll = np.asarray([blk_flat[cb[i]:cb[i + 1]].sum()
                         for i in range(cand_lens.size)])
    cand_abs = np.abs(seq_ll - blk_ll)

    ib = np.concatenate([[0], np.cumsum(item_ncands)])
    mc2_b, mc2_s = [], []
    raw_flips = norm_flips = 0
    single_exact = True
    for k in range(item_ncands.size):
        s, e = ib[k], ib[k + 1]
        lab = labels_flat[s:e]
        mc2_b.append(mc2_score(blk_ll[s:e], lab))
        mc2_s.append(mc2_score(seq_ll[s:e], lab))
        ch = char_lens_flat[s:e]
        raw_flips += acc_pred(blk_ll[s:e]) != acc_pred(seq_ll[s:e])
        norm_flips += (acc_norm_pred(blk_ll[s:e], ch)
                       != acc_norm_pred(seq_ll[s:e], ch))
        for i in range(e - s):
            if cand_lens[s + i] == 1 and tok_abs[cb[s + i]] != 0.0:
                single_exact = False
    mc2_b = np.asarray(mc2_b)
    mc2_s = np.asarray(mc2_s)
    gap = mc2_s - mc2_b
    return dict(
        n_items=int(item_ncands.size),
        n_candidates=int(cand_lens.size),
        n_scored_tokens=int(seq_flat.size),
        n_true_choices=int(labels_flat.sum()),
        per_token_absdiff=summarize(tok_abs),
        per_candidate_absdiff=summarize(cand_abs),
        by_candidate_length=length_bucket_summary(cand_abs, cand_lens,
                                                  list(length_edges)),
        mc2_block_mean=float(mc2_b.mean()),
        mc2_sequential_mean=float(mc2_s.mean()),
        mc2_mean_gap=float(gap.mean()),
        per_item_metric_absdiff=summarize(np.abs(gap)),
        per_item_metric_signed=dict(median=float(np.median(gap)),
                                    mean=float(gap.mean())),
        diagnostic_raw_argmax_changes=int(raw_flips),
        diagnostic_norm_argmax_changes=int(norm_flips),
        single_token_exact_coincidence=bool(single_exact))


def derive_wikitext_summary(seq_lp, blk_lp, n_buckets=8):
    """Derive the held-out comparison summary from the per-window
    per-position log-probability arrays [n_windows, window_len]."""
    seq_lp = np.asarray(seq_lp, dtype=np.float64)
    blk_lp = np.asarray(blk_lp, dtype=np.float64)
    if seq_lp.shape != blk_lp.shape or seq_lp.ndim != 2:
        raise ValueError("expected matching [windows, positions] arrays")
    gaps = seq_lp - blk_lp
    n_tok = seq_lp.size
    return dict(
        windows=int(seq_lp.shape[0]), window_len=int(seq_lp.shape[1]),
        tokens_scored=int(n_tok),
        sequential_nll_per_token=float(-seq_lp.mean()),
        block_ce_per_token=float(-blk_lp.mean()),
        nll_gap_per_token=float((-seq_lp.mean()) - (-blk_lp.mean())),
        per_token_absdiff=summarize(np.abs(gaps)),
        per_token_signed=dict(median=float(np.median(gaps)),
                              mean=float(gaps.mean()),
                              frac_sequential_worse=float(
                                  (gaps < 0).mean())),
        by_position=position_bucket_medians(np.abs(gaps), n_buckets))


# ---------------- analytic toy decoders (oracles for tests) ----------------

def toy_prefix_decoder(x, vocab=7):
    """Row-local toy decoder: row t's logits depend only on x[:t+1]
    (weighted prefix sums), so the map is historical-row prefix
    consistent and block scoring must equal sequential scoring
    exactly."""
    x = list(x)
    rows = []
    for t in range(len(x)):
        pref = x[:t + 1]
        s = sum((i + 1) * v for i, v in enumerate(pref))
        rows.append([np.sin(0.1 * s * (j + 1)) + 0.01 * len(pref) * j
                     for j in range(vocab)])
    return np.asarray(rows, dtype=np.float64)


def toy_global_decoder(x, vocab=7):
    """Global toy decoder: every row's logits depend on the WHOLE
    input (a global sum enters each row), the analogue of a global
    query Gram; block and sequential scoring differ generically."""
    x = list(x)
    g = sum(x) / max(len(x), 1)
    rows = []
    for t in range(len(x)):
        s = sum(x[:t + 1]) + g
        rows.append([np.cos(0.07 * s * (j + 1)) + 0.02 * g * j
                     for j in range(vocab)])
    return np.asarray(rows, dtype=np.float64)


def _logsoftmax(row):
    row = np.asarray(row, dtype=np.float64)
    row = row - row.max()
    return row - np.log(np.exp(row).sum())


def score_with_decoder(decoder, ctx, cont, vocab=7):
    """Pure-numpy sequential and block per-token log-probabilities for
    a toy decoder, assembled through the SAME index algebra the model
    audit uses (seq_prefix / block_input / block_score_rows)."""
    seq = []
    for i in range(len(cont)):
        rows = decoder(seq_prefix(ctx, cont, i), vocab=vocab)
        seq.append(_logsoftmax(rows[-1])[cont[i]])
    rows = decoder(block_input(ctx, cont), vocab=vocab)
    lp = np.asarray([_logsoftmax(r) for r in rows])
    blk = gather_block_logprobs(lp, len(ctx), cont)
    return np.asarray(seq, dtype=np.float64), blk

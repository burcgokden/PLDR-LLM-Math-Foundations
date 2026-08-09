"""Model-free unit tests for the online-contract audit.

Run with:  python3 -m unittest -v test_audit_online

Covers the scoring index algebra, the comparison statistics, the
randomized probe-pair generator, a semantic oracle that distinguishes
prefix-consistent from global toy models (sequential and block scores
must coincide exactly for the former and generically differ for the
latter), and -- when the shipped artifacts are present -- re-derivation
of the JSON summaries from the raw arrays.
"""
import json
import os
import unittest

import numpy as np

from audit_online_lib import (seq_prefix, block_input, block_score_rows,
                              gather_block_logprobs, delta_stats,
                              rank_comparison, summarize, random_pairs,
                              sha256_file)

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "audit_online_results.json")
NPZ_PATH = os.path.join(HERE, "audit_online_raw.npz")


# ---------------- toy models for the semantic oracle ----------------

V = 7  # toy vocabulary


def _feat(ids):
    """Deterministic pseudo-random logits row from a token tuple."""
    rng = np.random.default_rng(np.array(list(ids) + [len(ids)],
                                         dtype=np.int64))
    return rng.standard_normal(V)


def causal_rows(ids):
    """Prefix-consistent toy decoder: row r depends on ids[:r+1] only."""
    return np.stack([_feat(ids[: r + 1]) for r in range(len(ids))])


def global_rows(ids):
    """Global toy decoder: every row also depends on the whole input,
    mimicking a global query Gram.  The global summary multiplies a
    vocabulary-dependent direction (a common per-row shift would be
    removed exactly by the softmax's shift invariance)."""
    g = _feat(ids).sum()
    n = len(ids)
    bump = np.stack([0.1 * g * _feat([r, 999983]) for r in range(n)])
    return causal_rows(ids) + bump


def _logsoftmax(rows):
    rows = rows - rows.max(axis=-1, keepdims=True)
    return rows - np.log(np.exp(rows).sum(axis=-1, keepdims=True))


def toy_sequential(rows_fn, ctx, cont):
    lps = []
    for i in range(len(cont)):
        rows = _logsoftmax(rows_fn(seq_prefix(ctx, cont, i)))
        lps.append(rows[-1][cont[i]])
    return np.asarray(lps)


def toy_block(rows_fn, ctx, cont):
    rows = _logsoftmax(rows_fn(block_input(ctx, cont)))
    return gather_block_logprobs(rows, len(ctx), cont)


class TestIndexAlgebra(unittest.TestCase):
    def test_seq_prefix_is_online_prefix(self):
        ctx, cont = [10, 11, 12], [3, 4, 5, 6]
        for i in range(4):
            self.assertEqual(seq_prefix(ctx, cont, i), ctx + cont[:i])
        with self.assertRaises(ValueError):
            seq_prefix(ctx, cont, 4)

    def test_block_input_drops_only_last(self):
        ctx, cont = [10, 11, 12], [3, 4]
        self.assertEqual(block_input(ctx, cont), [10, 11, 12, 3])

    def test_block_rows_predict_next_token(self):
        # row r of a causal call predicts token r+1: y[i] is token n+i of
        # x+y, so its row is n-1+i.
        self.assertEqual(block_score_rows(3, 4), [2, 3, 4, 5])
        self.assertEqual(block_score_rows(1, 1), [0])

    def test_single_token_reduces_to_context_call(self):
        ctx = [7, 8, 9]
        self.assertEqual(block_input(ctx, [2]), ctx)
        self.assertEqual(block_score_rows(len(ctx), 1), [len(ctx) - 1])
        self.assertEqual(seq_prefix(ctx, [2], 0), ctx)

    def test_gather_block_logprobs(self):
        rows = np.arange(12, dtype=np.float64).reshape(4, 3)
        got = gather_block_logprobs(rows, 3, [1, 2])
        self.assertEqual(got.tolist(), [rows[2][1], rows[3][2]])


class TestSemanticOracle(unittest.TestCase):
    """Sequential and block scoring coincide exactly on a
    prefix-consistent toy decoder and generically differ on a global
    one; any indexing error in the block gather breaks the first
    assertion."""

    CTX = [5, 1, 4]
    CONT = [2, 0, 3, 6]

    def test_causal_toy_scores_coincide_exactly(self):
        lps = toy_sequential(causal_rows, self.CTX, self.CONT)
        lpb = toy_block(causal_rows, self.CTX, self.CONT)
        self.assertTrue(np.array_equal(lps, lpb))

    def test_global_toy_scores_differ(self):
        lps = toy_sequential(global_rows, self.CTX, self.CONT)
        lpb = toy_block(global_rows, self.CTX, self.CONT)
        self.assertGreater(np.abs(lps - lpb).max(), 1e-6)

    def test_global_toy_single_token_still_coincides(self):
        lps = toy_sequential(global_rows, self.CTX, [2])
        lpb = toy_block(global_rows, self.CTX, [2])
        self.assertTrue(np.array_equal(lps, lpb))

    def test_misassembled_gather_fails_oracle(self):
        # An off-by-one in the score rows (a plausible mis-assembly)
        # must be caught by the causal-toy oracle.
        rows = _logsoftmax(causal_rows(block_input(self.CTX, self.CONT)))
        wrong = np.asarray([rows[r + 1][t] if r + 1 < len(rows)
                            else rows[r][t]
                            for r, t in zip(
                                block_score_rows(len(self.CTX),
                                                 len(self.CONT)),
                                self.CONT)])
        lps = toy_sequential(causal_rows, self.CTX, self.CONT)
        self.assertFalse(np.array_equal(lps, wrong))


class TestComparisonStats(unittest.TestCase):
    def test_delta_stats(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.5, 3.0])
        d = delta_stats(a, b)
        self.assertEqual(d["max_abs"], 0.5)
        self.assertEqual(d["nonzero"], 1)
        self.assertEqual(d["entries"], 3)
        z = delta_stats(a, a)
        self.assertEqual((z["max_abs"], z["l2"], z["nonzero"]), (0.0, 0.0, 0))

    def test_rank_comparison(self):
        same = rank_comparison([3.0, 2.0, 1.0], [30.0, 20.0, 10.0])
        self.assertFalse(same["argmax_flip"])
        self.assertEqual(same["discordant_pairs"], 0)
        flip = rank_comparison([3.0, 2.0, 1.0], [1.0, 2.0, 3.0])
        self.assertTrue(flip["argmax_flip"])
        self.assertEqual(flip["discordant_pairs"], 3)
        self.assertEqual(flip["pairs"], 3)
        tie = rank_comparison([1.0, 1.0], [2.0, 1.0])
        self.assertEqual(tie["discordant_pairs"], 0)

    def test_summarize(self):
        s = summarize([1.0, 3.0, 2.0])
        self.assertEqual((s["min"], s["median"], s["max"], s["n"]),
                         (1.0, 2.0, 3.0, 3))


class TestRandomPairs(unittest.TestCase):
    def test_deterministic_and_prefix_shared(self):
        a1, b1 = random_pairs(100, 8, 12, 4, 123)
        a2, b2 = random_pairs(100, 8, 12, 4, 123)
        self.assertTrue(np.array_equal(a1, a2))
        self.assertTrue(np.array_equal(b1, b2))
        self.assertTrue(np.array_equal(a1[:, :4], b1[:, :4]))
        # nonzero offsets mod vocab: every suffix position changes
        self.assertTrue((a1[:, 4:] != b1[:, 4:]).all())
        self.assertTrue((0 <= b1).all() and (b1 < 100).all())


@unittest.skipUnless(os.path.exists(JSON_PATH) and os.path.exists(NPZ_PATH),
                     "shipped audit_online artifacts not present")
class TestShippedArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(JSON_PATH) as f:
            cls.res = json.load(f)
        cls.raw = np.load(NPZ_PATH)

    def test_npz_hash_matches_json(self):
        self.assertEqual(self.res["raw_records"]["sha256"],
                         sha256_file(NPZ_PATH))

    def test_rederive_o4_summaries(self):
        for tag in ("soc110m5", "soc110m1"):
            o4 = self.res[tag]["o4_score_equivalence"]
            tok = self.raw[f"{tag}_o4_token_absdiffs"]
            cand = self.raw[f"{tag}_o4_candidate_absdiffs"]
            self.assertEqual(summarize(tok), o4["per_token_absdiff"])
            self.assertEqual(summarize(cand), o4["per_candidate_absdiff"])
            # candidate sums re-derive from the per-token raw arrays
            k = 0
            for pi, inst in enumerate(o4["instances"]):
                for ci in range(len(inst["candidates"])):
                    s = self.raw[f"{tag}_o4_p{pi}_c{ci}_seq"]
                    b = self.raw[f"{tag}_o4_p{pi}_c{ci}_blk"]
                    self.assertEqual(cand[k],
                                     abs(float(s.sum() - b.sum())))
                    k += 1
            self.assertEqual(k, len(cand))

    def test_rederive_o1_randomized(self):
        for tag in ("soc110m5", "soc110m1"):
            r = self.res[tag]["o1_prefix_rows"]["randomized"]
            la = self.raw[f"{tag}_o1_rand_logits_a"]
            lb = self.raw[f"{tag}_o1_rand_logits_b"]
            self.assertEqual(delta_stats(la, lb), r["suffix_change"])
            lam = self.raw[f"{tag}_o1_rand_logits_a_masked"]
            lbm = self.raw[f"{tag}_o1_rand_logits_b_masked"]
            self.assertEqual(delta_stats(lam, lbm),
                             r["masked_suffix_change"])

    def test_collapsed_checkpoint_verdicts(self):
        # the audited checkpoint: exact-zero historical rows and exact
        # single-token coincidence must hold in the shipped record
        c5 = self.res["soc110m5"]
        self.assertEqual(
            c5["o1_prefix_rows"]["randomized"]["suffix_change"]["nonzero"],
            0)
        self.assertTrue(c5["o2_repeat_call_bitwise"]["all_zero"])
        self.assertTrue(
            c5["o4_score_equivalence"]["single_token_exact_coincidence"])


if __name__ == "__main__":
    unittest.main()

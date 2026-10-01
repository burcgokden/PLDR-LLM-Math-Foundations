"""Model-free unit tests for the sequential-validation audit.

Run with:  python3 -m unittest test_audit_seq -v   (numpy only, no
model, no network).  Covers the wrapper-faithful encode-pair split,
every benchmark request template against hand-built documents, the
accuracy conventions and the TruthfulQA-MC2 probability-mass metric
(hand values, overflow stability, faithfulness of the
split-at-first-0 transcription), the window selection, the summary
derivations, and the analytic block-versus-sequential oracles
(including an MC2-metric oracle); when the shipped
audit_seq_results.json / audit_seq_raw.npz are present, the JSON
summaries of the published AND auxiliary task sections are re-derived
from the raw arrays and compared.
"""
import json
import os
import unittest

import numpy as np

from audit_online_lib import sha256_file
from audit_seq_lib import (
    shift_trailing_space, split_continuation_tokens, encode_pair_split,
    render_arc, render_hellaswag, hellaswag_preprocess, render_piqa,
    render_openbookqa, render_siqa, render_winogrande,
    render_truthfulqa_mc1, render_truthfulqa_mc2,
    acc_pred, acc_norm_pred, mc2_score,
    disjoint_window_starts, position_bucket_medians,
    length_bucket_summary, derive_task_summary, derive_mc2_summary,
    derive_wikitext_summary,
    toy_prefix_decoder, toy_global_decoder, score_with_decoder,
    block_input, seq_prefix, gather_block_logprobs, _TQA_PROMPT)

HERE = os.path.dirname(os.path.abspath(__file__))


class TestEncodePair(unittest.TestCase):
    def test_no_trailing_space_is_identity(self):
        self.assertEqual(shift_trailing_space("Q: x\nA:", " yes"),
                         ("Q: x\nA:", " yes"))

    def test_trailing_spaces_move_to_continuation(self):
        c, y = shift_trailing_space("prefix  ", "tail")
        self.assertEqual((c, y), ("prefix", "  tail"))

    def test_concatenation_invariant(self):
        for ctx, cont in [("a ", "b"), ("a  ", " b"), ("a", "b")]:
            c, y = shift_trailing_space(ctx, cont)
            self.assertEqual(c + y, ctx + cont)

    def test_split_continuation_tokens(self):
        whole, ctx = [5, 6, 7, 8, 9], [5, 6]
        self.assertEqual(split_continuation_tokens(whole, ctx), [7, 8, 9])
        with self.assertRaises(ValueError):
            split_continuation_tokens(whole, [5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            split_continuation_tokens(whole, [6, 5])

    def test_encode_pair_split_wrapper_faithful(self):
        whole = [5, 6, 7, 8, 9]
        c, y, ok = encode_pair_split(whole, [5, 6])
        self.assertEqual((c, y, ok), ([5, 6], [7, 8, 9], True))
        # boundary merge: the joint encoding is still what is scored,
        # the mismatch is only recorded
        c, y, ok = encode_pair_split(whole, [5, 99])
        self.assertEqual((c, y, ok), ([5, 6], [7, 8, 9], False))
        with self.assertRaises(ValueError):
            encode_pair_split(whole, [])
        with self.assertRaises(ValueError):
            encode_pair_split(whole, [5, 6, 7, 8, 9])


class TestTemplates(unittest.TestCase):
    def test_arc_letter_and_digit_labels(self):
        doc = dict(question="Why?",
                   choices=dict(text=["t0", "t1", "t2", "t3"],
                                label=["A", "B", "C", "D"]),
                   answerKey="C")
        r = render_arc(doc)
        self.assertEqual(r["requests"][0],
                         ("Question: Why?\nAnswer:", " t0"))
        self.assertEqual(r["gold"], 2)
        self.assertEqual(r["choice_char_lens"], [2.0, 2.0, 2.0, 2.0])
        doc["choices"]["label"] = ["1", "2", "3", "4"]
        doc["answerKey"] = "2"
        self.assertEqual(render_arc(doc)["gold"], 1)

    def test_hellaswag_preprocess_and_render(self):
        self.assertEqual(hellaswag_preprocess("a [title] b [x] c  d"),
                         "a. b c d")
        doc = dict(activity_label="Cooking", ctx_a="He stirs the pot.",
                   ctx_b="then he", endings=["adds salt", "leaves"],
                   label="1")
        r = render_hellaswag(doc)
        self.assertEqual(r["requests"][0][0],
                         "Cooking: He stirs the pot. Then he")
        self.assertEqual(r["requests"][0][1], " adds salt")
        self.assertEqual(r["gold"], 1)

    def test_piqa(self):
        r = render_piqa(dict(goal="open a jar", sol1="twist the lid",
                             sol2="hit it", label=0))
        self.assertEqual(r["requests"][1],
                         ("Question: open a jar\nAnswer:", " hit it"))
        self.assertEqual(r["gold"], 0)

    def test_openbookqa_lstrips_answer_key(self):
        doc = dict(question_stem="Photosynthesis needs",
                   choices=dict(text=["light", "dark"], label=["A", "B"]),
                   answerKey=" A")
        r = render_openbookqa(doc)
        self.assertEqual(r["requests"][0], ("Photosynthesis needs",
                                            " light"))
        self.assertEqual(r["gold"], 0)

    def test_siqa_one_based_label(self):
        r = render_siqa(dict(context="Alex helped.", question="Why?",
                             answerA="kind", answerB="bored",
                             answerC="lost", label="3"))
        self.assertEqual(r["requests"][0][0],
                         "Q: Alex helped. Why?\nA:")
        self.assertEqual(r["gold"], 2)

    def test_winogrande_multiple_input_form(self):
        doc = dict(sentence="The cup broke because _ was fragile.",
                   option1="the cup", option2="the table", answer="1")
        r = render_winogrande(doc)
        self.assertEqual(
            r["requests"][0],
            ("The cup broke because the cup", " was fragile."))
        self.assertEqual(
            r["requests"][1],
            ("The cup broke because the table", " was fragile."))
        self.assertEqual(r["gold"], 0)
        # normalized accuracy divides by the CONTEXT string lengths here
        self.assertEqual(r["choice_char_lens"],
                         [float(len(r["requests"][0][0])),
                          float(len(r["requests"][1][0]))])

    def test_truthfulqa_prompt_and_gold(self):
        doc = dict(question="What is 2+2?",
                   mc1_targets=dict(choices=["4", "5", "22"],
                                    labels=[1, 0, 0]))
        r = render_truthfulqa_mc1(doc)
        self.assertTrue(r["requests"][0][0].startswith(
            "Q: What is human life expectancy"))
        self.assertTrue(r["requests"][0][0].endswith(
            "\n\nQ: What is 2+2?\nA:"))
        self.assertEqual(r["gold"], 0)
        self.assertEqual(len(_TQA_PROMPT.splitlines()), 17)

    def test_truthfulqa_mc2_render(self):
        doc = dict(question="What is 2+2?",
                   mc1_targets=dict(choices=["4"], labels=[1]),
                   mc2_targets=dict(choices=["4", "four", "5", "22"],
                                    labels=[1, 1, 0, 0]))
        r = render_truthfulqa_mc2(doc)
        # identical context to the mc1 renderer (the pinned mc2 task
        # config includes the mc1 prompt configuration)
        self.assertEqual(r["requests"][0][0],
                         _TQA_PROMPT + "\n\nQ: What is 2+2?\nA:")
        self.assertEqual([c for _, c in r["requests"]],
                         [" 4", " four", " 5", " 22"])
        self.assertEqual(r["labels"], [1, 1, 0, 0])
        self.assertEqual(r["choice_char_lens"], [1.0, 4.0, 1.0, 2.0])
        self.assertNotIn("gold", r)


class TestAccuracy(unittest.TestCase):
    def test_acc_pred(self):
        self.assertEqual(acc_pred([-3.0, -1.0, -2.0]), 1)

    def test_acc_norm_flips_with_length(self):
        lls, lens = [-4.0, -4.5], [10.0, 30.0]
        self.assertEqual(acc_pred(lls), 0)
        self.assertEqual(acc_norm_pred(lls, lens), 1)
        with self.assertRaises(ValueError):
            acc_norm_pred(lls, [10.0, 0.0])
        with self.assertRaises(ValueError):
            acc_norm_pred(lls, [10.0])


class TestMc2Metric(unittest.TestCase):
    def test_hand_value(self):
        # two true, one false: (e^-1 + e^-2) / (e^-1 + e^-2 + e^-3)
        lls, labels = [-1.0, -2.0, -3.0], [1, 1, 0]
        expect = ((np.exp(-1) + np.exp(-2))
                  / (np.exp(-1) + np.exp(-2) + np.exp(-3)))
        self.assertAlmostEqual(mc2_score(lls, labels), expect, places=15)

    def test_matches_naive_formula_when_it_does_not_overflow(self):
        # the pinned source computes p = exp(ll) without max
        # subtraction; on non-overflowing inputs the two must agree
        rng = np.random.default_rng(3)
        for _ in range(50):
            n_true = int(rng.integers(1, 4))
            n_false = int(rng.integers(1, 4))
            lls = rng.normal(-5, 3, size=n_true + n_false)
            labels = [1] * n_true + [0] * n_false
            p = np.exp(lls)
            naive = p[:n_true].sum() / p.sum()
            self.assertAlmostEqual(mc2_score(lls, labels), naive,
                                   places=12)

    def test_overflow_stable(self):
        # the naive formula overflows to inf/inf = nan here; the
        # max-subtracted implementation is exact
        lls, labels = [1000.0, 999.0, 998.0], [1, 0, 0]
        with np.errstate(over="ignore", invalid="ignore"):
            p = np.exp(np.asarray(lls))
            self.assertTrue(np.isnan(p[:1].sum() / p.sum()))
        expect = 1.0 / (1.0 + np.exp(-1.0) + np.exp(-2.0))
        self.assertAlmostEqual(mc2_score(lls, labels), expect, places=15)

    def test_split_faithful_to_source(self):
        # the pinned source splits at the FIRST 0; sortedness makes
        # that equal to selecting labels == 1, which is what the
        # implementation verifies before splitting
        lls = [-1.0, -2.0, -3.0, -4.0]
        by_split = mc2_score(lls, [1, 1, 0, 0])
        p = np.exp(np.asarray(lls))
        by_mask = p[np.asarray([1, 1, 0, 0]) == 1].sum() / p.sum()
        self.assertAlmostEqual(by_split, by_mask, places=15)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            mc2_score([-1.0, -2.0], [1, 1])       # no false choice
        with self.assertRaises(ValueError):
            mc2_score([-1.0, -2.0], [0, 1])       # not sorted true-first
        with self.assertRaises(ValueError):
            mc2_score([-1.0, -2.0, -3.0], [1, 0, 1])  # 1 after first 0
        with self.assertRaises(ValueError):
            mc2_score([-1.0, -2.0], [1, 2])       # non-binary label
        with self.assertRaises(ValueError):
            mc2_score([-1.0], [1, 0])             # shape mismatch


class TestWindows(unittest.TestCase):
    def test_disjoint_sorted_deterministic(self):
        s1 = disjoint_window_starts(10000, 12, 257, seed=7)
        s2 = disjoint_window_starts(10000, 12, 257, seed=7)
        self.assertEqual(s1, s2)
        self.assertEqual(s1, sorted(s1))
        self.assertEqual(len(s1), 12)
        for a, b in zip(s1, s1[1:]):
            self.assertGreaterEqual(b - a, 257)
        self.assertTrue(all(st % 257 == 0 and st + 257 <= 10000
                            for st in s1))
        with self.assertRaises(ValueError):
            disjoint_window_starts(1000, 12, 257, seed=7)

    def test_position_buckets(self):
        g = np.arange(8, dtype=np.float64).reshape(2, 4)
        b = position_bucket_medians(g, 2)
        self.assertEqual(b[0]["positions"], [0, 2])
        self.assertEqual(b[0]["median"], 2.5)
        self.assertEqual(b[1]["median"], 4.5)

    def test_length_buckets(self):
        out = length_bucket_summary([1.0, 2.0, 3.0], [1, 4, 4],
                                    [1, 2, 5, 10])
        self.assertEqual(out[0]["n"], 1)
        self.assertEqual(out[1]["n"], 2)
        self.assertEqual(out[1]["median"], 2.5)
        self.assertEqual(out[2]["n"], 0)
        self.assertNotIn("median", out[2])


class TestDeriveTaskSummary(unittest.TestCase):
    def build(self):
        # two items; item 0 has two candidates (lens 1, 2), item 1 has
        # two candidates (lens 2, 1); crafted so raw argmax agrees but
        # one normalized argmax flips between protocols.
        seq_flat = np.array([-1.0, -0.5, -0.7, -2.0, -0.6, -3.0])
        blk_flat = np.array([-1.0, -0.5, -0.9, -2.0, -0.6, -3.0])
        cand_lens = [1, 2, 2, 1]
        item_ncands = [2, 2]
        gold = [0, 0]
        char_lens = [4.0, 40.0, 10.0, 10.0]
        return seq_flat, blk_flat, cand_lens, item_ncands, gold, char_lens

    def test_hand_case(self):
        s = derive_task_summary(*self.build())
        self.assertEqual(s["n_items"], 2)
        self.assertEqual(s["n_candidates"], 4)
        self.assertEqual(s["n_scored_tokens"], 6)
        # item 0: seq sums (-1.0, -1.2) vs blk (-1.0, -1.4); raw pred 0
        # both ways; normalized seq: (-.25, -.03) -> 1, blk
        # (-.25, -.035) -> 1: no flip; item 1: sums equal, no flips.
        self.assertEqual(s["raw_argmax_changes"], 0)
        self.assertEqual(s["acc_block"], 1.0)
        self.assertEqual(s["acc_sequential"], 1.0)
        self.assertAlmostEqual(s["per_token_absdiff"]["max"], 0.2)
        self.assertAlmostEqual(s["per_candidate_absdiff"]["max"], 0.2)
        self.assertEqual(s["pairs_total"], 2)
        self.assertEqual(s["discordant_pairs"], 0)
        self.assertTrue(s["single_token_exact_coincidence"])
        # margins: item 0 blk margin 0.4, gap max 0.2 >= 0.2 -> counted;
        # item 1 margin 0.4, gap 0 -> not counted.
        self.assertEqual(s["items_gap_above_half_margin"], 1)

    def test_single_token_violation_detected(self):
        seq, blk, cl, nc, g, ch = self.build()
        blk = blk.copy()
        blk[5] += 1e-9  # perturb a single-token candidate
        s = derive_task_summary(seq, blk, cl, nc, g, ch)
        self.assertFalse(s["single_token_exact_coincidence"])

    def test_norm_flip_counted(self):
        # one item, two single-token candidates; raw ordering flips
        # between protocols.
        s = derive_task_summary([-1.0, -2.0], [-2.0, -1.0], [1, 1], [2],
                                [0], [5.0, 5.0])
        self.assertEqual(s["raw_argmax_changes"], 1)
        self.assertEqual(s["norm_argmax_changes"], 1)
        self.assertEqual(s["discordant_pairs"], 1)
        self.assertEqual(s["acc_block"] + s["acc_sequential"], 1.0)

    def test_inconsistent_shapes_raise(self):
        with self.assertRaises(ValueError):
            derive_task_summary([-1.0], [-1.0], [2], [1], [0], [1.0])
        with self.assertRaises(ValueError):
            derive_task_summary([-1.0], [-1.0], [1], [2], [0], [1.0])


class TestDeriveMc2Summary(unittest.TestCase):
    def build(self):
        # two items: item 0 has three candidates (true, true, false;
        # lens 1, 2, 1), item 1 has two candidates (true, false;
        # lens 2, 1); the block/sequential scores differ only on the
        # first candidate of item 1 (a 2-token candidate).
        seq_flat = np.array([-1.0, -0.5, -0.7, -2.0, -0.6, -0.9, -3.0])
        blk_flat = np.array([-1.0, -0.5, -0.7, -2.0, -0.6, -1.1, -3.0])
        cand_lens = [1, 2, 1, 2, 1]
        item_ncands = [3, 2]
        labels = [1, 1, 0, 1, 0]
        char_lens = [4.0, 8.0, 4.0, 6.0, 3.0]
        return (seq_flat, blk_flat, cand_lens, item_ncands, labels,
                char_lens)

    def test_hand_case(self):
        s = derive_mc2_summary(*self.build())
        self.assertEqual(s["n_items"], 2)
        self.assertEqual(s["n_candidates"], 5)
        self.assertEqual(s["n_scored_tokens"], 7)
        self.assertEqual(s["n_true_choices"], 3)
        # item 0: candidate sums identical between protocols, so its
        # metric gap is exactly 0
        sums_b0 = [-1.0, -1.2, -2.0]
        m0 = mc2_score(sums_b0, [1, 1, 0])
        # item 1: block sums (-1.7, -3.0), sequential (-1.5, -3.0)
        m1b = mc2_score([-1.7, -3.0], [1, 0])
        m1s = mc2_score([-1.5, -3.0], [1, 0])
        self.assertAlmostEqual(s["mc2_block_mean"], (m0 + m1b) / 2,
                               places=15)
        self.assertAlmostEqual(s["mc2_sequential_mean"], (m0 + m1s) / 2,
                               places=15)
        self.assertAlmostEqual(s["per_item_metric_absdiff"]["max"],
                               abs(m1s - m1b), places=15)
        self.assertAlmostEqual(s["per_item_metric_signed"]["mean"],
                               (m1s - m1b) / 2, places=15)
        self.assertAlmostEqual(s["per_candidate_absdiff"]["max"], 0.2)
        self.assertEqual(s["diagnostic_raw_argmax_changes"], 0)
        self.assertTrue(s["single_token_exact_coincidence"])

    def test_single_token_violation_detected(self):
        seq, blk, cl, nc, lab, ch = self.build()
        blk = blk.copy()
        blk[6] += 1e-9  # perturb the last single-token candidate
        s = derive_mc2_summary(seq, blk, cl, nc, lab, ch)
        self.assertFalse(s["single_token_exact_coincidence"])

    def test_diagnostic_flip_counted(self):
        # one item, two single-token candidates (true, false); the raw
        # ordering flips between protocols; the metric moves but has
        # no argmax of its own
        s = derive_mc2_summary([-1.0, -2.0], [-2.0, -1.0], [1, 1], [2],
                               [1, 0], [5.0, 5.0])
        self.assertEqual(s["diagnostic_raw_argmax_changes"], 1)
        self.assertAlmostEqual(
            s["per_item_metric_absdiff"]["max"],
            abs(mc2_score([-1.0, -2.0], [1, 0])
                - mc2_score([-2.0, -1.0], [1, 0])), places=15)

    def test_inconsistent_shapes_raise(self):
        with self.assertRaises(ValueError):
            derive_mc2_summary([-1.0], [-1.0], [2], [1], [1], [1.0])
        with self.assertRaises(ValueError):
            derive_mc2_summary([-1.0, -2.0], [-1.0, -2.0], [1, 1], [2],
                               [1], [1.0, 1.0])
        with self.assertRaises(ValueError):
            # labels of an item not sorted true-first propagate the
            # mc2_score validation
            derive_mc2_summary([-1.0, -2.0], [-1.0, -2.0], [1, 1], [2],
                               [0, 1], [1.0, 1.0])


class TestDeriveWikitextSummary(unittest.TestCase):
    def test_hand_case(self):
        seq = np.array([[-1.0, -2.0], [-1.5, -2.5]])
        blk = np.array([[-1.0, -1.5], [-1.5, -2.0]])
        s = derive_wikitext_summary(seq, blk, n_buckets=2)
        self.assertEqual(s["tokens_scored"], 4)
        self.assertAlmostEqual(s["sequential_nll_per_token"], 1.75)
        self.assertAlmostEqual(s["block_ce_per_token"], 1.5)
        self.assertAlmostEqual(s["nll_gap_per_token"], 0.25)
        self.assertEqual(s["per_token_signed"]["frac_sequential_worse"],
                         0.5)
        self.assertEqual(s["by_position"][0]["median"], 0.0)
        self.assertEqual(s["by_position"][1]["median"], 0.5)
        with self.assertRaises(ValueError):
            derive_wikitext_summary(seq, blk[:1])


class TestOracles(unittest.TestCase):
    def test_prefix_consistent_decoder_scores_coincide(self):
        for ctx, cont in [([1, 2, 3], [4]), ([1, 2, 3], [4, 5, 6]),
                          ([2], [3, 1, 0, 6, 5])]:
            seq, blk = score_with_decoder(toy_prefix_decoder, ctx, cont)
            np.testing.assert_array_equal(seq, blk)

    def test_global_decoder_scores_differ(self):
        seq, blk = score_with_decoder(toy_global_decoder, [1, 2, 3],
                                      [4, 5, 6])
        self.assertGreater(np.abs(seq - blk)[:-1].max(), 1e-3)
        # the LAST candidate token is scored from the identical full
        # input under both protocols, so it must coincide exactly
        self.assertEqual(seq[-1], blk[-1])

    def test_global_decoder_hand_derivation(self):
        # re-derive both protocol scores for one token WITHOUT
        # score_with_decoder: candidate token y[0] with a longer
        # candidate present.
        ctx, cont = [1, 2], [3, 4]
        rows_blk = toy_global_decoder(block_input(ctx, cont))  # [1,2,3]
        r = rows_blk[len(ctx) - 1]
        r = r - r.max()
        blk0 = (r - np.log(np.exp(r).sum()))[cont[0]]
        rows_seq = toy_global_decoder(seq_prefix(ctx, cont, 0))  # [1,2]
        q = rows_seq[-1]
        q = q - q.max()
        seq0 = (q - np.log(np.exp(q).sum()))[cont[0]]
        seq, blk = score_with_decoder(toy_global_decoder, ctx, cont)
        self.assertEqual(seq[0], seq0)
        self.assertEqual(blk[0], blk0)
        self.assertNotEqual(seq0, blk0)

    def test_single_token_candidates_identical_inputs(self):
        ctx, cont = [3, 1, 4], [2]
        self.assertEqual(block_input(ctx, cont), seq_prefix(ctx, cont, 0))

    def _mc2_item_scores(self, decoder, ctx, cands, labels):
        """Score each candidate of one toy MC2 item under both
        protocols, then aggregate with the published metric."""
        blk_ll, seq_ll = [], []
        for cont in cands:
            seq, blk = score_with_decoder(decoder, ctx, cont)
            seq_ll.append(seq.sum())
            blk_ll.append(blk.sum())
        return (mc2_score(blk_ll, labels), mc2_score(seq_ll, labels))

    def test_mc2_metric_oracle_prefix_decoder(self):
        # a prefix-consistent decoder gives identical candidate sums
        # under both protocols, so the MC2 metric coincides exactly
        mb, ms = self._mc2_item_scores(toy_prefix_decoder, [1, 2, 3],
                                       [[4, 5], [6], [2, 0, 1]],
                                       [1, 1, 0])
        self.assertEqual(mb, ms)

    def test_mc2_metric_oracle_global_decoder(self):
        # a global decoder moves multi-token candidate sums between
        # protocols, so the MC2 metric differs
        mb, ms = self._mc2_item_scores(toy_global_decoder, [1, 2, 3],
                                       [[4, 5], [6], [2, 0, 1]],
                                       [1, 1, 0])
        self.assertNotEqual(mb, ms)
        self.assertGreater(abs(mb - ms), 1e-6)
        for m in (mb, ms):
            self.assertGreaterEqual(m, 0.0)
            self.assertLessEqual(m, 1.0)


class TestShippedArtifacts(unittest.TestCase):
    """Re-derive the shipped JSON summaries from the raw arrays."""

    @classmethod
    def setUpClass(cls):
        jpath = os.path.join(HERE, "audit_seq_results.json")
        npath = os.path.join(HERE, "audit_seq_raw.npz")
        cls.res = cls.raw = None
        if os.path.exists(jpath) and os.path.exists(npath):
            with open(jpath) as f:
                cls.res = json.load(f)
            cls.raw = np.load(npath)
            cls.npath = npath

    def need(self):
        if self.res is None:
            self.skipTest("shipped audit_seq artifacts not present")

    def test_raw_hash_matches_json(self):
        self.need()
        self.assertEqual(self.res["raw_records"]["sha256"],
                         sha256_file(self.npath))

    def test_rederive_all_summaries(self):
        self.need()
        from reconstruct_summaries import run
        report = run()
        self.assertEqual(report["status"], "passed")

    def test_published_section_is_mc2_not_mc1(self):
        """The published-task section must contain truthfulqa_mc2 and
        must NOT contain the auxiliary mc1 probe; skipped if the
        artifacts predate the auxiliary-probe layout."""
        self.need()
        if "s2_auxiliary" not in self.res["soc110m5"]:
            self.skipTest("artifacts predate the auxiliary-probe layout")
        for tag in ("soc110m5", "soc110m1"):
            pub = self.res[tag]["s2_benchmarks"]
            self.assertIn("truthfulqa_mc2", pub)
            self.assertNotIn("truthfulqa_mc1", pub)
            self.assertEqual(len(pub), 8)
            aux = self.res[tag]["s2_auxiliary"]
            self.assertEqual(list(aux), ["truthfulqa_mc1"])
            self.assertEqual(aux["truthfulqa_mc1"]["role"], "auxiliary")
            # both TruthfulQA variants sample the same seeded rows
            np.testing.assert_array_equal(
                self.raw[f"{tag}_truthfulqa_mc2_item_rows"],
                self.raw[f"{tag}_truthfulqa_mc1_item_rows"])


if __name__ == "__main__":
    unittest.main()

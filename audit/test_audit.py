"""Unit tests for the audit's numerical helpers (audit_lib.py) and
consistency checks of the shipped audit_results.json / audit_raw.npz.

Model-free: needs only numpy and the standard library (no torch, no GPU,
no checkpoint download).  Run from this directory with either

    python3 -m unittest -v
    pytest

The shipped-results tests recompute the JSON's summary statistics from
the raw per-instance arrays in audit_raw.npz (whose SHA-256 the JSON
records), so the summaries cannot silently drift from the raw data.
"""
import hashlib
import json
import math
import os
import unittest

import numpy as np

from audit_lib import (DK, THETA, C_theta_of, ap_tensor, assemble_chain,
                       build_freqs, commutant_residual, dag_loss_value,
                       freq_energy, glu_jac_bound, iswiglu_dmax,
                       iswiglu_dprime_env, layernorm_row, ln_jac_bound,
                       ln_pair_error, margin_ok, mult, numrank, order_pair,
                       rep_stats, rope_basis, rope_rotation, sha256_file,
                       silu_dmax, silu_dprime_env, spec2, stats, svdvals)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "audit_results.json")
RAWFILE = os.path.join(HERE, "audit_raw.npz")
# properties of the audited checkpoint PLDR-LLM-v51-SOC-110M-5
NUM_HEADS = 14
NUM_LAYERS = 5
N_PROMPTS = 8
N_INSTANCES = NUM_LAYERS * NUM_HEADS * N_PROMPTS   # 560


class TestTwirl(unittest.TestCase):
    def test_frequency_set(self):
        freqs = build_freqs()
        self.assertEqual(len(freqs), 32 + 2 * 32 * 31)
        self.assertTrue(all(w > 0 for w in freqs.values()))
        self.assertTrue(all(abs(math.sin(w / 2)) > 0 for w in freqs.values()))

    def test_multiplier_basic_properties(self):
        freqs = build_freqs()
        for w in list(freqs.values())[::97] + [2 * THETA[0], 2 * THETA[31]]:
            self.assertAlmostEqual(mult(w, 1), 1.0, places=12)
            for S in (2, 64, 1024):
                m = mult(w, S)
                self.assertGreaterEqual(m, 0.0)
                self.assertLessEqual(m, 1.0 + 1e-12)
                self.assertLessEqual(m, 1 / (S * abs(math.sin(w / 2))) + 1e-15)

    def test_multiplier_resonant_zero(self):
        S = 100
        self.assertAlmostEqual(mult(2 * math.pi / S, S), 0.0, places=12)

    def test_C_theta_value(self):
        C = C_theta_of(build_freqs())
        self.assertAlmostEqual(C / 4.4968e4, 1.0, delta=1e-3)
        self.assertAlmostEqual(C / 1024 / 43.9, 1.0, delta=1e-2)

    def test_rope_basis_diagonalizes_rotations(self):
        # Uc must be unitary and diagonalize R_n with eigenvalues
        # e^{i n omega_j}; this validates the twirl decomposition basis.
        Uc, omega = rope_basis()
        np.testing.assert_allclose(Uc.conj().T @ Uc, np.eye(DK), atol=1e-12)
        for n in (1, 7):
            Rn = rope_rotation(n)
            D = Uc.conj().T @ Rn @ Uc
            np.testing.assert_allclose(np.diag(D), np.exp(1j * n * omega),
                                       atol=1e-12)
            off = D - np.diag(np.diag(D))
            self.assertLess(np.abs(off).max(), 1e-12)

    def test_freq_energy_parseval(self):
        rng = np.random.default_rng(3)
        Uc, omega = rope_basis()
        M = rng.standard_normal((DK, DK))
        En, wdiff = freq_energy(M, Uc, omega)
        self.assertAlmostEqual(En.sum(), (M ** 2).sum(), places=8)
        self.assertEqual(wdiff.shape, (DK, DK))


class TestCommutantResidual(unittest.TestCase):
    def test_commutant_member_has_zero_residual(self):
        # A RoPE rotation commutes with the whole family, so its
        # off-commutant energy is exactly 0.
        Uc, omega = rope_basis()
        self.assertLess(commutant_residual(rope_rotation(5), Uc, omega), 1e-12)

    def test_reflection_is_fully_off_commutant(self):
        # The blockwise reflection diag(1, -1) swaps the two eigenvectors
        # of every plane, so it lies entirely in the off-frequency
        # entries: residual exactly 1.
        Uc, omega = rope_basis()
        G = np.diag([1.0, -1.0] * (DK // 2))
        self.assertAlmostEqual(commutant_residual(G, Uc, omega), 1.0,
                               places=12)

    def test_pythagoras(self):
        # Residual^2 + commutant-fraction = 1 for any G (orthogonal
        # projection, unitary basis).
        rng = np.random.default_rng(11)
        Uc, omega = rope_basis()
        G = rng.standard_normal((DK, DK))
        res = commutant_residual(G, Uc, omega)
        Gc = Uc.conj().T @ G @ Uc
        same = np.abs(omega[:, None] - omega[None, :]) < 1e-12
        comm_energy = float((np.abs(Gc)[same] ** 2).sum())
        total = float((np.abs(Gc) ** 2).sum())
        self.assertAlmostEqual(res ** 2 + comm_energy / total, 1.0,
                               places=10)
        self.assertTrue(0.0 <= res <= 1.0)


class TestLinearAlgebraHelpers(unittest.TestCase):
    def test_svdvals_matches_numpy(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((17, 9))
        np.testing.assert_allclose(svdvals(M),
                                   np.linalg.svd(M, compute_uv=False),
                                   rtol=1e-12)

    def test_svdvals_sanitizes_nonfinite(self):
        counter = {"count": 0}
        M = np.eye(3)
        M[0, 1] = np.nan
        M[2, 0] = np.inf
        s = svdvals(M, counter)
        self.assertEqual(counter["count"], 1)
        self.assertTrue(np.isfinite(s).all())
        np.testing.assert_allclose(s, np.ones(3), rtol=1e-12)

    def test_spec2(self):
        self.assertAlmostEqual(spec2(np.diag([3.0, -7.0, 1.0])), 7.0, places=12)

    def test_numrank(self):
        s = np.zeros(64)
        s[0] = 1.0
        s[1] = 1e-12
        self.assertEqual(numrank(s), 1)
        self.assertEqual(numrank(np.ones(64)), 64)
        s2 = np.ones(64)
        s2[0] = 1e12
        self.assertEqual(numrank(s2), 1)


class TestApTensor(unittest.TestCase):
    """Semantic sentinel test for the power stage: the learned exponent
    tensor P is [H, dk, dk] with no batch axis, and indexing it like
    the batched activations is accepted silently by NumPy broadcasting
    in two different ways.  A sentinel P with all three indices encoded
    distinctly must reproduce an explicit three-index loop and must
    DIFFER from both broadcast-accepted mis-indexing patterns."""

    @classmethod
    def setUpClass(cls):
        h, d = 3, 4
        rng = np.random.default_rng(7)
        cls.ALM = rng.uniform(0.5, 2.0, size=(h, d, d))
        # sentinel: P[h, i, j] distinct in every index
        cls.P = (np.arange(h)[:, None, None] * 1.0
                 + np.arange(d)[None, :, None] * 0.13
                 + np.arange(d)[None, None, :] * 0.017)

    def test_matches_three_index_loop(self):
        ap = ap_tensor(self.ALM, self.P)
        h, d, _ = self.ALM.shape
        loop = np.zeros_like(ap)
        for hh in range(h):
            for i in range(d):
                for j in range(d):
                    loop[hh, i, j] = abs(self.ALM[hh, i, j]) ** self.P[hh, i, j]
        np.testing.assert_allclose(ap, loop, rtol=1e-14)

    def test_fails_under_row_broadcast_pattern(self):
        # pw_n = npy(pw[0]) then pw_n[h]  ->  row h of head
        # 0's matrix, broadcast over every row of A_LM[h]
        ap = ap_tensor(self.ALM, self.P)
        for hh in range(self.ALM.shape[0]):
            buggy = np.power(np.abs(self.ALM[hh]), self.P[0][hh])
            self.assertFalse(np.allclose(buggy, ap[hh]),
                             "sentinel must expose the row-broadcast bug")

    def test_fails_under_head_broadcast_pattern(self):
        # pw = npy(t1[2][0])  ->  head 0's full matrix,
        # broadcast across all heads; only head 0 agrees
        ap = ap_tensor(self.ALM, self.P)
        buggy = np.power(np.abs(self.ALM), self.P[0])
        np.testing.assert_allclose(buggy[0], ap[0], rtol=1e-14)
        for hh in range(1, self.ALM.shape[0]):
            self.assertFalse(np.allclose(buggy[hh], ap[hh]),
                             "sentinel must expose the head-broadcast bug")

    def test_shape_guard_rejects_wrong_shapes(self):
        with self.assertRaises(ValueError):
            ap_tensor(self.ALM, self.P[0])          # matrix, not stack
        with self.assertRaises(ValueError):
            ap_tensor(self.ALM, self.P[:2])         # head-count mismatch
        with self.assertRaises(ValueError):
            ap_tensor(self.ALM[0], self.P[0])       # no head axis


class TestScalarHelpers(unittest.TestCase):
    def test_iswiglu_dmax_against_finite_differences(self):
        for U in (0.5, 3.0, 20.0):
            u = np.linspace(-U, U, 400001)
            f = u * u / (1 + np.exp(-u))
            fd = np.abs(np.diff(f) / np.diff(u)).max()
            self.assertAlmostEqual(iswiglu_dmax(U) / fd, 1.0, delta=1e-3)

    def test_iswiglu_dmax_monotone(self):
        vals = [iswiglu_dmax(U) for U in (0.1, 1.0, 5.0, 50.0)]
        self.assertEqual(vals, sorted(vals))
        self.assertGreater(iswiglu_dmax(10.0), 10.0)

    def test_silu_dmax(self):
        # global supremum of |silu'| is ~1.0998 (near u ~ 2.4); on a small
        # interval around 0 the max is silu'(U)
        self.assertAlmostEqual(silu_dmax(50.0), 1.0998, delta=2e-3)
        self.assertLess(silu_dmax(0.5), 0.9)

    def test_envelopes_dominate_grid_maxima(self):
        # The certified analytic envelopes must dominate the (lower-
        # estimate) grid maxima at every interval radius; this is the
        # property that makes them valid replacements in the chain.
        for U in (0.01, 0.1, 0.5, 2.0, 10.0, 80.0):
            self.assertGreaterEqual(iswiglu_dprime_env(U), iswiglu_dmax(U))
            self.assertGreaterEqual(silu_dprime_env(U), silu_dmax(U))

    def test_envelopes_are_tight_at_small_radius(self):
        # For small U the envelope 2 U sigmoid(U) + U^2/4 approaches the
        # true supremum ~ U (both derivative terms are first-order
        # exact), so the certification costs little where the audit's
        # median U actually sits.
        U = 0.018
        self.assertLess(iswiglu_dprime_env(U) / iswiglu_dmax(U), 1.05)

    def test_stats(self):
        st = stats([3.0, 1.0, 2.0])
        self.assertEqual((st["min"], st["median"], st["max"]), (1.0, 2.0, 3.0))


class TestLayerNorm(unittest.TestCase):
    def test_layernorm_row_reference(self):
        r = np.array([1.0, 2.0, 3.0, 4.0])
        g = np.ones(4); b = np.zeros(4)
        out = layernorm_row(r, g, b, 0.0)
        np.testing.assert_allclose(out, (r - 2.5) / r.std(), rtol=1e-12)

    def test_ln_pair_error_zero_variance_row(self):
        # On a constant row the two LayerNorm outputs coincide exactly
        # (both centered rows are zero), so the DIRECT error is 0 while
        # the first-order proxy diverges: the proxy is a bound, not a
        # measurement.
        r = np.full(8, 3.14)
        d_abs, d_rel, proxy = ln_pair_error(r, 40, np.ones(8), np.zeros(8), 1e-6)
        self.assertEqual(d_abs, 0.0)
        self.assertEqual(d_rel, 0.0)
        self.assertEqual(proxy, float("inf"))

    def test_ln_pair_error_matches_first_order_proxy(self):
        # For v >> eps the direct relative error approaches the proxy
        # eps/(2 v) (first-order expansion), validating both directions.
        rng = np.random.default_rng(1)
        r = rng.standard_normal(64)
        S = 40.0
        eps = 1e-6
        v = (r / S).var()
        d_abs, d_rel, proxy = ln_pair_error(r, S, np.ones(64), np.zeros(64), eps)
        self.assertAlmostEqual(proxy, eps / (2 * v), places=12)
        self.assertAlmostEqual(d_rel / proxy, 1.0, delta=0.1)

    def test_ln_jac_bound(self):
        self.assertAlmostEqual(ln_jac_bound(2.0, 0.04, 1e-6),
                               2.0 / math.sqrt(0.04 + 1e-6), places=12)


class TestGluBound(unittest.TestCase):
    def test_glu_jac_bound_dominates_finite_difference(self):
        # 1-d gated block f(x) = w3 * silu(w1 x) * (w2 x): the product-rule
        # bound with interval extrema must dominate |f'| on the interval.
        w1, w2, w3 = 0.7, -1.3, 0.9
        x = np.linspace(-2, 2, 100001)
        u = w1 * x
        sig = 1 / (1 + np.exp(-u))
        f = w3 * (u * sig) * (w2 * x)
        fp = np.abs(np.diff(f) / np.diff(x)).max()
        bound = glu_jac_bound(abs(w1), abs(w2), abs(w3),
                              x2_maxabs=np.abs(w2 * x).max(),
                              dact_max=silu_dmax(np.abs(u).max()),
                              act_maxabs=np.abs(u * sig).max())
        self.assertGreaterEqual(bound, fp)


class TestOrderParameter(unittest.TestCase):
    def test_identical_tensors_give_zero(self):
        x = np.arange(12.0).reshape(3, 4) + 1
        r, sgn = order_pair(x, x.copy())
        self.assertEqual((r, sgn), (0.0, 0.0))

    def test_all_zero_pair_gives_zero(self):
        # The 0/0 edge case: two all-zero tensors are
        # invariant, so the piecewise statistic is 0, not NaN.
        z = np.zeros((4, 4))
        r, sgn = order_pair(z, z.copy())
        self.assertEqual((r, sgn), (0.0, 0.0))
        self.assertFalse(math.isnan(r))

    def test_known_value(self):
        x1 = np.ones((2, 2))
        x2 = -np.ones((2, 2))
        r, sgn = order_pair(x1, x2)
        self.assertAlmostEqual(r, 2.0, places=12)
        self.assertAlmostEqual(sgn, 2.0, places=12)

    def test_signed_infinite_when_means_cancel(self):
        x1 = np.array([1.0, -1.0])
        x2 = np.array([-1.0, 1.0])
        r, sgn = order_pair(x1, x2)
        self.assertGreater(r, 0.0)
        self.assertEqual(sgn, float("inf"))


class TestMarginCriterion(unittest.TestCase):
    def test_factor_two_margin_example(self):
        # Logits (1,0), margin 1, l_inf perturbation 0.6: the argmax can
        # flip ((0.4, 0.6)), so B < margin is NOT sufficient; 2B < margin
        # is.
        self.assertFalse(margin_ok(0.6, 1.0))
        self.assertTrue(margin_ok(0.49, 1.0))
        self.assertFalse(margin_ok(0.5, 1.0))   # boundary excluded


class TestDagLoss(unittest.TestCase):
    def test_acyclic_is_zero(self):
        M = np.array([[0.0, 2.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
        self.assertAlmostEqual(dag_loss_value(M), 0.0, places=10)

    def test_identity_value(self):
        # M = I: N = I, tr e^N = d e, DL = |log e| = 1
        self.assertAlmostEqual(dag_loss_value(np.eye(5)), 1.0, places=10)

    def test_overflow_safe(self):
        # Entries large enough that tr e^{M o M} overflows float64: the
        # log-sum-exp path must still return a finite value ~ lam_max.
        M = np.full((4, 4), 40.0)
        val = dag_loss_value(M)
        self.assertTrue(np.isfinite(val))
        self.assertAlmostEqual(val, 4 * 1600 - math.log(4), delta=1.0)


class TestRepStats(unittest.TestCase):
    def test_repeated_text(self):
        st = rep_stats("a b c d " * 10)
        self.assertEqual(st["n_tokens"], 40)
        self.assertLess(st["distinct1"], 0.2)
        self.assertGreater(st["dup4_frac"], 0.8)

    def test_distinct_text(self):
        st = rep_stats("one two three four five six seven eight")
        self.assertEqual(st["distinct1"], 1.0)
        self.assertEqual(st["dup4_frac"], 0.0)


def _silu(u):
    return u / (1.0 + np.exp(-u))


class TestChainAssembly(unittest.TestCase):
    """Hand computations DERIVED FROM THE DECODER MAP, not from the
    helper: an injection at the attention output of layer l traverses
    that layer's own post-attention remainder LLN1*(1+Lffn)*LLN2 before
    the downstream Llayer products and the unembedding.  A unit test
    that repeats a helper's own formula is not a semantic oracle;
    TestChainOracle below checks the same path structure against
    finite differences of an actual miniature decoder."""

    def test_single_layer(self):
        # one layer: the injection still traverses that layer's own
        # remainder before the unembedding (a downstream-only assembly
        # would carry NO same-layer factor at all)
        per = [dict(WO=1.5, LLN1=2.0, Lffn=3.0, LLN2=0.5, Llayer=99.0)]
        post = 2.0 * (1 + 3.0) * 0.5                       # 4.0
        expected = 2.0 * 1.5 * 3.0 * post * 4.0            # hf*WO*sqrt(9)*post*Wvocab
        self.assertAlmostEqual(assemble_chain(per, 2.0, 9, 4.0),
                               expected, places=12)

    def test_two_layer_hand_computation(self):
        per_layer = [dict(WO=2.0, LLN1=1.5, Lffn=1.0, LLN2=0.5, Llayer=10.0),
                     dict(WO=3.0, LLN1=2.0, Lffn=0.0, LLN2=1.0, Llayer=100.0)]
        hfmax, H, Wvocab = 5.0, 4, 7.0
        post1 = 1.5 * (1 + 1.0) * 0.5                      # 1.5
        post2 = 2.0 * (1 + 0.0) * 1.0                      # 2.0
        expected = (hfmax * 2.0 * 2 * post1 * 100.0 * Wvocab
                    + hfmax * 3.0 * 2 * post2 * 1.0 * Wvocab)
        self.assertAlmostEqual(assemble_chain(per_layer, hfmax, H, Wvocab),
                               expected, places=9)

    def test_injection_layer_llayer_unused(self):
        # the injection layer's OWN Llayer (which contains 1 + Lattn)
        # must not enter its term: the perturbation enters at the
        # attention output, not at the layer input
        per = [dict(WO=1.0, LLN1=1.0, Lffn=0.0, LLN2=1.0, Llayer=1e6)]
        self.assertAlmostEqual(assemble_chain(per, 1.0, 1, 1.0), 1.0,
                               places=12)

    def test_same_layer_keys_required(self):
        # a record without the same-layer factors must fail loudly,
        # never default to the omitted-factor behavior
        with self.assertRaises(KeyError):
            assemble_chain([dict(WO=1.5, Llayer=99.0)], 2.0, 9, 4.0)


class TestChainOracle(unittest.TestCase):
    """Miniature-decoder finite-difference oracle: the assembled
    coefficient, evaluated on factors measured from an ACTUAL
    two-layer post-attention path, must dominate the measured logit
    sensitivity to an injection at the attention output -- and a
    downstream-only mis-assembly must FAIL domination on a
    small-variance input, proving the same-layer remainder is
    load-bearing.  The oracle evaluates the model, not the formula, so
    it is independent of the helper's algebra."""

    D = 6

    @classmethod
    def _forward_capture(cls, x0, deltas, params, Wv):
        """Column-convention miniature decoder: per layer, the
        attention output is frozen at zero and the injected
        perturbation delta enters exactly where assemble_chain models
        it (the concatenated head output, before W_O); then
        LN1(x + WO@delta), the gated FFN residual, LN2.  Returns the
        logits and the visited pre-LN rows / FFN inputs per layer."""
        x = x0
        cap = []
        for lp, dl in zip(params, deltas):
            pre1 = x + lp["WO"] @ dl
            u = layernorm_row(pre1, lp["g1"], lp["b1"], lp["eps1"])
            f = lp["W3"] @ (_silu(lp["W1"] @ u) * (lp["W2"] @ u))
            pre2 = u + f
            cap.append(dict(pre1=pre1, pre2=pre2, u=u))
            x = layernorm_row(pre2, lp["g2"], lp["b2"], lp["eps2"])
        return Wv @ x, cap

    @classmethod
    def _measured_factors(cls, lp, caps):
        """Certified per-layer factors, measured over the visited
        states of all runs in caps: LN bounds max|gamma|/sqrt(vmin+eps)
        at the minimum visited variance, FFN by the product rule with
        the certified silu-derivative envelope on the visited
        preactivation range, spectral norms exact."""
        vmin1 = min(float(np.var(c["pre1"])) for c in caps)
        vmin2 = min(float(np.var(c["pre2"])) for c in caps)
        Umax = max(float(np.abs(lp["W1"] @ c["u"]).max()) for c in caps)
        x2max = max(float(np.abs(lp["W2"] @ c["u"]).max()) for c in caps)
        actmax = max(float(np.abs(_silu(lp["W1"] @ c["u"])).max()) for c in caps)
        LLN1 = ln_jac_bound(float(np.abs(lp["g1"]).max()), vmin1, lp["eps1"])
        LLN2 = ln_jac_bound(float(np.abs(lp["g2"]).max()), vmin2, lp["eps2"])
        Lffn = glu_jac_bound(spec2(lp["W1"]), spec2(lp["W2"]), spec2(lp["W3"]),
                             x2max, silu_dprime_env(Umax), actmax)
        # frozen attention => Lattn = 0 in the downstream layer bound
        return dict(WO=spec2(lp["WO"]), LLN1=LLN1, Lffn=Lffn, LLN2=LLN2,
                    Llayer=LLN2 * (1 + Lffn) * LLN1)

    def test_corrected_assembly_dominates_finite_difference(self):
        rng = np.random.default_rng(42)
        D = self.D
        params = []
        for _ in range(2):
            params.append(dict(
                WO=0.5 * rng.standard_normal((D, D)),
                g1=1.0 + 0.2 * rng.standard_normal(D),
                b1=0.1 * rng.standard_normal(D), eps1=1e-6,
                g2=1.0 + 0.2 * rng.standard_normal(D),
                b2=0.1 * rng.standard_normal(D), eps2=1e-6,
                W1=0.4 * rng.standard_normal((D, D)),
                W2=0.4 * rng.standard_normal((D, D)),
                W3=0.4 * rng.standard_normal((D, D))))
        Wv = 0.7 * rng.standard_normal((3, D))
        x0 = rng.standard_normal(D)
        delta = 1e-6 * rng.standard_normal(D)
        y0, cap0 = self._forward_capture(x0, [np.zeros(D)] * 2, params, Wv)
        y1, cap1 = self._forward_capture(x0, [delta, np.zeros(D)], params, Wv)
        fd = float(np.abs(y1 - y0).max())
        per_layer = [self._measured_factors(params[li], [cap0[li], cap1[li]])
                     for li in range(2)]
        C = assemble_chain(per_layer, 1.0, 1, spec2(Wv))
        # tiny injection: first-order dominates, 1% slack absorbs the
        # O(delta) drift of the segment extrema
        self.assertGreater(fd, 0.0)
        self.assertLessEqual(fd, C * float(np.linalg.norm(delta)) * 1.01)

    def test_downstream_only_assembly_fails_domination(self):
        # a single layer whose post-attention LayerNorm genuinely
        # amplifies (small input variance, LLN1 ~ 1e4): the
        # downstream-only formula -- which for one layer reduces to
        # hf * ||WO|| * sqrt(H) * ||Wvocab|| with NO same-layer factor
        # -- is violated by the measured finite difference by orders of
        # magnitude, while the correct assembly still dominates
        D = self.D
        rng = np.random.default_rng(7)
        x0 = np.full(D, 5.0) + 1e-4 * rng.standard_normal(D)  # var ~ 1e-8
        params = [dict(WO=np.eye(D),
                       g1=np.ones(D), b1=np.zeros(D), eps1=1e-12,
                       g2=np.ones(D), b2=np.zeros(D), eps2=1e-6,
                       W1=np.zeros((D, D)), W2=np.zeros((D, D)),
                       W3=np.zeros((D, D)))]
        Wv = np.eye(D)
        delta = 1e-9 * rng.standard_normal(D)
        y0, cap0 = self._forward_capture(x0, [np.zeros(D)], params, Wv)
        y1, cap1 = self._forward_capture(x0, [delta], params, Wv)
        fd = float(np.abs(y1 - y0).max())
        dn = float(np.linalg.norm(delta))
        old_formula = 1.0 * spec2(params[0]["WO"]) * 1.0 * spec2(Wv)  # = 1
        self.assertGreater(fd, 5.0 * old_formula * dn,
                           "the same-layer LayerNorm amplification is real; "
                           "the downstream-only assembly must fail here")
        per = [self._measured_factors(params[0], [cap0[0], cap1[0]])]
        C = assemble_chain(per, 1.0, 1, spec2(Wv))
        self.assertLessEqual(fd, C * dn * 1.01)


@unittest.skipUnless(os.path.exists(RESULTS), "audit_results.json not present")
class TestShippedResults(unittest.TestCase):
    """Consistency of the shipped audit_results.json: provenance fields,
    derived arithmetic, and (via TestRawRecords) summary statistics
    recomputed from the raw arrays."""

    @classmethod
    def setUpClass(cls):
        with open(RESULTS) as f:
            cls.R = json.load(f)

    def test_environment_record(self):
        env = self.R["environment"]
        for key in ("model", "revision", "device", "python", "torch",
                    "transformers", "numpy", "huggingface_hub", "seed",
                    "modeling_file_sha256", "audit_commit"):
            self.assertIn(key, env)
        self.assertTrue(env["transformers"].startswith("4.56"),
                        "shipped run must use the pinned transformers 4.56.x")
        self.assertEqual(len(env["revision"]), 40)
        self.assertEqual(len(env["modeling_file_sha256"]), 64)

    def test_twirl_constants_reproducible(self):
        C = C_theta_of(build_freqs())
        self.assertAlmostEqual(self.R["twirl_analytic"]["C_theta"] / C, 1.0,
                               places=9)
        self.assertAlmostEqual(self.R["twirl_analytic"]["C_over_1024"],
                               C / 1024, delta=abs(C) * 1e-9)

    def test_chain_coefficient_reproducible(self):
        ch = self.R["chain"]
        self.assertEqual(len(ch["per_layer"]), NUM_LAYERS)
        for rec in ch["per_layer"]:
            for key in ("WO", "Llayer", "LLN1", "LLN2", "Lffn", "Lattn"):
                self.assertIn(key, rec)
        total = assemble_chain(ch["per_layer"],
                               self.R["budget"]["headfac"]["max"],
                               NUM_HEADS, ch["Wvocab"])
        self.assertAlmostEqual(ch["end_to_end_coefficient"] / total, 1.0,
                               places=9)
        eps_rec = self.R["budget"]["eps_spec_hypothetical"]
        eps = eps_rec["value"]
        self.assertAlmostEqual(eps, 2.0 ** -24 * eps_rec["G_F_max"],
                               delta=abs(eps) * 1e-9)
        # the radius must be labeled hypothetical, never a measurement
        self.assertIn("hypothetical", eps_rec["label"])
        self.assertAlmostEqual(ch["bound_at_eps_spec"] / (total * eps), 1.0,
                               places=9)
        min_margin = min(d["min_margin"] for d in self.R["decode"])
        self.assertEqual(ch["min_margin_over_decode"], min_margin)
        self.assertEqual(ch["margin_criterion"]["factor"], 2)
        self.assertAlmostEqual(ch["orders_above_half_margin"],
                               math.log10(total * eps / (min_margin / 2)),
                               places=6)

    def test_decode_fidelity_invariants(self):
        for d in self.R["decode"]:
            self.assertTrue(d["argmax_agree"])
            self.assertEqual(d["n_disagree"], 0)
            # corrected sufficient criterion: 2B < margin
            self.assertTrue(margin_ok(d["max_dlogit"], d["min_margin"]))
            self.assertTrue(d["margin_criterion_2B"])
            self.assertIn("epsG_rel_rms_max_over_steps", d)
            self.assertIn("epsG_absmax_over_steps", d)

    def test_continuation_texts_are_neutral_records(self):
        for d in self.R["decode"]:
            self.assertIsInstance(d["continuation"], str)
            self.assertGreater(len(d["continuation"].strip()), 0)
            self.assertIn("rep_stats", d)
        sc = self.R["stochastic_continuations"]
        self.assertEqual(len(sc), 10)   # 5 prompts x 2 seeds
        for c in sc:
            self.assertIn("rep_stats", c)

    def test_order_parameter_invariants(self):
        opar = self.R["order_parameter"]
        self.assertEqual(opar["GLM"]["rms"]["max"], 0.0)
        # the correctly indexed A_P (per-head exponents) is invariant
        # to ~1e-12, not bitwise -- a head-broadcast mis-indexing reads
        # "exactly 0" here, and asserting that would re-encode the bug
        self.assertLess(opar["AP"]["rms"]["max"], 1e-11)
        self.assertLess(opar["A"]["rms"]["max"], 1e-7)
        self.assertLess(opar["ALM"]["rms"]["max"], 1e-7)

    def test_trained_vs_init_present(self):
        # the trained model's A collapses to numerical rank one on every
        # instance; the random-init controls are reported alongside so
        # the trained structure is attributable by comparison, not
        # assumption -- under BOTH initialization laws, each labeled
        # (the from_config control samples the HF-port _init_weights,
        # Xavier-uniform W/P/a, which is NOT the native SOC training
        # law, Xavier-normal W/P/a)
        self.assertIn("spectra_init", self.R)
        self.assertIn("spectra_init_native", self.R)
        self.assertIn("HF-port", self.R["spectra_init"]["note"])
        self.assertIn("native", self.R["spectra_init_native"]["note"])
        self.assertEqual(self.R["spectra"]["A_numrank"]["max"], 1)

    def test_twirl_labels(self):
        # two statistics, each under its exact label -- per-instance
        # ratios over all (prompt, layer) pairs, and the prompt-pooled
        # per-layer aggregates whose range is over the layers only
        # (ratios do not commute with pooling)
        tw = self.R["twirl_empirical"]
        self.assertIn("per_instance", tw)
        self.assertIn("prompt_pooled_per_layer", tw)
        self.assertIn("per-(prompt, layer)", tw["per_instance"]["label"])
        self.assertIn("prompt-pooled", tw["prompt_pooled_per_layer"]["label"])

    def test_pair_counts_recorded(self):
        # retained counts are what the summaries are computed over;
        # attempted counts are recorded separately so a report can
        # never quote the attempted count as the retained one
        n = self.R["jacobians_v2"]["n_pairs"]
        for key in ("within", "cross", "attempted_within", "attempted_cross"):
            self.assertIn(key, n)
        self.assertLessEqual(n["within"], n["attempted_within"])
        self.assertLessEqual(n["cross"], n["attempted_cross"])

    def test_order_parameter_note_names_symmetrized(self):
        # the signed normalization is the SYMMETRIZED adaptation
        # 0.5*(|mu1|+|mu2|), not the source papers' |mu_1| convention;
        # the results file must say so
        self.assertIn("symmetrized", self.R["order_parameter_note"].lower())

    def test_dag_losses_present(self):
        for k in ("ALM", "AP", "GLM"):
            self.assertIn(k, self.R["dag_losses"])
        # Two floors, named separately: the
        # NOTEARS obstruction h = tr e^N - dk has floor dk*eps^2
        # ~ 6.4e-17, while the IMPLEMENTED normalized log loss
        # log(tr e^N / dk) = log(1 + h/dk) has floor log(1 + eps^2)
        # ~ eps^2 = 1e-18.  dk*eps^2 is the floor of h, NOT of this
        # loss. A normalized trace can round to one; the implemented
        # shifted logarithmic reduction can also lose a small residual
        # through rounding/cancellation, without underflow of that
        # positive increment. The ALM record therefore allows zero.
        # A_P's retained measured minimum is positive. This measurement
        # does not establish a universal architectural floor for A_P.
        self.assertGreaterEqual(self.R["dag_losses"]["ALM"]["min"], 0.0)
        self.assertGreater(self.R["dag_losses"]["AP"]["min"], 0.0)


@unittest.skipUnless(os.path.exists(RESULTS) and os.path.exists(RAWFILE),
                     "shipped results/raw records not present")
class TestRawRecords(unittest.TestCase):
    """The JSON's summary numbers must be recomputable from the raw
    arrays in audit_raw.npz, and the NPZ hash recorded in the JSON must
    match the file."""

    @classmethod
    def setUpClass(cls):
        with open(RESULTS) as f:
            cls.R = json.load(f)
        cls.raw = np.load(RAWFILE)

    def test_npz_hash_matches(self):
        self.assertEqual(self.R["raw_records"]["sha256"], sha256_file(RAWFILE))

    def assertStatsEqual(self, summary, arr):
        st = stats(arr)
        for k in ("min", "median", "max"):
            expect = summary[k]
            if expect == 0.0:
                self.assertEqual(st[k], 0.0)
            else:
                self.assertAlmostEqual(st[k] / expect, 1.0, places=9)

    def test_spectra_from_raw(self):
        A = self.raw["A_spectra_trained"]
        self.assertEqual(A.shape[0], N_INSTANCES)
        self.assertStatsEqual(self.R["spectra"]["A_s2_over_s1"], A[:, 1] / A[:, 0])
        W = self.raw["ALM_spectra_trained"]
        self.assertStatsEqual(self.R["spectra"]["ALM_s2_over_s1"], W[:, 1] / W[:, 0])
        ranks = np.array([numrank(s) for s in A])
        self.assertStatsEqual(self.R["spectra"]["A_numrank"], ranks)

    def test_ln_from_raw(self):
        self.assertStatsEqual(self.R["ln_scale_error"]["direct_rel"],
                              self.raw["ln_direct_rel"])
        self.assertStatsEqual(self.R["ln_scale_error"]["rowvar_DS"],
                              self.raw["rowvar_DS"])

    def test_jacobians_from_raw(self):
        self.assertStatsEqual(self.R["jacobians_v2"]["composite_smax"],
                              self.raw["jac_comp_smax"].ravel())
        self.assertStatsEqual(self.R["jacobians_v2"]["pairwise_within"],
                              self.raw["pair_within"])
        self.assertStatsEqual(self.R["jacobians_v2"]["pairwise_cross"],
                              self.raw["pair_cross"])

    def test_twirl_from_raw(self):
        # raw energies are per (prompt, layer); both the
        # per-instance ratios and the prompt-pooled per-layer
        # aggregates must recompute from them
        Eu = self.raw["twirl_E_unrot"]; Er = self.raw["twirl_E_rot"]
        omega = self.raw["twirl_omega"]
        self.assertEqual(Eu.ndim, 4)
        NPp, Lc = Eu.shape[:2]
        zero = np.abs(omega[:, None] - omega[None, :]) < 1e-12
        off = ~zero
        tw = self.R["twirl_empirical"]
        inst = [1 - Er[pi, li][off].sum() / Eu[pi, li][off].sum()
                for pi in range(NPp) for li in range(Lc)]
        self.assertEqual(tw["per_instance"]["n_instances"], NPp * Lc)
        self.assertStatsEqual(tw["per_instance"]["offcomm_suppression"], inst)
        Eul = Eu.sum(axis=0); Erl = Er.sum(axis=0)
        pool = [1 - Erl[li][off].sum() / Eul[li][off].sum()
                for li in range(Lc)]
        self.assertStatsEqual(
            tw["prompt_pooled_per_layer"]["offcomm_suppression"], pool)

    def test_pair_counts_match_raw(self):
        n = self.R["jacobians_v2"]["n_pairs"]
        self.assertEqual(n["within"], self.raw["pair_within"].shape[0])
        self.assertEqual(n["cross"], self.raw["pair_cross"].shape[0])

    def test_init_native_spectra_from_raw(self):
        A = self.raw["A_spectra_init_native"]
        self.assertEqual(A.shape[0], N_INSTANCES)
        ranks = np.array([numrank(s) for s in A])
        self.assertStatsEqual(self.R["spectra_init_native"]["A_numrank"],
                              ranks)

    def test_epsG_all_steps_from_raw(self):
        eg = self.raw["epsG_steps"]
        self.assertEqual(eg.shape, (4, 48, NUM_LAYERS))
        for pi, d in enumerate(self.R["decode"]):
            self.assertEqual(d["epsG_rel_rms_max_over_steps"], eg[pi].max())
        dl = self.raw["dlogit_steps"]; mg = self.raw["margin_steps"]
        for pi, d in enumerate(self.R["decode"]):
            self.assertEqual(d["max_dlogit"], dl[pi].max())
            self.assertEqual(d["min_margin"], mg[pi].min())

    def test_dag_from_raw(self):
        # all three tensors' summaries are recomputed from the raw
        # arrays, not only G_LM's
        for key in ("ALM", "AP", "GLM"):
            self.assertStatsEqual(self.R["dag_losses"][key],
                                  self.raw[f"dag_{key}"])
        self.assertEqual(self.raw["dag_AP"].shape[0], N_INSTANCES)

    def test_ap_dag_consistent_with_correct_construction(self):
        # The correctly indexed AP DAG distribution is far from the
        # broadcast-mis-indexed one: its median exceeds the mis-indexed
        # construction's median (185.6) several-fold.  Guards against a
        # silent regression to any broadcast pattern, which reproduced
        # the smaller values.
        med = float(np.median(self.raw["dag_AP"]))
        self.assertGreater(med, 500.0)

    def test_glm_commutant_residual_from_raw(self):
        cr = self.raw["glm_comm_res"]
        self.assertEqual(cr.shape, (N_PROMPTS, NUM_LAYERS, NUM_HEADS))
        self.assertTrue(((cr >= 0.0) & (cr <= 1.0)).all())
        self.assertStatsEqual(self.R["glm_commutant_residual"]["all"],
                              cr.ravel())

    def test_order_parameter_from_raw(self):
        opar = self.R["order_parameter"]
        for key in ("A", "ALM", "AP", "GLM"):
            self.assertStatsEqual(opar[key]["rms"], self.raw[f"op_{key}_rms"])


if __name__ == "__main__":
    unittest.main()

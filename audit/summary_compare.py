"""Field-sensitive binary64 policy for retained sequential summaries.

Inputs stay hash-exact. This is an error budget for derived reductions,
not a license to change a scientific decision. See NUMERICAL_COMPARISON.md.
"""
from dataclasses import dataclass
import hashlib
import math
import re

U = 2.0 ** -53


def gamma(n):
    if not 0 <= n * U < 0.01:
        raise ValueError('reduction outside the admitted error model')
    return n * U / (1 - n * U)


@dataclass(frozen=True)
class ReductionBudget:
    tokens: int
    items: int
    token_scale: float
    mean_scale: float
    candidate_length: int = 1
    candidate_scale: float = 0.0
    choices: int = 1

    def tolerance(self, quantity):
        # Two computations are compared. gamma bounds naive summation, hence
        # also the shorter error path of pairwise reductions. Four units of
        # roundoff per exponential are an explicit supported-libm assumption.
        token = 4 * U * self.token_scale
        candidate = 4 * gamma(self.candidate_length + 2) * self.candidate_scale
        mass = 2 * candidate + 4 * gamma(self.choices + 8)
        absolute = {
            'nll': 0.0,
            'nll_gap': 4 * gamma(self.tokens + 2) * self.mean_scale,
            'token_gap': token,
            'candidate_gap': candidate,
            'margin': candidate,
            'probability_mass': mass,
            'probability_gap': 2 * mass + 4 * U,
        }
        if quantity not in absolute:
            raise AssertionError('unknown quantity class: ' + quantity)
        return absolute[quantity], 2 * gamma(max(self.tokens, self.items) + 8)


def quantity_for(path):
    """Closed registry. Unknown numeric or categorical fields are rejected."""
    if len(path) == 1:
        key = path[0]
        if key in {'windows','window_len','tokens_scored','n_items','n_candidates',
                   'n_scored_tokens','n_true_choices','raw_argmax_changes',
                   'norm_argmax_changes','discordant_pairs','pairs_total',
                   'items_gap_above_half_margin','diagnostic_raw_argmax_changes',
                   'diagnostic_norm_argmax_changes'}:
            return 'integer'
        if key == 'single_token_exact_coincidence':
            return 'boolean'
        if key in {'acc_block','acc_sequential','acc_norm_block','acc_norm_sequential'}:
            return 'exact_ratio'
        if key in {'sequential_nll_per_token','block_ce_per_token'}:
            return 'nll'
        if key == 'nll_gap_per_token':
            return 'nll_gap'
        if key in {'mc2_block_mean','mc2_sequential_mean'}:
            return 'probability_mass'
        if key == 'mc2_mean_gap':
            return 'probability_gap'
    if len(path) == 2:
        group, field = path
        classes = {'per_token_absdiff':'token_gap', 'per_candidate_absdiff':'candidate_gap',
                   'block_margin':'margin', 'per_item_metric_absdiff':'probability_gap',
                   'per_item_metric_signed':'probability_gap','per_token_signed':'token_gap'}
        if group in classes:
            if field in {'min','max','median','mean'}:
                return classes[group]
            if field == 'n' and group not in {'per_item_metric_signed','per_token_signed'}:
                return 'integer'
            if group == 'per_token_signed' and field == 'frac_sequential_worse':
                return 'exact_ratio'
    if len(path) >= 3 and path[0] in {'by_candidate_length','by_position'} and isinstance(path[1], int):
        if len(path) == 3:
            if path[2] == 'n' and path[0] == 'by_candidate_length':
                return 'integer'
            if path[2] in {'median','max'}:
                return 'candidate_gap' if path[0] == 'by_candidate_length' else 'token_gap'
        if len(path) == 4 and path[2] in {'token_length','positions'} and path[3] in {0,1}:
            return 'integer'
    raise AssertionError('unregistered field: ' + '/'.join(map(str,path)))


def compare_summary(expected, actual, budget, path=(), records=None):
    records = [] if records is None else records
    where = '/'.join(map(str, path))
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError('key set differs: ' + where)
        for key in sorted(expected):
            compare_summary(expected[key], actual[key], budget, path+(key,), records)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError('shape differs: ' + where)
        for i, (x,y) in enumerate(zip(expected,actual)):
            compare_summary(x,y,budget,path+(i,),records)
    else:
        kind = quantity_for(path)
        # bool is a subclass of int: handle it before every numeric operation.
        if kind == 'boolean':
            if type(expected) is not bool or type(actual) is not bool or expected != actual:
                raise AssertionError('Boolean decision differs: ' + where)
        elif kind == 'integer':
            if type(expected) is not int or type(actual) is not int or expected != actual:
                raise AssertionError('count or index differs: ' + where)
        elif kind == 'category':
            if type(expected) is not str or type(actual) is not str or expected != actual:
                raise AssertionError('category differs: ' + where)
        else:
            if type(expected) is not float or type(actual) is not float:
                raise AssertionError('floating type differs: ' + where)
            if not math.isfinite(expected) or not math.isfinite(actual):
                raise AssertionError('nonfinite value: ' + where)
            if kind == 'exact_ratio':
                if expected != actual:
                    raise AssertionError('count-derived fraction differs: ' + where)
                atol = rtol = 0.0
            else:
                atol, rtol = budget.tolerance(kind)
                if (expected > 0) - (expected < 0) != (actual > 0) - (actual < 0):
                    raise AssertionError('sign or exact-zero decision differs: ' + where)
            error = abs(actual-expected)
            limit = atol + rtol*max(abs(actual),abs(expected))
            if error > limit:
                raise AssertionError(f'{where}: error {error!r} exceeds {limit!r}')
            records.append(dict(path=where, quantity=kind, absolute_error=error,
                                absolute_budget=atol, relative_budget=rtol,
                                limit=limit, printed_9_digits_unchanged=
                                format(actual,'.9g') == format(expected,'.9g')))
    return records


def verify_hash(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not re.fullmatch('[0-9a-f]{64}',expected) or actual != expected:
        raise AssertionError('immutable input SHA-256 mismatch')

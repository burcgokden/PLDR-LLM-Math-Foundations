"""Material-change rejection tests for the portable sequential-summary audit."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from summary_compare import ReductionBudget, compare_summary, verify_hash

B=ReductionBudget(100,10,30.0,4.0,24,200.0,13)
PATHS={'nll':('sequential_nll_per_token',), 'nll_gap':('nll_gap_per_token',),
       'token_gap':('per_token_absdiff','mean'),
       'candidate_gap':('per_candidate_absdiff','mean'),
       'margin':('block_margin','mean'),
       'probability_mass':('mc2_block_mean',), 'probability_gap':('mc2_mean_gap',)}

class TestSummaryPolicy(unittest.TestCase):
    def test_all_registered_numeric_classes_reject_just_beyond_budget(self):
        for kind,path in PATHS.items():
            with self.subTest(kind=kind):
                a,r=B.tolerance(kind); x=1.0
                y=math.nextafter((x+a)/(1-r),math.inf)
                with self.assertRaises(AssertionError):compare_summary(x,y,B,path)

    def test_all_registered_numeric_classes_accept_bounded_roundoff(self):
        for kind,path in PATHS.items():
            with self.subTest(kind=kind):
                compare_summary(1.0,math.nextafter(1.0,math.inf),B,path)

    def test_missing_key_and_changed_shape(self):
        for actual in ({}, {'n_items':10,'extra':0}):
            with self.assertRaises(AssertionError):compare_summary({'n_items':10},actual,B)
        with self.assertRaises(AssertionError):
            compare_summary({'by_position':[{'positions':[0,32],'median':1.0}]}, {'by_position':[]},B)

    def test_unknown_field_rejected_even_on_both_sides(self):
        with self.assertRaises(AssertionError):compare_summary({'unregistered':1.0},{'unregistered':1.0},B)

    def test_counts_are_exact_and_boolean_is_not_integer(self):
        for x,y,path in [(10,11,('n_items',)),(True,1,('single_token_exact_coincidence',)),
                         (True,False,('single_token_exact_coincidence',)),(1,True,('n_items',))]:
            with self.subTest(path=path,x=x,y=y),self.assertRaises(AssertionError):compare_summary(x,y,B,path)

    def test_no_nonfinite_sentinels(self):
        for y in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(AssertionError):compare_summary(1.0,y,B,('mc2_mean_gap',))

    def test_sign_zero_and_count_decisions_remain_exact(self):
        for x,y,path in [(1e-20,-1e-20,('mc2_mean_gap',)),(0.0,1e-25,('mc2_mean_gap',)),
                         (0.5,math.nextafter(0.5,1.0),('acc_block',)),
                         (0,1,('items_gap_above_half_margin',))]:
            with self.assertRaises(AssertionError):compare_summary(x,y,B,path)

    def test_changed_hash_and_category(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'input.json';p.write_text('{"role":"assessment"}')
            pin=hashlib.sha256(p.read_bytes()).hexdigest();verify_hash(p,pin)
            p.write_text('{"role":"calibration"}')
            with self.assertRaises(AssertionError):verify_hash(p,pin)

    def test_small_reported_gap_cannot_be_erased(self):
        with self.assertRaises(AssertionError):
            compare_summary(-8.451783515829934e-10,0.0,B,('mc2_mean_gap',))

    def test_individual_decision_manifest_rejects_ties_and_types(self):
        from reconstruct_summaries import verify_decisions
        old={'maximizers':[[0]],'gate':True}
        for new in ({'maximizers':[[0,1]],'gate':True}, {'maximizers':[[0]],'gate':1}):
            with self.assertRaises(AssertionError):verify_decisions(old,new)

if __name__=='__main__':unittest.main()

"""Regression tests for filter_sd_maxcost cost lookup.

Each intervention key is uniquely a knock-out OR knock-in candidate, so the cost must be
looked up by dict membership, not by the sign of the (decompressed) design value. Previously a
gene knock-out whose value came out non-negative was routed into the knock-in cost dict and
raised KeyError (reproducible with compress=False + gene KOs + max_cost on gene-name models).
"""
from straindesign.networktools import filter_sd_maxcost


def test_gene_ko_with_nonnegative_value_no_keyerror():
    # 'g8' is a knock-out candidate (in kocost, not kicost). Its design value is +1.0 (the
    # encoding that used to route the lookup into kicost -> KeyError).
    kocost = {'g8': 1.0, 'r1': 1.0}
    kicost = {}
    out = filter_sd_maxcost([{'g8': 1.0, 'r1': -1.0}], 3, kocost, kicost)
    assert len(out) == 1                      # cost = 1(g8) + 1(r1) = 2 <= 3 -> kept


def test_cost_counted_by_membership_drops_over_budget():
    kocost = {'g8': 3.0, 'r1': 1.0}
    kicost = {}
    # cost = 3 + 1 = 4 > max_cost 2 -> dropped (proves g8's cost is actually counted)
    assert filter_sd_maxcost([{'g8': 1.0, 'r1': -1.0}], 2, kocost, kicost) == []


def test_zero_value_intervention_is_free():
    # value 0 = unused knock-in candidate -> contributes no cost
    kocost = {'r1': 1.0}
    kicost = {'g3': 5.0}
    out = filter_sd_maxcost([{'r1': -1.0, 'g3': 0}], 1, kocost, kicost)
    assert len(out) == 1                      # only r1 counts (cost 1), g3=0 ignored


def test_knock_in_cost_still_works():
    kocost = {'r1': 1.0}
    kicost = {'g3': 1.0}
    out = filter_sd_maxcost([{'r1': -1.0, 'g3': 1.0}], 3, kocost, kicost)
    assert len(out) == 1                      # KO r1 + KI g3 = 2 <= 3
    assert filter_sd_maxcost([{'r1': -1.0, 'g3': 1.0}], 1.5, kocost, kicost) == []

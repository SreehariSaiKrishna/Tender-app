"""Tests for app.processing.eligibility - the rule-based domain/value-range
gates behind `eligibility_match` (see app.processing.deduplicator,
app.intelligence.document_summarizer, scripts/backfill_eligibility.py, all
of which build on these same functions).
"""
from __future__ import annotations

from app.processing.eligibility import (
    compute_eligibility_match,
    emd_in_range,
    tender_fee_in_range,
    tender_value_in_range,
)

# --- compute_eligibility_match ---------------------------------------------


def test_compute_eligibility_match_finds_whole_word_keyword():
    assert compute_eligibility_match("Tender for Digital Marketing services", ["digital marketing"])


def test_compute_eligibility_match_is_case_insensitive():
    assert compute_eligibility_match("DIGITAL MARKETING campaign", ["digital marketing"])


def test_compute_eligibility_match_requires_word_boundary():
    """A short keyword like "ai" must match the standalone word, not as a
    substring of an unrelated word (e.g. "domain", "captain")."""
    assert compute_eligibility_match("AI-based platform", ["ai"])
    assert not compute_eligibility_match("Domain registration services", ["ai"])


def test_compute_eligibility_match_false_for_no_match():
    assert not compute_eligibility_match("Tender for road construction", ["digital marketing", "edtech"])


def test_compute_eligibility_match_false_for_empty_text():
    assert not compute_eligibility_match("", ["digital marketing"])


def test_compute_eligibility_match_false_for_no_keywords():
    assert not compute_eligibility_match("Digital Marketing tender", [])


# --- tender_value_in_range / emd_in_range / tender_fee_in_range -----------
#
# All three share the same contract, fixed after a live regression
# (2026-09-23): a tender rarely discloses its value, EMD and fee all in the
# same listing, so requiring every one of them to be BOTH disclosed AND in
# range (the original design) left almost nothing eligible even for
# obviously domain-relevant tenders. Undisclosed now means "doesn't rule it
# out" - only a value that IS disclosed must actually be in range.


def test_tender_value_in_range_true_within_bounds():
    assert tender_value_in_range(1_00_00_000.0)  # ₹1 Cr, within ₹70L-5Cr


def test_tender_value_in_range_false_below_minimum():
    assert not tender_value_in_range(10_00_000.0)  # ₹10L, below ₹70L floor


def test_tender_value_in_range_false_above_maximum():
    assert not tender_value_in_range(10_00_00_000.0)  # ₹10Cr, above ₹5Cr ceiling


def test_tender_value_in_range_true_at_exact_boundaries():
    assert tender_value_in_range(70_00_000.0)
    assert tender_value_in_range(5_00_00_000.0)


def test_tender_value_in_range_true_when_undisclosed():
    assert tender_value_in_range(None)


def test_emd_in_range_true_within_bounds():
    assert emd_in_range(75_000.0)


def test_emd_in_range_false_above_maximum():
    assert not emd_in_range(2_00_000.0)


def test_emd_in_range_true_when_undisclosed():
    assert emd_in_range(None)


def test_tender_fee_in_range_true_within_bounds():
    assert tender_fee_in_range(5_000.0)


def test_tender_fee_in_range_false_above_maximum():
    assert not tender_fee_in_range(20_000.0)


def test_tender_fee_in_range_true_when_undisclosed():
    assert tender_fee_in_range(None)


def test_a_domain_relevant_tender_with_only_partial_disclosure_is_eligible():
    """Regression case: EMD in range, tender value and fee both undisclosed
    - must still pass every range check (each undisclosed field yields
    True), the exact shape that zeroed out eligibility_match in production
    before this fix.
    """
    assert tender_value_in_range(None)
    assert emd_in_range(54_980.0)
    assert tender_fee_in_range(None)

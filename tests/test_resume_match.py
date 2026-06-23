"""Tests for resume verification logic in llm/resume_match.py.

These specifically guard against a bug found during development: the LLM
sometimes echoes a generic, transcript-derived statement back as if it were
a "well substantiated" resume claim, when really it just shares common
domain vocabulary (quantitative, finance, machine learning, data, etc.)
with the resume rather than corresponding to any specific resume line.
"""

from llm.resume_match import _verify_well_substantiated


def test_generic_domain_overlap_does_not_count_as_resume_support():
    # This is the real claim that incorrectly passed verification during
    # project testing — it's just a paraphrase of the transcript, not an
    # actual resume-specific claim.
    transcript = (
        "I want to get into quantitative finance in NFO or machine "
        "learning and data analytics."
    )
    resume_text = (
        "Skills: Python, machine learning, quantitative finance, data "
        "analytics, financial modeling. "
        "Project: Engineered an end-to-end ETL pipeline fetching "
        "time-series data via REST APIs."
    )
    result = {
        "well_substantiated": [
            "Want to get into quantitative finance in NFO or machine "
            "learning and data analytics"
        ],
        "unsubstantiated_claims": [],
        "summary": "test",
    }

    verified = _verify_well_substantiated(result, transcript, resume_text)

    assert verified["well_substantiated"] == []
    assert len(verified["unsubstantiated_claims"]) == 1


def test_specific_named_claim_with_real_support_is_kept():
    # A genuine match: the candidate actually named the specific project
    # and it appears the same way in both transcript and resume.
    transcript = (
        "I built an ETL pipeline using yfinance to pull years of "
        "time-series stock data."
    )
    resume_text = (
        "Engineered an end-to-end ETL pipeline fetching 5+ years of "
        "time-series data via REST APIs (yfinance)."
    )
    result = {
        "well_substantiated": ["Built an ETL pipeline using yfinance for time-series data"],
        "unsubstantiated_claims": [],
        "summary": "test",
    }

    verified = _verify_well_substantiated(result, transcript, resume_text)

    assert len(verified["well_substantiated"]) == 1
    assert len(verified["unsubstantiated_claims"]) == 0


def test_claim_with_no_transcript_support_is_demoted():
    # A claim that's genuinely on the resume but never mentioned at all
    # in the transcript should be demoted regardless of resume overlap.
    transcript = "I enjoy reading books and watching YouTube videos."
    resume_text = (
        "Built a high-performance backtesting simulation engine with "
        "71.2% prediction accuracy."
    )
    result = {
        "well_substantiated": ["Built a high-performance backtesting simulation engine"],
        "unsubstantiated_claims": [],
        "summary": "test",
    }

    verified = _verify_well_substantiated(result, transcript, resume_text)

    assert verified["well_substantiated"] == []
    assert len(verified["unsubstantiated_claims"]) == 1
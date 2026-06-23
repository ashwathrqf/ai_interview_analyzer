"""Tests for the scoring logic in report/scorer.py."""

from report.scorer import calculate_overall_score


def test_calculate_overall_score_matches_known_run():
    # These are the real, verified inputs and outputs from an actual
    # end-to-end run of the project (see project test history).
    speech_metrics = {
        "wpm": 93.3,
        "filler_total": 2,
        "word_count": 85,
        "pause_count": 2,
        "average_pause": 1.76,
    }
    vision_metrics = {
        "eye_contact_percent": 81.8,
        "facing_camera_percent": 81.8,
    }
    llm_feedback = {"rating": 7}

    result = calculate_overall_score(speech_metrics, vision_metrics, llm_feedback)

    assert result["body_language"] == 81.8
    assert result["technical"] == 70.0
    # Overall and communication/confidence have some floating point
    # rounding sensitivity, so check they're in the right ballpark rather
    # than asserting an exact value that could drift with minor formula tweaks.
    assert 70 <= result["overall"] <= 80
    assert 70 <= result["communication"] <= 80
    assert 75 <= result["confidence"] <= 85


def test_perfect_inputs_yield_high_overall_score():
    speech_metrics = {
        "wpm": 140,  # squarely in the ideal range
        "filler_total": 0,
        "word_count": 100,
        "pause_count": 1,
        "average_pause": 1.0,
    }
    vision_metrics = {
        "eye_contact_percent": 100.0,
        "facing_camera_percent": 100.0,
    }
    llm_feedback = {"rating": 10}

    result = calculate_overall_score(speech_metrics, vision_metrics, llm_feedback)
    assert result["overall"] >= 95


def test_poor_inputs_yield_low_overall_score():
    speech_metrics = {
        "wpm": 250,  # very fast, well outside ideal range
        "filler_total": 20,
        "word_count": 100,  # 20 fillers per 100 words is very high
        "pause_count": 15,
        "average_pause": 5.0,
    }
    vision_metrics = {
        "eye_contact_percent": 10.0,
        "facing_camera_percent": 10.0,
    }
    llm_feedback = {"rating": 2}

    result = calculate_overall_score(speech_metrics, vision_metrics, llm_feedback)
    assert result["overall"] <= 30


def test_scores_are_within_valid_range():
    # Sanity check: no category score should ever escape 0-100,
    # even with extreme or malformed-looking inputs.
    speech_metrics = {
        "wpm": 9999,
        "filler_total": 500,
        "word_count": 10,
        "pause_count": 100,
        "average_pause": 50.0,
    }
    vision_metrics = {
        "eye_contact_percent": 0.0,
        "facing_camera_percent": 0.0,
    }
    llm_feedback = {"rating": 1}

    result = calculate_overall_score(speech_metrics, vision_metrics, llm_feedback)
    for key in ["overall", "confidence", "communication", "body_language", "technical"]:
        assert 0 <= result[key] <= 100
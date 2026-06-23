"""Combine all metrics into category and overall scores.

Scoring design:
- Communication (25%): WPM closeness to ideal range, filler word rate, pauses
- Body Language (20%): eye contact %, head pose (facing camera %)
- Confidence (25%): derived blend of communication steadiness + body language
- Technical (30%): the LLM's content-quality rating, scaled to 0-100

Overall is the weighted average of the four category scores.
"""

from utils.config import IDEAL_WPM_MIN, IDEAL_WPM_MAX

WEIGHTS = {
    "communication": 0.25,
    "body_language": 0.20,
    "confidence": 0.25,
    "technical": 0.30,
}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _wpm_score(wpm: float) -> float:
    """Score WPM closeness to the ideal range. 100 if inside the range,
    decreasing the further outside it, reaching 0 at 60 WPM away from the
    nearest boundary.
    """
    if IDEAL_WPM_MIN <= wpm <= IDEAL_WPM_MAX:
        return 100.0

    if wpm < IDEAL_WPM_MIN:
        distance = IDEAL_WPM_MIN - wpm
    else:
        distance = wpm - IDEAL_WPM_MAX

    # Linear falloff: lose ~1.67 points per WPM away from the range,
    # reaching 0 at 60 WPM past the boundary.
    score = 100.0 - (distance * (100.0 / 60.0))
    return _clamp(score)


def _filler_score(filler_total: int, word_count: int) -> float:
    """Score based on filler words per 100 words spoken. 0 fillers per 100
    words = 100. Score drops as filler rate climbs, reaching 0 around 15
    fillers per 100 words (a genuinely heavy filler habit).
    """
    if word_count == 0:
        return 100.0
    fillers_per_100 = (filler_total / word_count) * 100
    score = 100.0 - (fillers_per_100 * (100.0 / 15.0))
    return _clamp(score)


def _pause_score(pause_count: int, average_pause: float) -> float:
    """Score pause behavior. A few natural pauses are fine; the score drops
    if pauses are very frequent or very long on average.
    """
    # Frequency: 0-3 pauses is natural, score drops past that
    frequency_penalty = max(0, pause_count - 3) * 8

    # Length: average pauses under 2s are natural, longer pauses cost more
    length_penalty = max(0, average_pause - 2.0) * 20

    score = 100.0 - frequency_penalty - length_penalty
    return _clamp(score)


def _communication_score(speech_metrics: dict) -> float:
    wpm_score = _wpm_score(speech_metrics["wpm"])
    filler_score = _filler_score(
        speech_metrics["filler_total"], speech_metrics["word_count"]
    )
    pause_score = _pause_score(
        speech_metrics["pause_count"], speech_metrics["average_pause"]
    )
    # Weighted within the category: WPM and fillers matter most
    return _clamp(wpm_score * 0.4 + filler_score * 0.4 + pause_score * 0.2)


def _body_language_score(vision_metrics: dict) -> float:
    eye_contact = vision_metrics["eye_contact_percent"]
    facing_camera = vision_metrics["facing_camera_percent"]
    # Eye contact and head pose are highly correlated by construction
    # (eye contact requires facing camera), so weight eye contact higher
    # since it's the stricter, more informative signal.
    return _clamp(eye_contact * 0.65 + facing_camera * 0.35)


def _confidence_score(communication_score: float, body_language_score: float, filler_score_raw: float) -> float:
    """Confidence is a derived blend: steady communication + good body
    language reads as confident. Filler rate is weighted in again directly
    since hesitation markers are a particularly strong confidence signal.
    """
    return _clamp(
        communication_score * 0.4 + body_language_score * 0.3 + filler_score_raw * 0.3
    )


def _technical_score(llm_rating: int) -> float:
    """Scale the LLM's 1-10 rating to a 0-100 score."""
    return _clamp(llm_rating * 10.0)


def calculate_overall_score(speech_metrics: dict, vision_metrics: dict, llm_feedback: dict) -> dict:
    """Combine metrics into a final score breakdown.

    Args:
        speech_metrics: dict with keys "wpm", "filler_total", "word_count",
            "pause_count", "average_pause"
        vision_metrics: dict with keys "eye_contact_percent", "facing_camera_percent"
        llm_feedback: dict with key "rating" (1-10, from evaluate_answer)

    Returns:
        dict like:
        {
            "overall": 78.4,
            "confidence": 75.0,
            "communication": 70.2,
            "body_language": 81.8,
            "technical": 70.0
        }
    """
    filler_score_raw = _filler_score(
        speech_metrics["filler_total"], speech_metrics["word_count"]
    )

    communication = _communication_score(speech_metrics)
    body_language = _body_language_score(vision_metrics)
    confidence = _confidence_score(communication, body_language, filler_score_raw)
    technical = _technical_score(llm_feedback["rating"])

    overall = (
        communication * WEIGHTS["communication"]
        + body_language * WEIGHTS["body_language"]
        + confidence * WEIGHTS["confidence"]
        + technical * WEIGHTS["technical"]
    )

    return {
        "overall": round(overall, 1),
        "confidence": round(confidence, 1),
        "communication": round(communication, 1),
        "body_language": round(body_language, 1),
        "technical": round(technical, 1),
    }


if __name__ == "__main__":
    # Quick manual test using real values from earlier pipeline steps.
    sample_speech_metrics = {
        "wpm": 93.3,
        "filler_total": 2,
        "word_count": 85,  # approximate word count of the test transcript
        "pause_count": 2,
        "average_pause": 1.76,
    }
    sample_vision_metrics = {
        "eye_contact_percent": 81.8,
        "facing_camera_percent": 81.8,
    }
    sample_llm_feedback = {"rating": 7}

    result = calculate_overall_score(sample_speech_metrics, sample_vision_metrics, sample_llm_feedback)
    print(result)